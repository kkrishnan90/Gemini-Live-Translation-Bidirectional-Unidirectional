"""FastAPI + WebSocket Server for Prosodia Language Gate + Gemini 3.8 Flash Live Translation.

Architecture:
1. Authenticates via Google Cloud Application Default Credentials (ADC) or `GOOGLE_APPLICATION_CREDENTIALS`
   on Agent Platform (formerly known as Vertex AI, `location="us-central1"` by default).
2. Connects to `gemini-3.8-live` via `google-genai` SDK (`2.31.0`) with:
   - `AutomaticActivityDetection(disabled=True)` (Manual VAD)
   - `TurnCoverage.TURN_INCLUDES_ONLY_ACTIVITY`
   - `SpeechConfig(language_code=source_lang, ...)` so translations are spoken in the user's Source language
   - Unidirectional interpreter `system_instruction` (`Target -> Source`)
3. Segments incoming 16 kHz PCM speech with Silero VAD and evaluates each utterance with
   `ProsodiaLanguageGateEngine` (`openai/whisper-small` shared encoder + `ProsodiaModel` `best.pt`):
   - Target language spoken -> sends `activity_start -> audio -> activity_end` to `gemini-3.8-live`,
     which translates the utterance into the Source language.
   - Source language spoken -> sends zero audio and zero activity signals to `gemini-3.8-live`,
     guaranteeing 100% silence.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
from pathlib import Path
import time
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types
import google.auth
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2 import service_account
import librosa
import numpy as np
import soundfile as sf

from prosodia.features import SAMPLE_RATE
from prosodia_lid_engine import (
    ProsodiaLanguageGateEngine,
    SUPPORTED_LANGUAGES,
    validate_language_pair,
)
from translation_gate import (
    DEFAULT_VOICE_NAME,
    GATE_EXECUTOR,
    SileroProsodiaTranslationGate,
    build_translation_live_config,
    sanitize_live_transcript,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("gemini_live_translation")

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
MODEL_DIR = APP_DIR / "model"
CKPT_PATH = MODEL_DIR / "best.pt"
BINS_PATH = MODEL_DIR / "bins.json"
SAMPLE_AUDIO_DIR = APP_DIR / "sample_audio"

CREDENTIALS_PATH = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1").strip()
MODEL_ID = os.environ.get("GEMINI_LIVE_MODEL", "gemini-3.8-live").strip()
DEFAULT_SOURCE_LANG = "en-US"
DEFAULT_TARGET_LANG = "hi-IN"

SAMPLE_CATALOG: dict[str, dict[str, Any]] = {
    "english": {
        "sample_id": "english",
        "filename": "human_query_en.wav",
        "label": "English Speech Sample (en-US)",
        "language_code": "en-US",
        "iso": "en",
        "description": "Real human English spoken query",
    },
    "hindi": {
        "sample_id": "hindi",
        "filename": "human_sentence_hindi.wav",
        "label": "Hindi Speech Sample (hi-IN)",
        "language_code": "hi-IN",
        "iso": "hi",
        "description": "Real human Hindi spoken sentence",
    },
    "bengali": {
        "sample_id": "bengali",
        "filename": "human_sentence_bengali.wav",
        "label": "Bengali Speech Sample (bn-BD)",
        "language_code": "bn-BD",
        "iso": "bn",
        "description": "Real human Bengali spoken sentence",
    },
    "tamil": {
        "sample_id": "tamil",
        "filename": "human_sentence_tamil.wav",
        "label": "Tamil Speech Sample (ta-IN)",
        "language_code": "ta-IN",
        "iso": "ta",
        "description": "Real human Tamil spoken sentence",
    },
    "english_interrupt": {
        "sample_id": "english_interrupt",
        "filename": "human_interrupt_en.wav",
        "label": "English Second Sentence (en-US)",
        "language_code": "en-US",
        "iso": "en",
        "description": "Real human English follow-up sentence",
    },
}

logger.info("Loading ProsodiaLanguageGateEngine from %s ...", CKPT_PATH)
PROSODIA_GATE_ENGINE = ProsodiaLanguageGateEngine(ckpt_path=CKPT_PATH, bins_path=BINS_PATH)
logger.info(
    "ProsodiaLanguageGateEngine ready on device=%s (epoch=%s)",
    PROSODIA_GATE_ENGINE.device,
    PROSODIA_GATE_ENGINE.checkpoint_epoch,
)


def create_genai_client() -> genai.Client:
    """Create an authenticated Agent Platform (formerly known as Vertex AI) genai.Client via ADC or GOOGLE_APPLICATION_CREDENTIALS."""
    scopes = ["https://www.googleapis.com/auth/cloud-platform"]
    creds = None
    detected_project = ""

    if CREDENTIALS_PATH and Path(CREDENTIALS_PATH).exists():
        try:
            cand = service_account.Credentials.from_service_account_file(
                CREDENTIALS_PATH,
                scopes=scopes,
            )
            cand.refresh(GoogleAuthRequest())
            creds = cand
            detected_project = getattr(cand, "project_id", "") or ""
            logger.info("Authenticated with GOOGLE_APPLICATION_CREDENTIALS")
        except Exception as exc:
            logger.warning(
                "Service account file failed refresh (%s); falling back to Application Default Credentials.",
                exc,
            )

    if creds is None:
        creds, adc_proj = google.auth.default(scopes=scopes)
        creds.refresh(GoogleAuthRequest())
        detected_project = adc_proj or ""
        logger.info("Authenticated with Application Default Credentials")

    resolved_project = os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip() or detected_project
    if not resolved_project:
        raise RuntimeError(
            "GOOGLE_CLOUD_PROJECT environment variable is required when project cannot be inferred from ADC."
        )

    return genai.Client(
        vertexai=True,
        project=resolved_project,
        location=LOCATION,
        credentials=creds,
    )


GENAI_CLIENT = create_genai_client()


def load_sample_waveform(sample_id: str) -> tuple[np.ndarray, dict[str, Any]]:
    if sample_id not in SAMPLE_CATALOG:
        raise HTTPException(status_code=404, detail=f"Unknown sample_id: {sample_id!r}")
    meta = SAMPLE_CATALOG[sample_id]
    wav_path = SAMPLE_AUDIO_DIR / meta["filename"]
    if not wav_path.exists():
        raise HTTPException(status_code=404, detail=f"Sample audio file missing: {wav_path}")
    wav, sr = sf.read(str(wav_path), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SAMPLE_RATE:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SAMPLE_RATE)
    return wav.astype(np.float32), meta


app = FastAPI(
    title="Gemini Live Translation (Unidirectional)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def get_health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_id": MODEL_ID,
        "platform": "Agent Platform (formerly known as Vertex AI)",
        "location": LOCATION,
        "google_genai_version": genai.__version__,
        "default_source_lang": DEFAULT_SOURCE_LANG,
        "default_target_lang": DEFAULT_TARGET_LANG,
        "default_voice_name": DEFAULT_VOICE_NAME,
        "vad_engine": "silero-vad + prosodia-lid-gate (AutomaticActivityDetection disabled=True)",
        "prosodia_device": str(PROSODIA_GATE_ENGINE.device),
        "prosodia_checkpoint_epoch": PROSODIA_GATE_ENGINE.checkpoint_epoch,
        "supported_languages": SUPPORTED_LANGUAGES,
    }


@app.get("/api/samples")
async def list_samples() -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for sample_id, meta in SAMPLE_CATALOG.items():
        wav_path = SAMPLE_AUDIO_DIR / meta["filename"]
        dur_s = 0.0
        if wav_path.exists():
            info = sf.info(str(wav_path))
            dur_s = round(float(info.duration), 2)
        items.append(
            {
                **meta,
                "duration_s": dur_s,
                "audio_url": f"/api/samples/{sample_id}/audio",
            }
        )
    return {"samples": items}


@app.get("/api/samples/{sample_id}/audio")
async def get_sample_audio(sample_id: str) -> FileResponse:
    if sample_id not in SAMPLE_CATALOG:
        raise HTTPException(status_code=404, detail=f"Unknown sample_id: {sample_id!r}")
    wav_path = SAMPLE_AUDIO_DIR / SAMPLE_CATALOG[sample_id]["filename"]
    if not wav_path.exists():
        raise HTTPException(status_code=404, detail=f"Audio file not found: {wav_path}")
    return FileResponse(str(wav_path), media_type="audio/wav")


@app.post("/api/analyze-sample")
async def analyze_sample_endpoint(payload: dict[str, Any]) -> JSONResponse:
    sample_id = str(payload.get("sample_id", "hindi"))
    source_lang = str(payload.get("source_lang", DEFAULT_SOURCE_LANG))
    target_lang = str(payload.get("target_lang", DEFAULT_TARGET_LANG))
    bidirectional = bool(payload.get("bidirectional", False))
    try:
        validate_language_pair(source_lang, target_lang)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    wav, meta = load_sample_waveform(sample_id)
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        GATE_EXECUTOR,
        PROSODIA_GATE_ENGINE.analyze_utterance,
        wav,
        source_lang,
        target_lang,
        None,
        0.55,
        bidirectional,
    )
    result["sample"] = meta
    return JSONResponse(result)


@app.post("/api/analyze-upload")
async def analyze_upload_endpoint(
    audio: UploadFile = File(...),
    source_lang: str = Form(DEFAULT_SOURCE_LANG),
    target_lang: str = Form(DEFAULT_TARGET_LANG),
    bidirectional: bool = Form(False),
) -> JSONResponse:
    try:
        validate_language_pair(source_lang, target_lang)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    raw_bytes = await audio.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Empty audio upload")
    try:
        wav, _ = librosa.load(io.BytesIO(raw_bytes), sr=SAMPLE_RATE, mono=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not decode audio: {exc}") from exc

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        GATE_EXECUTOR,
        PROSODIA_GATE_ENGINE.analyze_utterance,
        wav.astype(np.float32),
        source_lang,
        target_lang,
        None,
        0.55,
        bidirectional,
    )
    return JSONResponse(result)


@app.websocket("/ws/translate")
async def websocket_translate_endpoint(ws: WebSocket) -> None:
    """Bidirectional WebSocket bridging browser audio, Prosodia Language Gate, and `gemini-3.8-live`."""
    await ws.accept()
    send_lock = asyncio.Lock()

    async def ws_send_json(payload: dict[str, Any]) -> None:
        async with send_lock:
            try:
                await ws.send_text(json.dumps(payload))
            except Exception:
                pass

    async def ws_send_bytes(data: bytes) -> None:
        async with send_lock:
            try:
                await ws.send_bytes(data)
            except Exception:
                pass

    session_config: dict[str, Any] = {
        "source_lang": DEFAULT_SOURCE_LANG,
        "target_lang": DEFAULT_TARGET_LANG,
        "voice_name": DEFAULT_VOICE_NAME,
        "bidirectional": False,
        "custom_instruction": "",
        "silero_vad": {
            "threshold": 0.50,
            "min_speech_duration_ms": 120,
            "min_silence_duration_ms": 280,
            "speech_pad_ms": 160,
            "early_lid_ms": 1152,
            "enable_early_stream_to_gemini": False,
            "confidence_threshold": 0.55,
        },
    }

    try:
        first_msg = await asyncio.wait_for(ws.receive(), timeout=1.5)
        if "text" in first_msg and first_msg["text"]:
            data = json.loads(first_msg["text"])
            if data.get("type") in ("setup", "update_config", "update_languages"):
                cfg_in = data.get("config", data)
                src_cand = str(cfg_in.get("source_lang", session_config["source_lang"]))
                tgt_cand = str(cfg_in.get("target_lang", session_config["target_lang"]))
                validate_language_pair(src_cand, tgt_cand)
                session_config["source_lang"] = src_cand
                session_config["target_lang"] = tgt_cand
                if "voice_name" in cfg_in:
                    session_config["voice_name"] = str(cfg_in["voice_name"])
                if "bidirectional" in cfg_in:
                    session_config["bidirectional"] = bool(cfg_in["bidirectional"])
                if "custom_instruction" in cfg_in:
                    session_config["custom_instruction"] = str(cfg_in["custom_instruction"])
                if "silero_vad" in cfg_in and isinstance(cfg_in["silero_vad"], dict):
                    session_config["silero_vad"].update(cfg_in["silero_vad"])
    except asyncio.TimeoutError:
        pass
    except WebSocketDisconnect:
        return
    except Exception as exc:
        logger.warning("Initial setup warning: %s", exc)
        await ws_send_json({"type": "error", "message": f"Invalid language config: {exc}"})
        return

    reconnect_requested = True
    client_disconnected = False

    while reconnect_requested and not client_disconnected:
        reconnect_requested = False
        source_lang = session_config["source_lang"]
        target_lang = session_config["target_lang"]
        voice_name = session_config["voice_name"]
        bidirectional = bool(session_config.get("bidirectional", False))
        custom_instruction = session_config["custom_instruction"]

        live_connect_config = build_translation_live_config(
            source_lang=source_lang,
            target_lang=target_lang,
            voice_name=voice_name,
            custom_instruction=custom_instruction,
            bidirectional=bidirectional,
        )

        await ws_send_json(
            {
                "type": "session_connecting",
                "model_id": MODEL_ID,
                "platform": "Agent Platform (formerly known as Vertex AI)",
                "location": LOCATION,
                "config": session_config,
            }
        )

        try:
            async with GENAI_CLIENT.aio.live.connect(
                model=MODEL_ID,
                config=live_connect_config,
            ) as live_session:
                logger.info(
                    "Connected to %s on Agent Platform (Source=%s, Target=%s, Voice=%s)",
                    MODEL_ID,
                    source_lang,
                    target_lang,
                    voice_name,
                )

                await ws_send_json(
                    {
                        "type": "session_ready",
                        "model_id": MODEL_ID,
                        "platform": "Agent Platform (formerly known as Vertex AI)",
                        "location": LOCATION,
                        "config": session_config,
                        "source_language_info": SUPPORTED_LANGUAGES[source_lang],
                        "target_language_info": SUPPORTED_LANGUAGES[target_lang],
                    }
                )

                stop_inner = asyncio.Event()
                current_turn_user_text: list[str] = []
                current_turn_agent_text: list[str] = []

                async def trigger_activity_start() -> None:
                    current_turn_user_text.clear()
                    current_turn_agent_text.clear()
                    await live_session.send_realtime_input(
                        activity_start=types.ActivityStart()
                    )
                    await ws_send_json(
                        {
                            "type": "gemini_activity_signal",
                            "signal": "activity_start",
                            "timestamp": time.time(),
                        }
                    )

                async def stream_pcm_to_gemini(pcm_bytes: bytes) -> None:
                    chunk_sz = 4096
                    for i in range(0, len(pcm_bytes), chunk_sz):
                        await live_session.send_realtime_input(
                            audio=types.Blob(
                                data=pcm_bytes[i : i + chunk_sz],
                                mime_type="audio/pcm;rate=16000",
                            )
                        )

                async def trigger_activity_end() -> None:
                    await live_session.send_realtime_input(
                        activity_end=types.ActivityEnd()
                    )
                    await ws_send_json(
                        {
                            "type": "gemini_activity_signal",
                            "signal": "activity_end",
                            "timestamp": time.time(),
                        }
                    )

                gate = SileroProsodiaTranslationGate(
                    engine=PROSODIA_GATE_ENGINE,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    on_trigger_activity_start=trigger_activity_start,
                    on_stream_pcm_to_gemini=stream_pcm_to_gemini,
                    on_trigger_activity_end=trigger_activity_end,
                    on_gate_decision=ws_send_json,
                    on_vad_telemetry=ws_send_json,
                    vad_cfg=session_config["silero_vad"],
                    bidirectional=bidirectional,
                )

                async def receive_from_gemini() -> None:
                    try:
                        while not stop_inner.is_set():
                            async for response in live_session.receive():
                                if stop_inner.is_set():
                                    break

                                if response.usage_metadata is not None:
                                    um = response.usage_metadata
                                    await ws_send_json(
                                        {
                                            "type": "usage_metadata",
                                            "prompt_token_count": getattr(um, "prompt_token_count", 0),
                                            "response_token_count": getattr(um, "candidates_token_count", 0)
                                            or getattr(um, "response_token_count", 0),
                                            "total_token_count": getattr(um, "total_token_count", 0),
                                        }
                                    )

                                sc = response.server_content
                                if sc is None:
                                    continue

                                if sc.interrupted:
                                    await ws_send_json(
                                        {
                                            "type": "interrupted",
                                            "timestamp": time.time(),
                                        }
                                    )

                                if sc.input_transcription is not None:
                                    in_tx = sc.input_transcription
                                    tx_text = in_tx.text or ""
                                    tx_finished = bool(in_tx.finished)
                                    if tx_text:
                                        current_turn_user_text.append(tx_text)
                                        await ws_send_json(
                                            {
                                                "type": "user_transcript",
                                                "delta": tx_text,
                                                "text": "".join(current_turn_user_text).strip(),
                                                "finished": tx_finished,
                                                "language_code": target_lang,
                                            }
                                        )

                                if sc.output_transcription is not None:
                                    out_tx = sc.output_transcription
                                    tx_text = out_tx.text or ""
                                    tx_finished = bool(out_tx.finished)
                                    if tx_text:
                                        current_turn_agent_text.append(tx_text)
                                        cleaned_agent_text = sanitize_live_transcript(
                                            "".join(current_turn_agent_text)
                                        )
                                        if cleaned_agent_text:
                                            await ws_send_json(
                                                {
                                                    "type": "agent_transcript",
                                                    "delta": tx_text,
                                                    "text": cleaned_agent_text,
                                                    "finished": tx_finished,
                                                    "language_code": source_lang,
                                                }
                                            )

                                if sc.model_turn and sc.model_turn.parts:
                                    for part in sc.model_turn.parts:
                                        if part.inline_data and part.inline_data.data:
                                            await ws_send_bytes(part.inline_data.data)
                                        elif part.text:
                                            cleaned_part = sanitize_live_transcript(part.text)
                                            if cleaned_part:
                                                await ws_send_json(
                                                    {
                                                        "type": "agent_text_part",
                                                        "text": cleaned_part,
                                                    }
                                                )

                                if sc.turn_complete:
                                    final_user = "".join(current_turn_user_text).strip()
                                    final_agent = sanitize_live_transcript(
                                        "".join(current_turn_agent_text)
                                    )
                                    await ws_send_json(
                                        {
                                            "type": "turn_complete",
                                            "user_transcript": final_user,
                                            "agent_transcript": final_agent,
                                            "source_lang": source_lang,
                                            "target_lang": target_lang,
                                        }
                                    )
                                    current_turn_user_text.clear()
                                    current_turn_agent_text.clear()

                    except asyncio.CancelledError:
                        pass
                    except Exception as exc:
                        if not stop_inner.is_set():
                            logger.error("Gemini receive loop error: %s", exc)
                            await ws_send_json(
                                {
                                    "type": "error",
                                    "message": f"Gemini stream error: {exc}",
                                }
                            )
                            stop_inner.set()

                async def receive_from_browser() -> None:
                    nonlocal reconnect_requested, client_disconnected
                    try:
                        while not stop_inner.is_set():
                            msg = await ws.receive()
                            if msg.get("type") == "websocket.disconnect":
                                client_disconnected = True
                                stop_inner.set()
                                break

                            if "bytes" in msg and msg["bytes"]:
                                await gate.process_pcm_bytes(msg["bytes"])

                            elif "text" in msg and msg["text"]:
                                data = json.loads(msg["text"])
                                mtype = data.get("type")

                                if mtype in ("update_config", "update_languages"):
                                    cfg_in = data.get("config", data)
                                    src_cand = str(cfg_in.get("source_lang", session_config["source_lang"]))
                                    tgt_cand = str(cfg_in.get("target_lang", session_config["target_lang"]))
                                    try:
                                        validate_language_pair(src_cand, tgt_cand)
                                    except ValueError as exc:
                                        await ws_send_json({"type": "error", "message": str(exc)})
                                        continue
                                    session_config["source_lang"] = src_cand
                                    session_config["target_lang"] = tgt_cand
                                    if "voice_name" in cfg_in:
                                        session_config["voice_name"] = str(cfg_in["voice_name"])
                                    if "bidirectional" in cfg_in:
                                        session_config["bidirectional"] = bool(cfg_in["bidirectional"])
                                    if "custom_instruction" in cfg_in:
                                        session_config["custom_instruction"] = str(cfg_in["custom_instruction"])
                                    if "silero_vad" in cfg_in and isinstance(cfg_in["silero_vad"], dict):
                                        session_config["silero_vad"].update(cfg_in["silero_vad"])
                                    reconnect_requested = True
                                    stop_inner.set()
                                    break

                                elif mtype == "audio_stream_end":
                                    await gate.flush_pending()

                                elif mtype == "inject_sample_audio":
                                    sample_id = str(data.get("sample_id", "hindi"))
                                    try:
                                        wav, meta = load_sample_waveform(sample_id)
                                    except HTTPException as exc:
                                        await ws_send_json({"type": "error", "message": str(exc.detail)})
                                        continue

                                    await ws_send_json(
                                        {
                                            "type": "sample_injected",
                                            "sample": meta,
                                            "source_lang": source_lang,
                                            "target_lang": target_lang,
                                        }
                                    )
                                    lead_sil = np.zeros(int(0.06 * SAMPLE_RATE), dtype=np.float32)
                                    trail_sil = np.zeros(int(0.48 * SAMPLE_RATE), dtype=np.float32)
                                    full_wav = np.concatenate([lead_sil, np.clip(wav, -1.0, 1.0), trail_sil])
                                    pcm16 = (full_wav * 32767.0).astype(np.int16).tobytes()
                                    await gate.process_pcm_bytes(pcm16)
                                    await gate.flush_pending()

                    except WebSocketDisconnect:
                        client_disconnected = True
                        stop_inner.set()
                    except asyncio.CancelledError:
                        pass
                    except Exception as exc:
                        logger.error("Browser receive loop error: %s", exc)
                        client_disconnected = True
                        stop_inner.set()

                gem_task = asyncio.create_task(receive_from_gemini())
                brw_task = asyncio.create_task(receive_from_browser())

                await stop_inner.wait()
                gem_task.cancel()
                brw_task.cancel()
                await asyncio.gather(gem_task, brw_task, return_exceptions=True)

        except Exception as exc:
            logger.exception("Failed to connect or run Gemini 3.8 Live session: %s", exc)
            await ws_send_json(
                {
                    "type": "error",
                    "message": f"Gemini 3.8 Live connection failed: {exc}",
                }
            )
            break


STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def serve_index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8092"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
