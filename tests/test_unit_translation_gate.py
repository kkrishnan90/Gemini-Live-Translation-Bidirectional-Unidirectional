"""Unit tests for SileroProsodiaTranslationGate and build_translation_live_config (Seam 2).

Tests:
1. Agent Platform (formerly known as Vertex AI) wire serialization of `build_translation_live_config` via `google-genai` `_live_converters`.
2. Streaming Source language PCM through `SileroProsodiaTranslationGate` -> zero calls to
   `activity_start`, `stream_pcm`, or `activity_end` (100% deterministic silence).
3. Streaming Target language PCM through `SileroProsodiaTranslationGate` -> exact call order
   `activity_start -> stream_pcm (non-empty 16kHz PCM) -> activity_end`.
4. Mid-session language swap (`update_languages`) flips gate polarity immediately.
5. Short noise / silence frames never trigger empty manual-VAD turns.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import pytest
import soundfile as sf
from google import genai
from google.auth import credentials as auth_credentials
from google.genai import _live_converters as live_converters
from google.genai import types

from prosodia_lid_engine import ProsodiaLanguageGateEngine
from translation_gate import (
    SileroProsodiaTranslationGate,
    build_translation_live_config,
    sanitize_live_transcript,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
CKPT_PATH = PROJECT_DIR / "model" / "best.pt"
BINS_PATH = PROJECT_DIR / "model" / "bins.json"
SAMPLE_DIR = PROJECT_DIR / "sample_audio"


@pytest.fixture(scope="module")
def lid_engine() -> ProsodiaLanguageGateEngine:
    return ProsodiaLanguageGateEngine(ckpt_path=CKPT_PATH, bins_path=BINS_PATH)


def load_pcm_with_trailing_silence(filename: str, trailing_silence_s: float = 0.6) -> bytes:
    wav, sr = sf.read(str(SAMPLE_DIR / filename), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    assert sr == 16000
    pad_tail = np.zeros(int(trailing_silence_s * sr), dtype=np.float32)
    full = np.concatenate([wav, pad_tail])
    pcm_i16 = (np.clip(full, -1.0, 1.0) * 32767.0).astype(np.int16)
    return pcm_i16.tobytes()


def test_build_translation_live_config_vertex_wire_serialization() -> None:
    """Verify LiveConnectConfig serializes cleanly for Agent Platform (formerly known as Vertex AI) and sets manual VAD + Source SpeechConfig."""
    cfg = build_translation_live_config(
        source_lang="en-US",
        target_lang="hi-IN",
        voice_name="Kore",
    )
    assert cfg.response_modalities == [types.Modality.AUDIO]
    assert cfg.speech_config.language_code == "en-US"
    assert cfg.realtime_input_config.automatic_activity_detection.disabled is True
    assert (
        cfg.realtime_input_config.turn_coverage
        == types.TurnCoverage.TURN_INCLUDES_ONLY_ACTIVITY
    )

    creds = auth_credentials.AnonymousCredentials()
    client = genai.Client(
        vertexai=True,
        project="example-gcp-project",
        location="us-central1",
        credentials=creds,
    )
    params = types.LiveConnectParameters(model="gemini-3.8-live", config=cfg)
    vertex_setup = live_converters._LiveConnectParameters_to_vertex(
        client._api_client, params
    )
    setup_dict = vertex_setup["setup"]
    assert "publishers/google/models/gemini-3.8-live" in setup_dict["model"]
    assert setup_dict["generationConfig"]["speechConfig"]["languageCode"] == "en-US"
    assert (
        setup_dict["realtimeInputConfig"].automatic_activity_detection.disabled
        is True
    )
    si_text = setup_dict["systemInstruction"]["parts"][0]["text"]
    assert "Hindi" in si_text
    assert "English" in si_text
    assert "RESPOND UNMISTAKABLY IN English" in si_text
    assert "Direct Voice of the Speaker" in si_text


def test_sanitize_live_transcript_strips_s2st_cot_and_tags() -> None:
    """Verify sanitize_live_transcript strips S2ST CoT prefixes (`🧠[en]<not ta>...✨`) and `<partial>` tags."""
    raw = "🧠[en]<not ta>Hey, how are you doing?✨ஹேய், நீ எப்படி<partial>"
    assert sanitize_live_transcript(raw) == "ஹேய், நீ எப்படி"

    # Streaming partial CoT before `✨` arrives must return empty string so UI never flashes raw tags
    assert sanitize_live_transcript("🧠[en]<not ta>Hey, how") == ""

    # Clean normal translation passes through untouched
    clean_ta = "ஹேய், நீங்க எப்படி இருக்கீங்க?"
    assert sanitize_live_transcript(clean_ta) == clean_ta


@pytest.mark.asyncio
async def test_gate_suppresses_source_language_completely(
    lid_engine: ProsodiaLanguageGateEngine,
) -> None:
    """When Source=en-US and Target=hi-IN, streaming English speech must NOT call Gemini at all."""
    events: list[str] = []
    streamed_bytes = bytearray()
    gate_decisions: list[dict[str, Any]] = []

    async def on_start() -> None:
        events.append("activity_start")

    async def on_pcm(data: bytes) -> None:
        events.append("pcm")
        streamed_bytes.extend(data)

    async def on_end() -> None:
        events.append("activity_end")

    async def on_decision(payload: dict[str, Any]) -> None:
        gate_decisions.append(payload)

    async def on_telemetry(payload: dict[str, Any]) -> None:
        pass

    gate = SileroProsodiaTranslationGate(
        engine=lid_engine,
        source_lang="en-US",
        target_lang="hi-IN",
        on_trigger_activity_start=on_start,
        on_stream_pcm_to_gemini=on_pcm,
        on_trigger_activity_end=on_end,
        on_gate_decision=on_decision,
        on_vad_telemetry=on_telemetry,
    )

    pcm_en = load_pcm_with_trailing_silence("human_query_en.wav")
    await gate.process_pcm_bytes(pcm_en)

    # Zero calls to Gemini Live API!
    assert events == []
    assert len(streamed_bytes) == 0
    assert len(gate_decisions) == 1
    decision = gate_decisions[0]
    assert decision["detected_role"] == "source"
    assert decision["should_translate"] is False
    assert decision["stay_silent"] is True
    assert decision["turn_triggered"] is False
    assert decision["bytes_forwarded_to_gemini"] == 0
    assert decision["gate_action"] == "STAY_SILENT_SOURCE_SPOKEN"


@pytest.mark.asyncio
async def test_gate_forwards_target_language_with_manual_vad_framing(
    lid_engine: ProsodiaLanguageGateEngine,
) -> None:
    """When Source=en-US and Target=hi-IN, streaming Hindi speech triggers activity_start -> pcm -> activity_end."""
    events: list[str] = []
    streamed_bytes = bytearray()
    gate_decisions: list[dict[str, Any]] = []

    async def on_start() -> None:
        events.append("activity_start")

    async def on_pcm(data: bytes) -> None:
        events.append("pcm")
        streamed_bytes.extend(data)

    async def on_end() -> None:
        events.append("activity_end")

    async def on_decision(payload: dict[str, Any]) -> None:
        gate_decisions.append(payload)

    async def on_telemetry(payload: dict[str, Any]) -> None:
        pass

    gate = SileroProsodiaTranslationGate(
        engine=lid_engine,
        source_lang="en-US",
        target_lang="hi-IN",
        on_trigger_activity_start=on_start,
        on_stream_pcm_to_gemini=on_pcm,
        on_trigger_activity_end=on_end,
        on_gate_decision=on_decision,
        on_vad_telemetry=on_telemetry,
    )

    pcm_hi = load_pcm_with_trailing_silence("human_sentence_hindi.wav")
    await gate.process_pcm_bytes(pcm_hi)

    assert events == ["activity_start", "pcm", "activity_end"]
    assert len(streamed_bytes) > 16000  # > 0.5s of 16kHz int16 PCM
    assert len(gate_decisions) == 1
    decision = gate_decisions[0]
    assert decision["detected_role"] == "target"
    assert decision["should_translate"] is True
    assert decision["stay_silent"] is False
    assert decision["turn_triggered"] is True
    assert decision["bytes_forwarded_to_gemini"] == len(streamed_bytes)
    assert decision["gate_action"] == "TRANSLATE_TARGET_TO_SOURCE"


@pytest.mark.asyncio
async def test_gate_language_swap_flips_behavior_immediately(
    lid_engine: ProsodiaLanguageGateEngine,
) -> None:
    """Swapping Source and Target mid-session flips which language translates vs stays silent."""
    events: list[str] = []
    gate_decisions: list[dict[str, Any]] = []

    async def on_start() -> None:
        events.append("activity_start")

    async def on_pcm(data: bytes) -> None:
        events.append("pcm")

    async def on_end() -> None:
        events.append("activity_end")

    async def on_decision(payload: dict[str, Any]) -> None:
        gate_decisions.append(payload)

    async def on_telemetry(payload: dict[str, Any]) -> None:
        pass

    gate = SileroProsodiaTranslationGate(
        engine=lid_engine,
        source_lang="en-US",
        target_lang="hi-IN",
        on_trigger_activity_start=on_start,
        on_stream_pcm_to_gemini=on_pcm,
        on_trigger_activity_end=on_end,
        on_gate_decision=on_decision,
        on_vad_telemetry=on_telemetry,
    )

    # Swap to Source=hi-IN, Target=en-US
    gate.update_languages("hi-IN", "en-US")

    # Now Hindi is Source -> must stay silent!
    pcm_hi = load_pcm_with_trailing_silence("human_sentence_hindi.wav")
    await gate.process_pcm_bytes(pcm_hi)
    assert events == []
    assert gate_decisions[-1]["stay_silent"] is True

    # And English is Target -> must translate!
    pcm_en = load_pcm_with_trailing_silence("human_query_en.wav")
    await gate.process_pcm_bytes(pcm_en)
    assert events == ["activity_start", "pcm", "activity_end"]
    assert gate_decisions[-1]["should_translate"] is True


@pytest.mark.asyncio
async def test_gate_early_prefix_streaming_lid_on_realtime_mic_chunks(
    lid_engine: ProsodiaLanguageGateEngine,
) -> None:
    """When 32ms mic chunks arrive in real time, Early-Prefix LID fires mid-utterance (~1.15s) before speech ends."""
    events: list[str] = []
    gate_decisions: list[dict[str, Any]] = []

    async def on_start() -> None:
        events.append("activity_start")

    async def on_pcm(data: bytes) -> None:
        events.append("pcm")

    async def on_end() -> None:
        events.append("activity_end")

    async def on_decision(payload: dict[str, Any]) -> None:
        gate_decisions.append(payload)

    async def on_telemetry(payload: dict[str, Any]) -> None:
        pass

    gate = SileroProsodiaTranslationGate(
        engine=lid_engine,
        source_lang="en-US",
        target_lang="ta-IN",
        on_trigger_activity_start=on_start,
        on_stream_pcm_to_gemini=on_pcm,
        on_trigger_activity_end=on_end,
        on_gate_decision=on_decision,
        on_vad_telemetry=on_telemetry,
        vad_cfg={"enable_early_stream_to_gemini": True},
    )

    pcm_ta = load_pcm_with_trailing_silence("human_sentence_tamil.wav")
    chunk_bytes = 1024  # 512 int16 samples = 32ms real-time mic chunk
    early_decision_fired_before_end = False

    for offset in range(0, len(pcm_ta), chunk_bytes):
        await gate.process_pcm_bytes(pcm_ta[offset : offset + chunk_bytes])
        if gate_decisions and "activity_end" not in events:
            early_decision_fired_before_end = True

    assert early_decision_fired_before_end is True
    assert events[0] == "activity_start"
    assert events[-1] == "activity_end"
    assert gate_decisions[0]["should_translate"] is True
    assert gate_decisions[0]["trigger_source"] == "early_prefix_lid"

