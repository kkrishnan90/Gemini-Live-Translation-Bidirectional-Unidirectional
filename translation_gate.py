"""Silero VAD + Prosodia Unidirectional Language Gate & GEAP LiveConnectConfig Builder.

High-Performance Streaming Gate Architecture:
1. `build_translation_live_config`: Builds `types.LiveConnectConfig` for `gemini-3.8-live` on
   Agent Platform (formerly known as Vertex AI) with:
   - `AutomaticActivityDetection(disabled=True)` (Manual VAD)
   - `TurnCoverage.TURN_INCLUDES_ONLY_ACTIVITY`
   - `SpeechConfig(language_code=source_lang, ...)` so translations are spoken in Source language
   - Strict unidirectional interpreter `system_instruction` (`Target -> Source`)
2. `SileroProsodiaTranslationGate`:
   - Segments speech with Silero VAD (512-sample / 32ms windows at 16 kHz) with fast `260 ms`
     end-of-speech silence detection.
   - **Early-Prefix Streaming LID (`~1.15s`)**: During real-time microphone speech, as soon as
     `1.15s` (36 active frames) of speech is accumulated, evaluates `ProsodiaLanguageGateEngine`
     (~120 ms) while the speaker is still talking:
     * If Target language (`should_translate=True`, `P(target) >= 0.70`): immediately triggers
       `activity_start`, flushes the `1.15s` prefix to `gemini-3.8-live`, emits `gate_decision` to
       the UI, streams all subsequent speech frames live to `gemini-3.8-live`, and sends
       `activity_end` the instant the speaker pauses (zero post-speech LID delay).
     * If Source language (`stay_silent=True`, `P(source) >= 0.70`): immediately emits
       `gate_decision` (`STAY_SILENT_SOURCE_SPOKEN`) to the UI at `1.15s` and suppresses the rest
       of the utterance with zero bytes sent to `gemini-3.8-live`.
"""

from __future__ import annotations

import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import re
import time
from typing import Any, Awaitable, Callable
import uuid

from google.genai import types
import numpy as np
from silero_vad import load_silero_vad
import torch

from prosodia.features import SAMPLE_RATE
from prosodia_lid_engine import (
    _TORCH_WORKER as GATE_EXECUTOR,
    OPTIMAL_TORCH_THREADS,
    ProsodiaLanguageGateEngine,
    validate_language_pair,
)

# Counteract silero_vad/model.py's import-time `torch.set_num_threads(1)` side-effect
torch.set_num_threads(OPTIMAL_TORCH_THREADS)

SILERO_WINDOW_SAMPLES = 512  # 32ms at 16,000 Hz
DEFAULT_VOICE_NAME = "Kore"

_S2ST_SPLIT_RE = re.compile(r"^.*?✨\s*", re.DOTALL)
_S2ST_BRAIN_PREFIX_RE = re.compile(
    r"🧠\s*(?:\[[a-zA-Z]{2,3}(?:[-_][a-zA-Z]{2,4})?\])?\s*(?:<[^>]*>)?",
    re.DOTALL,
)
_S2ST_CONTROL_TAGS_RE = re.compile(
    r"<(?:punting|partial|catch-up|no speech|not\s+[^>]*|to\s+[^>]*|already\s+[^>]*)>"
    r"|\{(?:cont|pause)\}"
    r"|[✨🧠📝💬]"
)
_MULTI_SPACE_RE = re.compile(r"\s+")


def sanitize_live_transcript(raw_text: str) -> str:
    """Strip internal S2ST training CoT prefixes (`🧠[xx]<not yy>...✨`) and control tags (`<partial>`)."""
    if not raw_text:
        return ""
    text = raw_text
    if "✨" in text:
        text = _S2ST_SPLIT_RE.sub("", text, count=1)
    elif "🧠" in text:
        # While streaming before `✨` arrives, suppress the internal CoT prefix completely
        return ""
    text = _S2ST_BRAIN_PREFIX_RE.sub(" ", text)
    text = _S2ST_CONTROL_TAGS_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    return text.strip()


def build_translation_system_instruction(
    source_lang: str,
    target_lang: str,
    custom_instruction: str = "",
) -> str:
    """Build first-person interpreter system instruction for `gemini-3.8-live` (`target_lang -> source_lang`).

    Configures the model as the direct voice of the speaker in the listener's language while avoiding
    chunked dubbing trigger phrases so complete utterances are synthesized in a single uninterrupted turn.
    """
    src_info, tgt_info = validate_language_pair(source_lang, target_lang)
    src_name = src_info["name"]
    tgt_name = tgt_info["name"]

    base_si = (
        f"You are the direct voice of the speaker, conveying their exact intent to the listener in a clean, "
        f"concise, and natural, colloquial manner. You facilitate a direct conversation where the speaker talks in "
        f"{tgt_name} ({target_lang}) and the listener understands {src_name} ({source_lang}).\n\n"
        f"How you must operate:\n"
        f"1. Direct Voice of the Speaker:\n"
        f"   - Speak directly to the listener in the first person in {src_name} ({source_lang}), as if the speaker is speaking the listener's language.\n"
        f"   - Never speak in the third person or refer to the speaker as an outside party.\n"
        f"   - Never act like an AI assistant, mediator, or conversational third party.\n"
        f"   - Use natural, everyday colloquial speech and common spoken vocabulary as locals naturally talk in daily life, avoiding stiff, formal, or bookish translations.\n"
        f"   - Always speak the complete, full sentence from start to finish in a single uninterrupted response.\n"
        f"2. Efficient Interpretation & Intent Distillation:\n"
        f"   - Act as an efficient interpreter: convey the speaker's true intent to the other side in a concise manner, omitting unnecessary information, repetitions, and filler.\n"
        f"   - Discard all speech disfluencies: repetitions, stutters, false starts, vocal filler sounds, and pauses in any language.\n"
        f"   - Preserve all substantive details: quantities, numbers, names, constraints, questions, and preferences.\n"
        f"3. Strict Non-Interference & Identity Boundaries:\n"
        f"   - Pass-Through Question Handling: All questions and requests are directed to the human listener, never to you. "
        f"If the speaker asks a question or requests assistance, translate the question into {src_name} so the listener can respond. NEVER answer questions, provide status updates, or fulfill requests yourself.\n"
        f"   - Absolute Disclaimer Prohibition: NEVER output safety warnings, medical disclaimers, legal caveats, or health disclaimers under any circumstances.\n"
        f"   - Translate, Do Not Reply: Treat every input as source speech to translate into {src_name}. If the speaker says 'Thank you', translate 'Thank you' into {src_name}; never reply 'You're welcome'. Never generate unprompted greetings, polite pleasantries, or meta-commentary.\n"
        f"   - Never ask follow-up questions, seek clarification, or offer unprompted advice.\n"
        f"4. CRITICAL RULES:\n"
        f"   - When speech is in {tgt_name}, speak to the listener in {src_name}.\n"
        f"   - RESPOND UNMISTAKABLY IN {src_name} ({source_lang}).\n"
        f"   - Always output text and transcriptions strictly in the standard native script of {src_name} (never Roman or Latin script transliteration for non-Latin scripts, and never XML or angle-bracket tags).\n"
        f"   - Never answer questions, append disclaimers, generate counter-replies, or produce meta-commentary.\n"
        f"   - Output ONLY the clean, direct {src_name} speech to the listener."
    )
    if custom_instruction and custom_instruction.strip():
        return f"{base_si}\n\nADDITIONAL CONTEXT: {custom_instruction.strip()}"
    return base_si


def build_translation_live_config(
    source_lang: str,
    target_lang: str,
    voice_name: str = DEFAULT_VOICE_NAME,
    custom_instruction: str = "",
) -> types.LiveConnectConfig:
    """Build GEAP `LiveConnectConfig` for `gemini-3.8-live` with Manual VAD and Source language output."""
    validate_language_pair(source_lang, target_lang)
    system_instruction = build_translation_system_instruction(
        source_lang=source_lang,
        target_lang=target_lang,
        custom_instruction=custom_instruction,
    )

    realtime_input_config = types.RealtimeInputConfig(
        automatic_activity_detection=types.AutomaticActivityDetection(
            disabled=True,
        ),
        activity_handling=types.ActivityHandling.START_OF_ACTIVITY_INTERRUPTS,
        turn_coverage=types.TurnCoverage.TURN_INCLUDES_ONLY_ACTIVITY,
    )

    return types.LiveConnectConfig(
        response_modalities=[types.Modality.AUDIO],
        speech_config=types.SpeechConfig(
            language_code=source_lang,
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=voice_name or DEFAULT_VOICE_NAME,
                )
            ),
        ),
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
        realtime_input_config=realtime_input_config,
        system_instruction=types.Content(
            role="user",
            parts=[types.Part.from_text(text=system_instruction)],
        ),
    )


class SileroProsodiaTranslationGate:
    """Segments speech with Silero VAD and gates `gemini-3.8-live` turns using Prosodia LID."""

    def __init__(
        self,
        engine: ProsodiaLanguageGateEngine,
        source_lang: str,
        target_lang: str,
        on_trigger_activity_start: Callable[[], Awaitable[None]],
        on_stream_pcm_to_gemini: Callable[[bytes], Awaitable[None]],
        on_trigger_activity_end: Callable[[], Awaitable[None]],
        on_gate_decision: Callable[[dict[str, Any]], Awaitable[None]],
        on_vad_telemetry: Callable[[dict[str, Any]], Awaitable[None]],
        vad_cfg: dict[str, Any] | None = None,
    ) -> None:
        validate_language_pair(source_lang, target_lang)
        self.engine = engine
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.on_trigger_activity_start = on_trigger_activity_start
        self.on_stream_pcm_to_gemini = on_stream_pcm_to_gemini
        self.on_trigger_activity_end = on_trigger_activity_end
        self.on_gate_decision = on_gate_decision
        self.on_vad_telemetry = on_vad_telemetry

        cfg = vad_cfg or {}
        self.threshold = float(cfg.get("threshold", 0.50))
        self.min_speech_duration_ms = int(cfg.get("min_speech_duration_ms", 120))
        self.min_silence_duration_ms = int(cfg.get("min_silence_duration_ms", 280))
        self.speech_pad_ms = int(cfg.get("speech_pad_ms", 160))
        self.early_lid_ms = int(cfg.get("early_lid_ms", 1152))
        self.enable_early_stream_to_gemini = bool(cfg.get("enable_early_stream_to_gemini", False))
        self.confidence_threshold = float(cfg.get("confidence_threshold", 0.55))

        self.vad_model = load_silero_vad()
        self.reset()

    def update_languages(self, source_lang: str, target_lang: str) -> None:
        """Update Source and Target language pair and reset buffered VAD state."""
        validate_language_pair(source_lang, target_lang)
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.reset()

    def reset(self) -> None:
        self.vad_model.reset_states()
        self.rem_bytes = bytearray()
        self.state = "IDLE"
        pad_frames = max(2, int(round(self.speech_pad_ms / 32.0)))
        self.pre_roll: deque[np.ndarray] = deque(maxlen=pad_frames)
        self.buffered_frames: list[np.ndarray] = []
        self.speech_frames_count = 0
        self.silence_frames_count = 0
        self.frame_counter = 0
        self.early_lid_checked = False
        self.streamed_bytes_in_turn = 0

    async def process_pcm_bytes(self, pcm_bytes: bytes) -> None:
        """Ingest raw 16 kHz 16-bit mono little-endian PCM bytes in 512-sample (32ms) Silero windows."""
        is_realtime_chunk = len(pcm_bytes) <= 4096
        self.rem_bytes.extend(pcm_bytes)
        window_bytes = SILERO_WINDOW_SAMPLES * 2
        while len(self.rem_bytes) >= window_bytes:
            chunk = bytes(self.rem_bytes[:window_bytes])
            del self.rem_bytes[:window_bytes]
            frame_i16 = np.frombuffer(chunk, dtype=np.int16).copy()
            await self._process_window(frame_i16, allow_early_lid=is_realtime_chunk)

    async def _process_window(self, frame_i16: np.ndarray, allow_early_lid: bool = True) -> None:
        self.frame_counter += 1
        frame_f32 = frame_i16.astype(np.float32) / 32768.0
        tensor_frame = torch.from_numpy(frame_f32)
        with torch.no_grad():
            speech_prob = float(self.vad_model(tensor_frame, SAMPLE_RATE).item())

        is_speech = speech_prob >= self.threshold
        is_silence = speech_prob < max(0.15, self.threshold - 0.15)

        min_speech_frames = max(2, int(round(self.min_speech_duration_ms / 32.0)))
        min_silence_frames = max(4, int(round(self.min_silence_duration_ms / 32.0)))
        early_lid_frames = max(28, int(round(self.early_lid_ms / 32.0)))

        if self.frame_counter % 5 == 0:
            await self.on_vad_telemetry(
                {
                    "type": "silero_vad_Prob",
                    "speech_prob": round(speech_prob, 4),
                    "threshold": self.threshold,
                    "state": self.state,
                }
            )

        if self.state == "IDLE":
            self.pre_roll.append(frame_i16)
            if is_speech:
                self.state = "SPEECH_ACTIVE"
                self.speech_frames_count = 1
                self.silence_frames_count = 0
                self.early_lid_checked = False
                self.streamed_bytes_in_turn = 0
                self.buffered_frames = list(self.pre_roll)
                await self.on_vad_telemetry(
                    {
                        "type": "silero_vad_event",
                        "event": "SPEECH_ONSET",
                        "state": self.state,
                        "speech_prob": round(speech_prob, 4),
                        "message": "Silero VAD detected speech onset; buffering for Prosodia language identification.",
                    }
                )
            return

        if self.state == "STREAMING_TARGET":
            # Early-prefix LID already verified Target language; stream audio live to Gemini!
            frame_bytes = frame_i16.tobytes()
            await self.on_stream_pcm_to_gemini(frame_bytes)
            self.streamed_bytes_in_turn += len(frame_bytes)
            if is_speech:
                self.speech_frames_count += 1
                self.silence_frames_count = 0
            elif is_silence:
                self.silence_frames_count += 1

            if self.silence_frames_count >= min_silence_frames:
                await self.on_trigger_activity_end()
                self.state = "IDLE"
                self.speech_frames_count = 0
                self.silence_frames_count = 0
                self.buffered_frames.clear()
            return

        if self.state == "SUPPRESSING_SOURCE":
            # Early-prefix LID already verified Source/unselected language; stay 100% silent until speech ends!
            if is_speech:
                self.speech_frames_count += 1
                self.silence_frames_count = 0
            elif is_silence:
                self.silence_frames_count += 1

            if self.silence_frames_count >= min_silence_frames:
                self.state = "IDLE"
                self.speech_frames_count = 0
                self.silence_frames_count = 0
                self.buffered_frames.clear()
            return

        if self.state == "SPEECH_ACTIVE":
            self.buffered_frames.append(frame_i16)
            if is_speech:
                self.speech_frames_count += 1
                self.silence_frames_count = 0
            elif is_silence:
                self.silence_frames_count += 1

            # Early-Prefix Streaming LID (~1.15s into live microphone speech)
            if (
                allow_early_lid
                and self.enable_early_stream_to_gemini
                and not self.early_lid_checked
                and self.speech_frames_count >= early_lid_frames
                and self.silence_frames_count == 0
            ):
                self.early_lid_checked = True
                await self._run_early_prefix_lid()
                if self.state != "SPEECH_ACTIVE":
                    return

            if self.silence_frames_count >= min_silence_frames:
                await self._finalize_utterance(
                    min_speech_frames=min_speech_frames,
                    trigger_source="silero_end_of_speech",
                )

    async def _run_early_prefix_lid(self) -> None:
        """Evaluate the first ~1.15s of active speech mid-utterance for zero-wait streaming."""
        silero_speech_dur_s = round(
            (self.speech_frames_count * SILERO_WINDOW_SAMPLES) / float(SAMPLE_RATE),
            3,
        )
        wav_i16 = np.concatenate(self.buffered_frames)
        wav_f32 = wav_i16.astype(np.float32) / 32768.0

        loop = asyncio.get_running_loop()
        analysis = await loop.run_in_executor(
            GATE_EXECUTOR,
            self.engine.analyze_utterance,
            wav_f32,
            self.source_lang,
            self.target_lang,
            silero_speech_dur_s,
            self.confidence_threshold,
        )

        src_prob = float(analysis["language_probabilities"].get("source_prob", 0.0))
        tgt_prob = float(analysis["language_probabilities"].get("target_prob", 0.0))
        utterance_id = f"utt-{uuid.uuid4().hex[:8]}"

        if analysis["should_translate"] and tgt_prob >= 0.70:
            # High-confidence Target speech mid-utterance -> start streaming immediately!
            pcm_bytes = wav_i16.tobytes()
            self.buffered_frames.clear()
            self.state = "STREAMING_TARGET"
            self.streamed_bytes_in_turn = len(pcm_bytes)
            await self.on_trigger_activity_start()
            await self.on_stream_pcm_to_gemini(pcm_bytes)
            decision_payload = {
                "type": "gate_decision",
                "utterance_id": utterance_id,
                "timestamp": time.time(),
                "trigger_source": "early_prefix_lid",
                "turn_triggered": True,
                "bytes_forwarded_to_gemini": len(pcm_bytes),
                "silero_speech_duration_s": silero_speech_dur_s,
                **analysis,
            }
            await self.on_gate_decision(decision_payload)
        elif analysis["stay_silent"] and (src_prob >= 0.70 or analysis["detected_role"] == "other"):
            # High-confidence Source or unselected language mid-utterance -> suppress immediately!
            self.buffered_frames.clear()
            self.state = "SUPPRESSING_SOURCE"
            decision_payload = {
                "type": "gate_decision",
                "utterance_id": utterance_id,
                "timestamp": time.time(),
                "trigger_source": "early_prefix_lid",
                "turn_triggered": False,
                "bytes_forwarded_to_gemini": 0,
                "silero_speech_duration_s": silero_speech_dur_s,
                **analysis,
            }
            await self.on_gate_decision(decision_payload)

    async def flush_pending(self) -> None:
        """Finalize any buffered speech when microphone stops or sample playback ends."""
        if self.state == "STREAMING_TARGET":
            await self.on_trigger_activity_end()
            self.reset()
        elif self.state == "SPEECH_ACTIVE" and self.buffered_frames:
            min_speech_frames = max(2, int(round(self.min_speech_duration_ms / 32.0)))
            await self._finalize_utterance(
                min_speech_frames=min_speech_frames,
                trigger_source="flush_pending",
            )
        else:
            self.reset()

    async def _finalize_utterance(
        self,
        min_speech_frames: int,
        trigger_source: str,
    ) -> None:
        if self.speech_frames_count < min_speech_frames:
            self.state = "IDLE"
            self.speech_frames_count = 0
            self.silence_frames_count = 0
            self.buffered_frames.clear()
            return

        silero_speech_dur_s = round(
            (self.speech_frames_count * SILERO_WINDOW_SAMPLES) / float(SAMPLE_RATE),
            3,
        )
        strip_trailing = max(0, self.silence_frames_count - 2)
        keep_count = max(1, len(self.buffered_frames) - strip_trailing)
        trimmed_frames = self.buffered_frames[:keep_count]
        wav_i16 = np.concatenate(trimmed_frames)
        wav_f32 = wav_i16.astype(np.float32) / 32768.0

        self.state = "IDLE"
        self.speech_frames_count = 0
        self.silence_frames_count = 0
        self.buffered_frames.clear()

        loop = asyncio.get_running_loop()
        analysis = await loop.run_in_executor(
            GATE_EXECUTOR,
            self.engine.analyze_utterance,
            wav_f32,
            self.source_lang,
            self.target_lang,
            silero_speech_dur_s,
            self.confidence_threshold,
        )

        utterance_id = f"utt-{uuid.uuid4().hex[:8]}"
        should_translate = bool(analysis["should_translate"])

        if should_translate:
            pcm_bytes = wav_i16.tobytes()
            await self.on_trigger_activity_start()
            await self.on_stream_pcm_to_gemini(pcm_bytes)
            await self.on_trigger_activity_end()
            bytes_forwarded = len(pcm_bytes)
            turn_triggered = True
        else:
            bytes_forwarded = 0
            turn_triggered = False

        decision_payload = {
            "type": "gate_decision",
            "utterance_id": utterance_id,
            "timestamp": time.time(),
            "trigger_source": trigger_source,
            "turn_triggered": turn_triggered,
            "bytes_forwarded_to_gemini": bytes_forwarded,
            "silero_speech_duration_s": silero_speech_dur_s,
            **analysis,
        }
        await self.on_gate_decision(decision_payload)
