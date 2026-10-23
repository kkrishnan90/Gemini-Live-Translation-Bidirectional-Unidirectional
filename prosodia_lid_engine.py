"""Prosodia Spoken Language Identification (LID) & Paralinguistic Gate Engine.

High-Performance Architecture (<150 ms CPU latency):
1. Vectorized FFT Autocorrelation Prosody Extractor (`fast_extract_prosody_tensor`):
   Replaces slow `librosa.pyin` (1,900 ms cold / 200 ms warm) with a vectorized FFT autocorrelation
   and RMS extractor that computes 50 Hz `[f0_hz, rms, voiced_prob]` in ~3.5 ms.
2. Dynamic-Length Shared `openai/whisper-small` Encoder (`_fast_whisper_encode`):
   Instead of padding every utterance to 30.0 seconds (3,000 mel frames / 1,500 tokens = ~1,270 ms),
   slices the active speech window (up to 2.2s + 0.8s native Whisper silence tail, <= 300 mel frames /
   150 encoder tokens) and runs the 12-layer encoder + 1-step SOT (`<|startoftranscript|>`) decoder
   plus `ProsodiaModel` (`best.pt`) in ~115-145 ms total.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from transformers import (
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
)
from transformers.modeling_outputs import BaseModelOutput

from prosodia.config import RunConfig
from prosodia.device import get_device
from prosodia.features import (
    ENCODERS,
    SAMPLE_RATE,
    WHISPER_FRAMES_PER_SECOND,
    truncate_to_max_seconds,
)
from prosodia.labels.acoustic import (
    LOUDNESS_NAMES,
    PITCH_NAMES,
    RATE_NAMES,
    TertileBinner,
    mean_loudness_db,
    pitch_slope_semitones_per_second,
)
from prosodia.model.heads import confidence
from prosodia.model.prosodia import ProsodiaModel
from prosodia.schema import QuestionSpec
from prosodia.train.loop import load_checkpoint

WHISPER_DIM = 768
EMOTION_OPTIONS = ["neutral", "joy", "sadness", "anger", "surprise", "fear", "disgust"]
OPTIMAL_TORCH_THREADS = max(4, min(12, (os.cpu_count() or 8)))
torch.set_num_threads(OPTIMAL_TORCH_THREADS)
_TORCH_WORKER = ThreadPoolExecutor(max_workers=1, thread_name_prefix="prosodia_torch_worker")

# Supported GEAP Gemini 3.8 Live BCP-47 languages mapped to Whisper language tokens
SUPPORTED_LANGUAGES: dict[str, dict[str, str]] = {
    "en-US": {"name": "English (US)", "iso": "en", "whisper_token": "<|en|>"},
    "en-IN": {"name": "English (India)", "iso": "en", "whisper_token": "<|en|>"},
    "hi-IN": {"name": "Hindi (India)", "iso": "hi", "whisper_token": "<|hi|>"},
    "bn-BD": {"name": "Bengali", "iso": "bn", "whisper_token": "<|bn|>"},
    "ta-IN": {"name": "Tamil (India)", "iso": "ta", "whisper_token": "<|ta|>"},
    "te-IN": {"name": "Telugu (India)", "iso": "te", "whisper_token": "<|te|>"},
    "mr-IN": {"name": "Marathi (India)", "iso": "mr", "whisper_token": "<|mr|>"},
    "es-US": {"name": "Spanish (US)", "iso": "es", "whisper_token": "<|es|>"},
    "fr-FR": {"name": "French (France)", "iso": "fr", "whisper_token": "<|fr|>"},
    "de-DE": {"name": "German (Germany)", "iso": "de", "whisper_token": "<|de|>"},
    "ja-JP": {"name": "Japanese (Japan)", "iso": "ja", "whisper_token": "<|ja|>"},
    "ko-KR": {"name": "Korean (Korea)", "iso": "ko", "whisper_token": "<|ko|>"},
    "pt-BR": {"name": "Portuguese (Brazil)", "iso": "pt", "whisper_token": "<|pt|>"},
    "ar-EG": {"name": "Arabic (Egypt)", "iso": "ar", "whisper_token": "<|ar|>"},
    "it-IT": {"name": "Italian (Italy)", "iso": "it", "whisper_token": "<|it|>"},
    "ru-RU": {"name": "Russian (Russia)", "iso": "ru", "whisper_token": "<|ru|>"},
}


def validate_language_pair(source_lang: str, target_lang: str) -> tuple[dict[str, str], dict[str, str]]:
    """Validate that source_lang and target_lang are supported and have distinct ISO codes."""
    if source_lang not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language code: {source_lang!r}")
    if target_lang not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language code: {target_lang!r}")
    src_info = SUPPORTED_LANGUAGES[source_lang]
    tgt_info = SUPPORTED_LANGUAGES[target_lang]
    if src_info["iso"] == tgt_info["iso"]:
        raise ValueError(
            f"Source ({source_lang}) and target ({target_lang}) must have distinct base languages "
            f"(both resolve to ISO '{src_info['iso']}')."
        )
    return src_info, tgt_info


def fast_extract_prosody_tensor(wav: np.ndarray, sr: int = SAMPLE_RATE) -> torch.Tensor:
    """Vectorized 50 Hz `[f0_hz, rms, vprob]` prosody extraction in ~3.5 ms (replaces slow `librosa.pyin`)."""
    hop = 320  # 20ms at 16 kHz (50 Hz frame rate)
    win = 640  # 40ms analysis window
    if len(wav) < win:
        wav = np.pad(wav, (0, win - len(wav)))
    n_frames = 1 + (len(wav) - win) // hop
    strides = (wav.strides[0] * hop, wav.strides[0])
    frames = np.lib.stride_tricks.as_strided(wav, shape=(n_frames, win), strides=strides).copy()
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12).astype(np.float32)

    # Vectorized FFT autocorrelation for F0 in [70 Hz, 400 Hz] -> lag in [40, 228] samples at 16 kHz
    frames_zm = frames - np.mean(frames, axis=1, keepdims=True)
    window = np.hanning(win).astype(np.float32)
    fw = frames_zm * window
    spec = np.fft.rfft(fw, n=1024, axis=1)
    acf = np.fft.irfft(spec * np.conj(spec), n=1024, axis=1)[:, :240]
    norm = acf[:, 0:1] + 1e-9
    acf_norm = acf / norm

    min_lag, max_lag = 40, 228
    search_region = acf_norm[:, min_lag:max_lag]
    best_rel_lag = np.argmax(search_region, axis=1)
    peak_corr = search_region[np.arange(n_frames), best_rel_lag]
    best_lag = best_rel_lag + min_lag

    rms_max = float(np.max(rms)) if n_frames > 0 else 0.0
    voiced = (peak_corr >= 0.30) & (rms >= max(0.005, rms_max * 0.14))
    f0 = np.where(voiced, float(sr) / best_lag.astype(np.float32), 0.0).astype(np.float32)
    vprob = np.clip(peak_corr, 0.0, 1.0).astype(np.float32) * voiced.astype(np.float32)
    return torch.from_numpy(np.stack([f0, rms, vprob], axis=-1))


def extract_utterance_prosody_metrics(pros: torch.Tensor) -> dict[str, Any]:
    """Extract AGC-invariant acoustic & syllabic structure from `(T, 3)` `[f0, rms, vprob]`."""
    n_frames = int(pros.shape[0])
    hop_s = 0.02  # 20ms hop (50 Hz frame rate)
    total_dur = n_frames * hop_s
    if n_frames == 0:
        return {
            "total_duration_s": 0.0,
            "active_duration_s": 0.0,
            "syllable_nuclei": 0,
            "voiced_frames": 0,
            "voiced_fraction": 0.0,
            "mean_f0_hz": None,
            "f0_range_st": None,
            "peak_loudness_db": -80.0,
        }

    rms = pros[:, 1].detach().cpu().numpy()
    f0 = pros[:, 0].detach().cpu().numpy()

    max_rms = float(np.max(rms)) if len(rms) > 0 else 0.0
    if len(rms) >= 5:
        kernel = np.array([0.1, 0.2, 0.4, 0.2, 0.1], dtype=np.float32)
        smooth_rms = np.convolve(rms, kernel, mode="same")
    elif len(rms) >= 3:
        kernel = np.array([0.25, 0.5, 0.25], dtype=np.float32)
        smooth_rms = np.convolve(rms, kernel, mode="same")
    else:
        smooth_rms = rms

    smooth_max = float(np.max(smooth_rms)) if len(smooth_rms) > 0 else 0.0
    energy_floor = max(0.008, smooth_max * 0.25)
    active_mask = smooth_rms >= energy_floor
    active_frames_count = int(np.sum(active_mask))
    active_dur = float(active_frames_count * hop_s)

    peak_thresh = max(0.012, smooth_max * 0.38)
    peaks: list[int] = []
    min_dist_frames = 8
    for i in range(len(smooth_rms)):
        val = smooth_rms[i]
        if val < peak_thresh:
            continue
        left = smooth_rms[i - 1] if i > 0 else -1.0
        right = smooth_rms[i + 1] if i + 1 < len(smooth_rms) else -1.0
        if val >= left and val >= right:
            if not peaks or (i - peaks[-1]) >= min_dist_frames:
                peaks.append(i)
            elif val > smooth_rms[peaks[-1]]:
                peaks[-1] = i

    voiced_mask = f0 > 0
    voiced_frames = int(np.sum(voiced_mask))
    voiced_fraction = float(voiced_frames / max(1, active_frames_count))
    mean_f0 = float(np.mean(f0[voiced_mask])) if voiced_frames > 0 else None
    f0_range_st = None
    if voiced_frames >= 2:
        f_min = float(np.percentile(f0[voiced_mask], 10))
        f_max = float(np.percentile(f0[voiced_mask], 90))
        if f_min > 0:
            f0_range_st = float(12.0 * math.log2(f_max / f_min))

    peak_db = float(20.0 * math.log10(max_rms + 1e-9))

    return {
        "total_duration_s": round(total_dur, 3),
        "active_duration_s": round(active_dur, 3),
        "syllable_nuclei": len(peaks),
        "voiced_frames": voiced_frames,
        "voiced_fraction": round(min(1.0, voiced_fraction), 3),
        "mean_f0_hz": round(mean_f0, 1) if mean_f0 is not None else None,
        "f0_range_st": round(f0_range_st, 2) if f0_range_st is not None else None,
        "peak_loudness_db": round(peak_db, 2),
    }


class ProsodiaLanguageGateEngine:
    """Thread-safe, low-latency (<150ms) Prosodia + Whisper-small spoken language identification & prosody engine."""

    def __init__(self, ckpt_path: Path, bins_path: Path) -> None:
        self._lock = threading.Lock()
        self.device = get_device()
        self.cfg = RunConfig(name="live", encoder="whisper", brier_weight=0.5)

        self.prosodia_model = ProsodiaModel(
            in_dim=WHISPER_DIM,
            d_model=self.cfg.d_model,
            state_layers=self.cfg.state_layers,
            branch_layers=self.cfg.branch_layers,
            n_heads=self.cfg.n_heads,
            stride=self.cfg.stride,
        )
        self.checkpoint_epoch = load_checkpoint(ckpt_path, self.prosodia_model)
        self.prosodia_model.to(self.device).eval()

        if not bins_path.exists():
            raise FileNotFoundError(f"Missing tertile bins file: {bins_path}")
        raw_bins = json.loads(bins_path.read_text())
        self.bins = {
            "pitch": TertileBinner(raw_bins["pitch"][0], raw_bins["pitch"][1], PITCH_NAMES),
            "rate": TertileBinner(raw_bins["rate"][0], raw_bins["rate"][1], RATE_NAMES),
            "loud": TertileBinner(raw_bins["loud"][0], raw_bins["loud"][1], LOUDNESS_NAMES),
        }

        whisper_model_id = ENCODERS["whisper"]
        self.whisper_fe = WhisperFeatureExtractor.from_pretrained(whisper_model_id)
        self.whisper_tokenizer = WhisperTokenizer.from_pretrained(whisper_model_id)
        self.whisper_model = (
            WhisperForConditionalGeneration.from_pretrained(whisper_model_id)
            .to(self.device)
            .eval()
        )
        for p in self.whisper_model.parameters():
            p.requires_grad_(False)

        self.sot_id = int(self.whisper_tokenizer.convert_tokens_to_ids("<|startoftranscript|>"))
        self.iso_to_token_id: dict[str, int] = {}
        self.iso_to_display: dict[str, str] = {}
        for info in SUPPORTED_LANGUAGES.values():
            iso = info["iso"]
            if iso not in self.iso_to_token_id:
                tid = int(self.whisper_tokenizer.convert_tokens_to_ids(info["whisper_token"]))
                self.iso_to_token_id[iso] = tid
                self.iso_to_display[iso] = info["name"].split(" (")[0]

        self.emotion_spec = QuestionSpec(
            key="emotion",
            qtype="choice",
            instructions="Which emotion is the speaker expressing?",
            criteria={e: None for e in EMOTION_OPTIONS},
        )
        self.rate_spec = QuestionSpec(
            key="speaking_rate",
            qtype="score",
            instructions="How quickly is the speaker talking?",
            criteria=list(RATE_NAMES),
        )
        self._questions_payload = [
            {
                self.emotion_spec.key: {
                    "instructions": self.emotion_spec.instructions,
                    "options": self.emotion_spec.options,
                    "qtype": self.emotion_spec.qtype,
                },
                self.rate_spec.key: {
                    "instructions": self.rate_spec.instructions,
                    "options": self.rate_spec.options,
                    "qtype": self.rate_spec.qtype,
                },
            }
        ]
        def _warmup_on_worker() -> None:
            with torch.no_grad():
                preload = (
                    [""]
                    + [self.emotion_spec.instructions]
                    + list(self.emotion_spec.options)
                    + [self.rate_spec.instructions]
                    + list(self.rate_spec.options)
                )
                self.prosodia_model.question_encoder.embed_texts(preload)
                dummy_mel = torch.zeros(1, 80, 240, device=self.device)
                dummy_h = self._fast_whisper_encode(dummy_mel)
                _ = self.whisper_model(
                    encoder_outputs=BaseModelOutput(last_hidden_state=dummy_h),
                    decoder_input_ids=torch.tensor([[self.sot_id]], device=self.device),
                )
                _ = self.prosodia_model(
                    {
                        "uid": ["warmup"],
                        "audio": dummy_h[:, :50, :],
                        "audio_mask": torch.ones(1, 50, dtype=torch.bool, device=self.device),
                        "audio_present": torch.tensor([True], device=self.device),
                        "context": [""],
                        "context_present": torch.tensor([False], device=self.device),
                        "questions": self._questions_payload,
                        "targets": [{}],
                        "gold": [{}],
                        "speaker": [None],
                    }
                )

        _TORCH_WORKER.submit(_warmup_on_worker).result()

    def _fast_whisper_encode(self, mel_features: torch.Tensor) -> torch.Tensor:
        """Run `WhisperEncoder` on dynamic mel length `T_mel` (`<= 300` instead of `3000`) for ~10x speedup."""
        enc = self.whisper_model.model.encoder
        inputs_embeds = F.gelu(enc.conv1(mel_features))
        inputs_embeds = F.gelu(enc.conv2(inputs_embeds))
        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        seq_len = inputs_embeds.shape[1]
        hidden_states = inputs_embeds + enc.embed_positions.weight[:seq_len]
        for encoder_layer in enc.layers:
            hidden_states = encoder_layer(hidden_states, None)
        hidden_states = enc.layer_norm(hidden_states)
        return hidden_states

    def analyze_utterance(
        self,
        wav: np.ndarray,
        source_lang: str,
        target_lang: str,
        silero_speech_duration_s: float | None = None,
        confidence_threshold: float = 0.55,
        bidirectional: bool = False,
    ) -> dict[str, Any]:
        """Identify spoken language and extract Prosodia paralinguistic prosody in ~115-150 ms."""
        if not threading.current_thread().name.startswith("prosodia_torch_worker"):
            return _TORCH_WORKER.submit(
                self._analyze_utterance_impl,
                wav,
                source_lang,
                target_lang,
                silero_speech_duration_s,
                confidence_threshold,
                bidirectional,
            ).result()
        return self._analyze_utterance_impl(
            wav,
            source_lang,
            target_lang,
            silero_speech_duration_s,
            confidence_threshold,
            bidirectional,
        )

    def _analyze_utterance_impl(
        self,
        wav: np.ndarray,
        source_lang: str,
        target_lang: str,
        silero_speech_duration_s: float | None = None,
        confidence_threshold: float = 0.55,
        bidirectional: bool = False,
    ) -> dict[str, Any]:
        if torch.get_num_threads() < OPTIMAL_TORCH_THREADS:
            torch.set_num_threads(OPTIMAL_TORCH_THREADS)
        t0 = time.perf_counter()
        src_info, tgt_info = validate_language_pair(source_lang, target_lang)
        src_iso = src_info["iso"]
        tgt_iso = tgt_info["iso"]

        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        wav = wav.astype(np.float32)

        abs_wav = np.abs(wav)
        peak_val = float(np.max(abs_wav)) if len(abs_wav) > 0 else 0.0
        if peak_val > 0.005:
            active_idx = np.where(abs_wav >= max(0.004, peak_val * 0.12))[0]
            if len(active_idx) > 0:
                pad_samples = int(0.04 * SAMPLE_RATE)
                start_i = max(0, int(active_idx[0]) - pad_samples)
                end_i = min(len(wav), int(active_idx[-1]) + pad_samples)
                wav = wav[start_i:end_i]

        if len(wav) < int(0.16 * SAMPLE_RATE):
            pad = np.zeros(int(0.16 * SAMPLE_RATE) - len(wav), dtype=np.float32)
            wav = np.concatenate([wav, pad])

        wav, _ = truncate_to_max_seconds(wav)

        with self._lock:
            pros = fast_extract_prosody_tensor(wav, SAMPLE_RATE)
            pros_metrics = extract_utterance_prosody_metrics(pros)
            active_dur = pros_metrics["active_duration_s"]
            if silero_speech_duration_s is not None and silero_speech_duration_s > 0:
                active_dur = round(min(active_dur, float(silero_speech_duration_s)), 3)
                pros_metrics["active_duration_s"] = active_dur

            # Check for silence or low-energy noise before running Whisper/Prosodia
            if (
                peak_val < 0.005
                or pros_metrics["peak_loudness_db"] < -48.0
                or active_dur < 0.14
                or pros_metrics["voiced_frames"] < 2
            ):
                elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
                return {
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                    "source_iso": src_iso,
                    "target_iso": tgt_iso,
                    "bidirectional": bool(bidirectional),
                    "detected_iso": "none",
                    "detected_language_name": "Silence / Noise",
                    "detected_role": "noise",
                    "should_translate": False,
                    "stay_silent": True,
                    "translated_to_lang": None,
                    "translated_to_language_name": None,
                    "gate_action": "IGNORE_NOISE_OR_SILENCE",
                    "reason": (
                        f"Ignored low-energy/unvoiced audio (active={active_dur}s, "
                        f"peak={pros_metrics['peak_loudness_db']} dB)"
                    ),
                    "language_probabilities": {
                        "source_prob": 0.0,
                        "target_prob": 0.0,
                        "top_languages": {},
                    },
                    "prosodia": {
                        **pros_metrics,
                        "pitch_direction": "level",
                        "speaking_rate": "medium",
                        "loudness_level": "quiet",
                        "emotion": "neutral",
                        "emotion_confidence": 0.0,
                        "is_backchannel": True,
                    },
                    "inference_ms": elapsed_ms,
                }

            # Cap the LID acoustic slice to the first 2.2s of active speech + 0.8s Whisper silence tail (<= 300 mel frames)
            wav_lid = wav[: int(2.2 * SAMPLE_RATE)]
            actual_mel = int(math.ceil(len(wav_lid) / 160.0))
            t_mel = min(300, max(240, actual_mel + 80))
            if t_mel % 2 == 1:
                t_mel += 1

            with torch.no_grad():
                mel = self.whisper_fe(
                    wav_lid,
                    sampling_rate=SAMPLE_RATE,
                    return_tensors="pt",
                    max_length=t_mel * 160,
                    truncation=True,
                )
                enc_hidden = self._fast_whisper_encode(mel.input_features.to(self.device))
                dec_ids = torch.tensor([[self.sot_id]], device=self.device)
                dec_out = self.whisper_model(
                    encoder_outputs=BaseModelOutput(last_hidden_state=enc_hidden),
                    decoder_input_ids=dec_ids,
                )
                sot_logits = dec_out.logits[0, 0]

                # 1. Pairwise softmax over {source_iso, target_iso}
                pair_token_ids = [self.iso_to_token_id[src_iso], self.iso_to_token_id[tgt_iso]]
                pair_probs_t = torch.softmax(sot_logits[pair_token_ids], dim=-1).cpu()
                src_pair_prob = float(pair_probs_t[0].item())
                tgt_pair_prob = float(pair_probs_t[1].item())

                # 2. Multi-language distribution over all supported ISO languages
                all_isos = list(self.iso_to_token_id.keys())
                all_tids = [self.iso_to_token_id[k] for k in all_isos]
                all_probs_t = torch.softmax(sot_logits[all_tids], dim=-1).cpu()
                all_probs_map = {
                    iso: round(float(p), 4)
                    for iso, p in zip(all_isos, all_probs_t.tolist())
                }
                sorted_langs = sorted(all_probs_map.items(), key=lambda kv: kv[1], reverse=True)
                top_global_iso, top_global_prob = sorted_langs[0]
                pair_global_mass = all_probs_map.get(src_iso, 0.0) + all_probs_map.get(tgt_iso, 0.0)

                # 3. ProsodiaModel (best.pt) paralinguistic readout on trimmed encoder frames
                valid_frames = max(
                    1,
                    min(
                        math.ceil(len(wav_lid) / SAMPLE_RATE * WHISPER_FRAMES_PER_SECOND),
                        int(enc_hidden.shape[1]),
                    ),
                )
                trimmed_feats = enc_hidden[:, :valid_frames, :]
                batch = {
                    "uid": ["live"],
                    "audio": trimmed_feats,
                    "audio_mask": torch.ones(
                        1, valid_frames, dtype=torch.bool, device=self.device
                    ),
                    "audio_present": torch.tensor([True], device=self.device),
                    "context": [""],
                    "context_present": torch.tensor([False], device=self.device),
                    "questions": self._questions_payload,
                    "targets": [{}],
                    "gold": [{}],
                    "speaker": [None],
                }
                p_out = self.prosodia_model(batch)[0]
                emo_probs = torch.softmax(p_out["emotion"].detach().float().cpu(), dim=-1)
                emo_idx = int(torch.argmax(emo_probs).item())
                emotion_label = EMOTION_OPTIONS[emo_idx]
                emotion_conf = round(float(confidence(emo_probs).item()), 4)

                rate_probs = torch.softmax(p_out["speaking_rate"].detach().float().cpu(), dim=-1)
                rate_idx = int(torch.argmax(rate_probs).item())

            slope = pitch_slope_semitones_per_second(pros)
            loud_db = mean_loudness_db(pros)
            pitch_label = self.bins["pitch"](slope) if slope is not None else "level"
            loud_label = self.bins["loud"](loud_db)
            rate_label = RATE_NAMES[rate_idx]
            is_backchannel = bool(
                active_dur <= 0.70 and pros_metrics["syllable_nuclei"] <= 2
            )

        # Gate decision logic:
        # Check if an unselected third language clearly dominates over both Source and Target
        if (
            top_global_iso not in (src_iso, tgt_iso)
            and (
                (top_global_prob >= 0.45 and pair_global_mass < 0.45)
                or (top_global_prob > pair_global_mass and pair_global_mass < 0.35)
            )
        ):
            detected_iso = top_global_iso
            detected_role = "other"
            should_translate = False
            stay_silent = True
            translated_to_lang = None
            translated_to_language_name = None
            gate_action = "STAY_SILENT_UNSELECTED_LANGUAGE"
            reason = (
                f"Prosodia detected unselected language '{self.iso_to_display.get(detected_iso, detected_iso)}' "
                f"(P={top_global_prob:.2%}); neither Source ({src_info['name']}) nor Target ({tgt_info['name']}). "
                f"Staying silent."
            )
        elif tgt_pair_prob > src_pair_prob and tgt_pair_prob >= confidence_threshold:
            detected_iso = tgt_iso
            detected_role = "target"
            should_translate = True
            stay_silent = False
            translated_to_lang = source_lang
            translated_to_language_name = src_info["name"]
            gate_action = "TRANSLATE_TARGET_TO_SOURCE"
            reason = (
                f"Prosodia identified Target language {tgt_info['name']} "
                f"(P({tgt_iso})={tgt_pair_prob:.2%} vs P({src_iso})={src_pair_prob:.2%}). "
                f"Forwarding to Gemini 3.8 Live to translate into {src_info['name']}."
            )
        elif bidirectional and src_pair_prob >= tgt_pair_prob and src_pair_prob >= confidence_threshold:
            detected_iso = src_iso
            detected_role = "source"
            should_translate = True
            stay_silent = False
            translated_to_lang = target_lang
            translated_to_language_name = tgt_info["name"]
            gate_action = "TRANSLATE_SOURCE_TO_TARGET"
            reason = (
                f"Prosodia identified Source language {src_info['name']} "
                f"(P({src_iso})={src_pair_prob:.2%} vs P({tgt_iso})={tgt_pair_prob:.2%}). "
                f"Bidirectional mode active: forwarding to Gemini 3.8 Live to translate into {tgt_info['name']}."
            )
        else:
            detected_iso = src_iso
            detected_role = "source"
            should_translate = False
            stay_silent = True
            translated_to_lang = None
            translated_to_language_name = None
            gate_action = "STAY_SILENT_SOURCE_SPOKEN"
            reason = (
                f"Prosodia identified Source language {src_info['name']} "
                f"(P({src_iso})={src_pair_prob:.2%} vs P({tgt_iso})={tgt_pair_prob:.2%}). "
                f"Suppressing turn so Gemini 3.8 Live stays completely silent."
            )

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        return {
            "source_lang": source_lang,
            "target_lang": target_lang,
            "source_iso": src_iso,
            "target_iso": tgt_iso,
            "bidirectional": bool(bidirectional),
            "detected_iso": detected_iso,
            "detected_language_name": self.iso_to_display.get(detected_iso, detected_iso),
            "detected_role": detected_role,
            "should_translate": should_translate,
            "stay_silent": stay_silent,
            "translated_to_lang": translated_to_lang,
            "translated_to_language_name": translated_to_language_name,
            "gate_action": gate_action,
            "reason": reason,
            "language_probabilities": {
                "source_prob": round(src_pair_prob, 4),
                "target_prob": round(tgt_pair_prob, 4),
                "pair_global_mass": round(pair_global_mass, 4),
                "top_languages": dict(sorted_langs[:5]),
            },
            "prosodia": {
                **pros_metrics,
                "pitch_slope_st_per_s": round(float(slope), 2) if slope is not None else None,
                "pitch_direction": pitch_label,
                "speaking_rate": rate_label,
                "mean_loudness_db": round(float(loud_db), 2),
                "loudness_level": loud_label,
                "emotion": emotion_label,
                "emotion_confidence": emotion_conf,
                "is_backchannel": is_backchannel,
            },
            "inference_ms": elapsed_ms,
        }
