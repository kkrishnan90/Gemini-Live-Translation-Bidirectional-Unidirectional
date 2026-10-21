"""Unit tests for ProsodiaLanguageGateEngine (Seam 1).

Tests language identification and paralinguistic prosody analysis across:
1. Target language spoken -> translates to Source (should_translate=True, stay_silent=False).
2. Source language spoken -> stays silent (should_translate=False, stay_silent=True).
3. Unselected third language spoken -> stays silent (detected_role='other', stay_silent=True).
4. Silence / low-energy noise -> ignored safely (stay_silent=True).
5. Identical base language validation error (e.g., en-US and en-IN).
6. ProsodiaModel (best.pt) paralinguistic & prosody metrics readout.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import soundfile as sf

from prosodia_lid_engine import (
    ProsodiaLanguageGateEngine,
    SUPPORTED_LANGUAGES,
    validate_language_pair,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
CKPT_PATH = PROJECT_DIR / "model" / "best.pt"
BINS_PATH = PROJECT_DIR / "model" / "bins.json"
SAMPLE_DIR = PROJECT_DIR / "sample_audio"


@pytest.fixture(scope="module")
def engine() -> ProsodiaLanguageGateEngine:
    return ProsodiaLanguageGateEngine(ckpt_path=CKPT_PATH, bins_path=BINS_PATH)


def load_sample_wav(filename: str) -> np.ndarray:
    wav, sr = sf.read(str(SAMPLE_DIR / filename), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    assert sr == 16000
    return wav


def test_validate_language_pair_rejects_same_base_language() -> None:
    with pytest.raises(ValueError, match="distinct base languages"):
        validate_language_pair("en-US", "en-IN")
    with pytest.raises(ValueError, match="Unsupported language"):
        validate_language_pair("xx-YY", "hi-IN")


def test_source_spoken_stays_silent_en_to_hi(engine: ProsodiaLanguageGateEngine) -> None:
    """When Source=en-US and Target=hi-IN, speaking English (Source) must stay silent."""
    wav_en = load_sample_wav("human_query_en.wav")
    res = engine.analyze_utterance(
        wav=wav_en,
        source_lang="en-US",
        target_lang="hi-IN",
    )
    assert res["detected_iso"] == "en"
    assert res["detected_role"] == "source"
    assert res["should_translate"] is False
    assert res["stay_silent"] is True
    assert res["gate_action"] == "STAY_SILENT_SOURCE_SPOKEN"
    assert res["language_probabilities"]["source_prob"] > 0.85


def test_target_spoken_translates_to_source_en_to_hi(engine: ProsodiaLanguageGateEngine) -> None:
    """When Source=en-US and Target=hi-IN, speaking Hindi (Target) must trigger translation to English."""
    wav_hi = load_sample_wav("human_sentence_hindi.wav")
    res = engine.analyze_utterance(
        wav=wav_hi,
        source_lang="en-US",
        target_lang="hi-IN",
    )
    assert res["detected_iso"] == "hi"
    assert res["detected_role"] == "target"
    assert res["should_translate"] is True
    assert res["stay_silent"] is False
    assert res["gate_action"] == "TRANSLATE_TARGET_TO_SOURCE"
    assert res["language_probabilities"]["target_prob"] > 0.75


def test_reversed_direction_hi_source_en_target(engine: ProsodiaLanguageGateEngine) -> None:
    """When Source=hi-IN and Target=en-US:
    - Hindi (Source) stays silent.
    - English (Target) triggers translation to Hindi.
    """
    wav_hi = load_sample_wav("human_sentence_hindi.wav")
    res_hi = engine.analyze_utterance(wav=wav_hi, source_lang="hi-IN", target_lang="en-US")
    assert res_hi["detected_role"] == "source"
    assert res_hi["should_translate"] is False
    assert res_hi["stay_silent"] is True
    assert res_hi["gate_action"] == "STAY_SILENT_SOURCE_SPOKEN"

    wav_en = load_sample_wav("human_query_en.wav")
    res_en = engine.analyze_utterance(wav=wav_en, source_lang="hi-IN", target_lang="en-US")
    assert res_en["detected_role"] == "target"
    assert res_en["should_translate"] is True
    assert res_en["stay_silent"] is False
    assert res_en["gate_action"] == "TRANSLATE_TARGET_TO_SOURCE"


def test_target_spoken_bengali_and_tamil(engine: ProsodiaLanguageGateEngine) -> None:
    """Verify Bengali (bn-BD) and Tamil (ta-IN) Target speech detection."""
    wav_bn = load_sample_wav("human_sentence_bengali.wav")
    res_bn = engine.analyze_utterance(wav=wav_bn, source_lang="en-US", target_lang="bn-BD")
    assert res_bn["detected_iso"] == "bn"
    assert res_bn["detected_role"] == "target"
    assert res_bn["should_translate"] is True
    assert res_bn["stay_silent"] is False

    wav_ta = load_sample_wav("human_sentence_tamil.wav")
    res_ta = engine.analyze_utterance(wav=wav_ta, source_lang="en-US", target_lang="ta-IN")
    assert res_ta["detected_iso"] == "ta"
    assert res_ta["detected_role"] == "target"
    assert res_ta["should_translate"] is True
    assert res_ta["stay_silent"] is False


def test_unselected_third_language_stays_silent(engine: ProsodiaLanguageGateEngine) -> None:
    """Edge case: When Source=en-US and Target=hi-IN, speaking Tamil (3rd unselected language) stays silent."""
    wav_ta = load_sample_wav("human_sentence_tamil.wav")
    res = engine.analyze_utterance(wav=wav_ta, source_lang="en-US", target_lang="hi-IN")
    assert res["detected_role"] == "other"
    assert res["detected_iso"] == "ta"
    assert res["should_translate"] is False
    assert res["stay_silent"] is True
    assert res["gate_action"] == "STAY_SILENT_UNSELECTED_LANGUAGE"


def test_silence_or_low_energy_noise_stays_silent(engine: ProsodiaLanguageGateEngine) -> None:
    """Edge case: Pure silence or sub-threshold microphone hiss must be ignored and stay silent."""
    silent_wav = np.zeros(16000, dtype=np.float32)
    res = engine.analyze_utterance(wav=silent_wav, source_lang="en-US", target_lang="hi-IN")
    assert res["should_translate"] is False
    assert res["stay_silent"] is True
    assert res["gate_action"] == "IGNORE_NOISE_OR_SILENCE"


def test_prosodia_paralinguistic_metrics_populated(engine: ProsodiaLanguageGateEngine) -> None:
    """Verify ProsodiaModel (best.pt) and explicit prosody metrics are returned alongside LID."""
    wav_hi = load_sample_wav("human_sentence_hindi.wav")
    res = engine.analyze_utterance(wav=wav_hi, source_lang="en-US", target_lang="hi-IN")
    prosody = res["prosodia"]
    assert prosody["pitch_direction"] in ("falling", "level", "rising")
    assert prosody["speaking_rate"] in ("slow", "medium", "fast")
    assert prosody["loudness_level"] in ("quiet", "moderate", "loud")
    assert prosody["emotion"] in ("neutral", "joy", "sadness", "anger", "surprise", "fear", "disgust")
    assert isinstance(prosody["is_backchannel"], bool)
    assert prosody["active_duration_s"] > 0.5


def test_analyze_utterance_sub_250ms_latency(engine: ProsodiaLanguageGateEngine) -> None:
    """Every utterance analysis (Whisper LID + Prosodia prosody) must finish in < 250 ms on CPU."""
    for fname, src, tgt, expected_role in [
        ("human_query_en.wav", "en-US", "hi-IN", "source"),
        ("human_sentence_hindi.wav", "en-US", "hi-IN", "target"),
        ("human_sentence_bengali.wav", "en-US", "bn-BD", "target"),
        ("human_sentence_tamil.wav", "en-US", "ta-IN", "target"),
    ]:
        wav = load_sample_wav(fname)
        res = engine.analyze_utterance(wav, source_lang=src, target_lang=tgt)
        assert res["detected_role"] == expected_role
        assert res["inference_ms"] < 250.0, (
            f"Expected < 250ms latency for {fname}, got {res['inference_ms']}ms"
        )

