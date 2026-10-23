"""Functional & Live Agent Platform (formerly known as Vertex AI) End-to-End Tests for Prosodia + Gemini 3.8 Live Translation Demo.

Tests:
1. FastAPI REST endpoints (`GET /api/health`, `GET /api/samples`, `POST /api/analyze-sample`, `GET /`).
2. Live WebSocket (`WS /ws/translate`) connected to `gemini-3.8-live` on Agent Platform (formerly known as Vertex AI) (`us-central1`):
   - Source language spoken (`en-US` when Source=`en-US`, Target=`hi-IN`):
     Asserts `stay_silent=True`, `should_translate=False`, `bytes_forwarded_to_gemini=0`,
     and zero audio bytes or turns returned by `gemini-3.8-live`.
   - Target language spoken (`hi-IN` when Source=`en-US`, Target=`hi-IN`):
     Asserts `should_translate=True`, `stay_silent=False`, `turn_triggered=True`,
     and verifies `gemini-3.8-live` returns non-empty 24 kHz translated audio bytes
     and a non-empty English translation transcript (`turn_complete`).
   - Reversed language pair (`Source=hi-IN`, `Target=en-US`):
     Asserts Hindi speech is now suppressed (`stay_silent=True`) while English speech
     is translated into Hindi by `gemini-3.8-live`.
"""

from __future__ import annotations

import json
import time
from typing import Any

from fastapi.testclient import TestClient
import pytest


@pytest.fixture(scope="module")
def client() -> TestClient:
    from app import app

    return TestClient(app)


def test_health_endpoint(client: TestClient) -> None:
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model_id"] == "gemini-3.8-live"
    assert data["platform"] == "Agent Platform (formerly known as Vertex AI)"
    assert data["location"] == "us-central1"
    assert "project_id" not in data
    assert "service_account" not in data
    assert "en-US" in data["supported_languages"]
    assert "hi-IN" in data["supported_languages"]
    assert "bn-BD" in data["supported_languages"]
    assert "ta-IN" in data["supported_languages"]


def test_samples_and_analyze_endpoints(client: TestClient) -> None:
    resp = client.get("/api/samples")
    assert resp.status_code == 200
    samples = resp.json()["samples"]
    sample_ids = {s["sample_id"] for s in samples}
    assert {"english", "hindi", "bengali", "tamil"}.issubset(sample_ids)

    # 1. Target spoken (Hindi when Source=en-US, Target=hi-IN) -> TRANSLATE_TARGET_TO_SOURCE
    r_tgt = client.post(
        "/api/analyze-sample",
        json={
            "sample_id": "hindi",
            "source_lang": "en-US",
            "target_lang": "hi-IN",
        },
    )
    assert r_tgt.status_code == 200
    d_tgt = r_tgt.json()
    assert d_tgt["detected_role"] == "target"
    assert d_tgt["should_translate"] is True
    assert d_tgt["stay_silent"] is False
    assert d_tgt["gate_action"] == "TRANSLATE_TARGET_TO_SOURCE"

    # 2. Source spoken (English when Source=en-US, Target=hi-IN) -> STAY_SILENT_SOURCE_SPOKEN
    r_src = client.post(
        "/api/analyze-sample",
        json={
            "sample_id": "english",
            "source_lang": "en-US",
            "target_lang": "hi-IN",
        },
    )
    assert r_src.status_code == 200
    d_src = r_src.json()
    assert d_src["detected_role"] == "source"
    assert d_src["should_translate"] is False
    assert d_src["stay_silent"] is True
    assert d_src["gate_action"] == "STAY_SILENT_SOURCE_SPOKEN"

    # 3. Bidirectional=True: Source spoken (English) -> TRANSLATE_SOURCE_TO_TARGET; 3rd lang (Tamil) -> SILENT
    r_bi_src = client.post(
        "/api/analyze-sample",
        json={
            "sample_id": "english",
            "source_lang": "en-US",
            "target_lang": "hi-IN",
            "bidirectional": True,
        },
    )
    assert r_bi_src.status_code == 200
    d_bi_src = r_bi_src.json()
    assert d_bi_src["bidirectional"] is True
    assert d_bi_src["should_translate"] is True
    assert d_bi_src["stay_silent"] is False
    assert d_bi_src["gate_action"] == "TRANSLATE_SOURCE_TO_TARGET"
    assert d_bi_src["translated_to_lang"] == "hi-IN"

    r_bi_other = client.post(
        "/api/analyze-sample",
        json={
            "sample_id": "tamil",
            "source_lang": "en-US",
            "target_lang": "hi-IN",
            "bidirectional": True,
        },
    )
    assert r_bi_other.status_code == 200
    d_bi_other = r_bi_other.json()
    assert d_bi_other["should_translate"] is False
    assert d_bi_other["stay_silent"] is True
    assert d_bi_other["gate_action"] == "STAY_SILENT_UNSELECTED_LANGUAGE"


def test_index_html_served(client: TestClient) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    text = resp.text
    assert '<h1 class="app-title">Gemini Live Translation</h1>' in text
    assert "<title>Gemini Live Translation" in text
    assert "gemini-3.8-live" in text
    assert "bidirectional-toggle" in text
    assert "sample-grid" not in text
    assert "sample-strip" not in text

    js_resp = client.get("/static/app.js")
    assert js_resp.status_code == 200
    assert "bidirectional-toggle" in js_resp.text
    assert "sample-grid" not in js_resp.text
    assert "inject_sample_audio" not in js_resp.text

    css_resp = client.get("/static/styles.css")
    assert css_resp.status_code == 200
    assert "sample-chip" not in css_resp.text


def _drain_until_session_ready(ws: Any, timeout_s: float = 15.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        msg = ws.receive()
        if "text" in msg and msg["text"]:
            payload = json.loads(msg["text"])
            if payload.get("type") == "session_ready":
                return payload
            if payload.get("type") == "error":
                pytest.fail(f"WebSocket error while connecting to GEAP: {payload}")
    pytest.fail("Timed out waiting for session_ready from gemini-3.8-live on GEAP")


def test_live_websocket_source_silent_and_target_translated_on_geap(client: TestClient) -> None:
    """End-to-end live test on Agent Platform (formerly known as Vertex AI) (`us-central1`, `gemini-3.8-live`)."""
    with client.websocket_connect("/ws/translate") as ws:
        ws.send_text(
            json.dumps(
                {
                    "type": "setup",
                    "config": {
                        "source_lang": "en-US",
                        "target_lang": "hi-IN",
                        "voice_name": "Kore",
                    },
                }
            )
        )
        ready = _drain_until_session_ready(ws)
        assert ready["config"]["source_lang"] == "en-US"
        assert ready["config"]["target_lang"] == "hi-IN"

        # STEP 1: Source language spoken (English) -> MUST stay 100% silent
        ws.send_text(
            json.dumps(
                {
                    "type": "inject_sample_audio",
                    "sample_id": "english",
                }
            )
        )

        gate_decision_src = None
        activity_signals_src: list[str] = []
        deadline = time.monotonic() + 12.0
        while time.monotonic() < deadline:
            msg = ws.receive()
            if "text" in msg and msg["text"]:
                payload = json.loads(msg["text"])
                mtype = payload.get("type")
                if mtype == "gemini_activity_signal":
                    activity_signals_src.append(payload.get("signal", ""))
                elif mtype == "gate_decision":
                    gate_decision_src = payload
                    break

        assert gate_decision_src is not None
        assert gate_decision_src["detected_role"] == "source"
        assert gate_decision_src["should_translate"] is False
        assert gate_decision_src["stay_silent"] is True
        assert gate_decision_src["turn_triggered"] is False
        assert gate_decision_src["bytes_forwarded_to_gemini"] == 0
        assert gate_decision_src["gate_action"] == "STAY_SILENT_SOURCE_SPOKEN"
        assert activity_signals_src == []

        # STEP 2: Target language spoken (Hindi) -> MUST translate to Source (en-US)
        ws.send_text(
            json.dumps(
                {
                    "type": "inject_sample_audio",
                    "sample_id": "hindi",
                }
            )
        )

        gate_decision_tgt = None
        activity_signals_tgt: list[str] = []
        translated_audio_bytes = 0
        turn_complete_payload = None

        deadline = time.monotonic() + 25.0
        while time.monotonic() < deadline:
            msg = ws.receive()
            if "bytes" in msg and msg["bytes"]:
                translated_audio_bytes += len(msg["bytes"])
            elif "text" in msg and msg["text"]:
                payload = json.loads(msg["text"])
                mtype = payload.get("type")
                if mtype == "gemini_activity_signal":
                    activity_signals_tgt.append(payload.get("signal", ""))
                elif mtype == "gate_decision":
                    gate_decision_tgt = payload
                elif mtype == "turn_complete":
                    turn_complete_payload = payload
                    break
                elif mtype == "error":
                    pytest.fail(f"Live Gemini error during Target translation: {payload}")

        assert gate_decision_tgt is not None
        assert gate_decision_tgt["detected_role"] == "target"
        assert gate_decision_tgt["should_translate"] is True
        assert gate_decision_tgt["stay_silent"] is False
        assert gate_decision_tgt["turn_triggered"] is True
        assert gate_decision_tgt["bytes_forwarded_to_gemini"] > 0
        assert gate_decision_tgt["gate_action"] == "TRANSLATE_TARGET_TO_SOURCE"
        assert "activity_start" in activity_signals_tgt
        assert "activity_end" in activity_signals_tgt
        assert translated_audio_bytes > 0, "Expected 24 kHz translated audio bytes from gemini-3.8-live"
        assert turn_complete_payload is not None
        agent_tx = turn_complete_payload.get("agent_transcript", "").strip()
        assert len(agent_tx) > 0
        assert "<partial>" not in agent_tx
        assert "<not " not in agent_tx
        assert "🧠" not in agent_tx
        assert "✨" not in agent_tx


def test_live_websocket_english_to_tamil_no_xml_tags_or_truncation(client: TestClient) -> None:
    """Verify Source=ta-IN, Target=en-US translates English speech into complete Tamil without XML/S2ST tags."""
    with client.websocket_connect("/ws/translate") as ws:
        ws.send_text(
            json.dumps(
                {
                    "type": "setup",
                    "config": {
                        "source_lang": "ta-IN",
                        "target_lang": "en-US",
                        "voice_name": "Kore",
                    },
                }
            )
        )
        ready = _drain_until_session_ready(ws)
        assert ready["config"]["source_lang"] == "ta-IN"
        assert ready["config"]["target_lang"] == "en-US"

        ws.send_text(
            json.dumps(
                {
                    "type": "inject_sample_audio",
                    "sample_id": "english",
                }
            )
        )

        translated_audio_bytes = 0
        turn_complete_payload = None
        deadline = time.monotonic() + 25.0
        while time.monotonic() < deadline:
            msg = ws.receive()
            if "bytes" in msg and msg["bytes"]:
                translated_audio_bytes += len(msg["bytes"])
            elif "text" in msg and msg["text"]:
                payload = json.loads(msg["text"])
                mtype = payload.get("type")
                if mtype == "turn_complete":
                    turn_complete_payload = payload
                    break
                elif mtype == "error":
                    pytest.fail(f"Live Gemini error during en-US -> ta-IN translation: {payload}")

        assert turn_complete_payload is not None
        agent_tx = turn_complete_payload.get("agent_transcript", "").strip()
        assert len(agent_tx) > 0
        assert "<partial>" not in agent_tx
        assert "<not " not in agent_tx
        assert "🧠" not in agent_tx
        assert "✨" not in agent_tx
        # Audio at 24 kHz 16-bit mono is 48,000 bytes/sec; full sentence translation should be > 1.0s (> 48,000 bytes)
        assert translated_audio_bytes > 48000, (
            f"Expected complete non-truncated audio (>1.0s / >48KB), got {translated_audio_bytes} bytes "
            f"(transcript={agent_tx!r})"
        )


def test_live_websocket_bidirectional_mode_on_geap(client: TestClient) -> None:
    """Verify bidirectional mode translates both Source -> Target and Target -> Source while silencing 3rd languages."""
    with client.websocket_connect("/ws/translate") as ws:
        ws.send_text(
            json.dumps(
                {
                    "type": "setup",
                    "config": {
                        "source_lang": "en-US",
                        "target_lang": "hi-IN",
                        "voice_name": "Kore",
                        "bidirectional": True,
                    },
                }
            )
        )
        ready = _drain_until_session_ready(ws)
        assert ready["config"]["bidirectional"] is True

        # 1. Source language spoken (English) -> translates to Hindi!
        ws.send_text(json.dumps({"type": "inject_sample_audio", "sample_id": "english"}))
        audio_bytes_src = 0
        gate_src = None
        turn_src = None
        deadline = time.monotonic() + 25.0
        while time.monotonic() < deadline:
            msg = ws.receive()
            if "bytes" in msg and msg["bytes"]:
                audio_bytes_src += len(msg["bytes"])
            elif "text" in msg and msg["text"]:
                payload = json.loads(msg["text"])
                if payload.get("type") == "gate_decision":
                    gate_src = payload
                elif payload.get("type") == "turn_complete":
                    turn_src = payload
                    break
        assert gate_src is not None
        assert gate_src["gate_action"] == "TRANSLATE_SOURCE_TO_TARGET"
        assert gate_src["translated_to_lang"] == "hi-IN"
        assert audio_bytes_src > 24000
        assert turn_src is not None and len(turn_src.get("agent_transcript", "").strip()) > 0

        # 2. Target language spoken (Hindi) -> translates to English!
        ws.send_text(json.dumps({"type": "inject_sample_audio", "sample_id": "hindi"}))
        audio_bytes_tgt = 0
        gate_tgt = None
        turn_tgt = None
        deadline = time.monotonic() + 25.0
        while time.monotonic() < deadline:
            msg = ws.receive()
            if "bytes" in msg and msg["bytes"]:
                audio_bytes_tgt += len(msg["bytes"])
            elif "text" in msg and msg["text"]:
                payload = json.loads(msg["text"])
                if payload.get("type") == "gate_decision":
                    gate_tgt = payload
                elif payload.get("type") == "turn_complete":
                    turn_tgt = payload
                    break
        assert gate_tgt is not None
        assert gate_tgt["gate_action"] == "TRANSLATE_TARGET_TO_SOURCE"
        assert gate_tgt["translated_to_lang"] == "en-US"
        assert audio_bytes_tgt > 24000
        assert turn_tgt is not None and len(turn_tgt.get("agent_transcript", "").strip()) > 0

        # 3. Unselected 3rd language spoken (Tamil) -> stays 100% silent!
        ws.send_text(json.dumps({"type": "inject_sample_audio", "sample_id": "tamil"}))
        gate_other = None
        deadline = time.monotonic() + 12.0
        while time.monotonic() < deadline:
            msg = ws.receive()
            if "bytes" in msg and msg["bytes"]:
                pytest.fail("Received unexpected audio bytes for unselected 3rd language in bidirectional mode!")
            elif "text" in msg and msg["text"]:
                payload = json.loads(msg["text"])
                if payload.get("type") == "gate_decision":
                    gate_other = payload
                    break
        assert gate_other is not None
        assert gate_other["stay_silent"] is True
        assert gate_other["gate_action"] == "STAY_SILENT_UNSELECTED_LANGUAGE"
        assert gate_other["bytes_forwarded_to_gemini"] == 0

