(() => {
  const sourceSelect = document.getElementById('source-lang-select');
  const targetSelect = document.getElementById('target-lang-select');
  const swapBtn = document.getElementById('swap-langs-btn');
  const voiceSelect = document.getElementById('voice-select');
  const connectBtn = document.getElementById('connect-btn');
  const micBtn = document.getElementById('mic-btn');
  const micBtnLabel = document.getElementById('mic-btn-label');
  const clearStreamBtn = document.getElementById('clear-stream-btn');
  const settingsToggleBtn = document.getElementById('settings-toggle-btn');
  const settingsSheet = document.getElementById('settings-sheet');

  const wsStatusDot = document.getElementById('ws-status-dot');
  const wsStatusText = document.getElementById('ws-status-text');
  const geapProjectTag = document.getElementById('geap-project-tag');
  const vadStateTag = document.getElementById('vad-state-tag');

  const gateLatencyBadge = document.getElementById('gate-latency-badge');
  const gateBanner = document.getElementById('gate-banner');
  const gateBannerAction = document.getElementById('gate-banner-action');
  const gateBannerHeadline = document.getElementById('gate-banner-headline');
  const gateBannerReason = document.getElementById('gate-banner-reason');

  const probSourceLabel = document.getElementById('prob-source-label');
  const probSourceVal = document.getElementById('prob-source-val');
  const probSourceBar = document.getElementById('prob-source-bar');
  const probTargetLabel = document.getElementById('prob-target-label');
  const probTargetVal = document.getElementById('prob-target-val');
  const probTargetBar = document.getElementById('prob-target-bar');

  const mDetectedLang = document.getElementById('m-detected-lang');
  const mBytesSent = document.getElementById('m-bytes-sent');
  const mEmotion = document.getElementById('m-emotion');
  const mRate = document.getElementById('m-rate');
  const mPitch = document.getElementById('m-pitch');
  const mLoudness = document.getElementById('m-loudness');

  const streamList = document.getElementById('stream-list');

  let supportedLanguages = {};
  let ws = null;
  let wsReady = false;
  let activeTranslatedCard = null;
  let currentTurnAudioChunks = [];

  // 24 kHz playback context for Gemini 3.8 Live audio output
  let playbackCtx = null;
  let nextPlayTime = 0;

  // 16 kHz microphone capture state
  let micStream = null;
  let micCtx = null;
  let micNode = null;
  let isRecording = false;

  function ensurePlaybackContext() {
    if (!playbackCtx) {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      playbackCtx = new AudioCtx({ sampleRate: 24000 });
      nextPlayTime = playbackCtx.currentTime;
    }
    if (playbackCtx.state === 'suspended') {
      playbackCtx.resume().catch(() => {});
    }
    return playbackCtx;
  }

  ['pointerdown', 'keydown', 'click', 'change'].forEach((evtName) => {
    window.addEventListener(
      evtName,
      () => {
        ensurePlaybackContext();
      },
      { passive: true }
    );
  });

  function scheduleFloat32At24k(float32) {
    const ctx = ensurePlaybackContext();
    const dispatchBuffer = () => {
      const audioBuf = ctx.createBuffer(1, float32.length, 24000);
      audioBuf.getChannelData(0).set(float32);
      const source = ctx.createBufferSource();
      source.buffer = audioBuf;
      source.connect(ctx.destination);
      const now = ctx.currentTime;
      if (nextPlayTime < now) {
        nextPlayTime = now + 0.015;
      }
      source.start(nextPlayTime);
      nextPlayTime += audioBuf.duration;
    };

    if (ctx.state === 'suspended') {
      ctx.resume().then(dispatchBuffer).catch(dispatchBuffer);
    } else {
      dispatchBuffer();
    }
  }

  function playPcm24kChunk(arrayBuffer) {
    const int16 = new Int16Array(arrayBuffer);
    if (int16.length === 0) return;
    const float32 = new Float32Array(int16.length);
    for (let i = 0; i < int16.length; i++) {
      float32[i] = int16[i] / 32768.0;
    }
    currentTurnAudioChunks.push(float32);
    scheduleFloat32At24k(float32);

    if (activeTranslatedCard) {
      const statusBadge = activeTranslatedCard.querySelector('.audio-play-status');
      if (statusBadge) {
        const totalSamples = currentTurnAudioChunks.reduce((acc, c) => acc + c.length, 0);
        const durSec = (totalSamples / 24000).toFixed(1);
        statusBadge.textContent = `Speaking (${durSec}s)`;
      }
    }
  }

  function replayCombinedChunks(chunks) {
    if (!chunks || chunks.length === 0) return;
    const totalLen = chunks.reduce((acc, c) => acc + c.length, 0);
    const merged = new Float32Array(totalLen);
    let offset = 0;
    for (const c of chunks) {
      merged.set(c, offset);
      offset += c.length;
    }
    ensurePlaybackContext();
    nextPlayTime = playbackCtx.currentTime + 0.01;
    scheduleFloat32At24k(merged);
  }

  function updateStatusUI(state, text) {
    wsStatusDot.className = 'status-indicator';
    if (state === 'connected') {
      wsStatusDot.classList.add('connected');
    } else if (state === 'connecting') {
      wsStatusDot.classList.add('connecting');
    }
    wsStatusText.textContent = text;
  }

  function shortLangName(code) {
    const raw = supportedLanguages[code]?.name || code;
    return raw.split(' (')[0];
  }

  function updateProbabilityLabels() {
    const srcCode = sourceSelect.value || 'en-US';
    const tgtCode = targetSelect.value || 'hi-IN';
    probSourceLabel.textContent = `${shortLangName(srcCode)} (Silent)`;
    probTargetLabel.textContent = `${shortLangName(tgtCode)} (Translate)`;
  }

  function clearEmptyState() {
    const empty = document.getElementById('empty-stream-state');
    if (empty) {
      empty.remove();
    }
  }

  function handleGateDecision(d, isOfflineCheck = false) {
    gateLatencyBadge.textContent = `LID: ${d.inference_ms ?? '--'} ms`;

    const srcProb = (d.language_probabilities?.source_prob ?? 0) * 100;
    const tgtProb = (d.language_probabilities?.target_prob ?? 0) * 100;
    probSourceVal.textContent = `${srcProb.toFixed(0)}%`;
    probTargetVal.textContent = `${tgtProb.toFixed(0)}%`;
    probSourceBar.style.width = `${Math.min(100, srcProb)}%`;
    probTargetBar.style.width = `${Math.min(100, tgtProb)}%`;

    const pros = d.prosodia || {};
    mDetectedLang.textContent = `Lang: ${d.detected_language_name || d.detected_iso}`;
    mBytesSent.textContent = `${(d.bytes_forwarded_to_gemini || 0).toLocaleString()} B`;
    mEmotion.textContent = `Emotion: ${pros.emotion || '--'}`;
    mRate.textContent = `Pace: ${pros.speaking_rate || '--'}`;
    mPitch.textContent = `Pitch: ${pros.pitch_direction || '--'}`;
    mLoudness.textContent = `Loudness: ${pros.loudness_level || '--'}`;

    gateBanner.className = 'live-gate-pill';
    if (d.should_translate) {
      gateBanner.classList.add('state-translate');
      gateBannerAction.textContent = `TRANSLATING · ${d.inference_ms} ms`;
      gateBannerHeadline.textContent = `${d.detected_language_name} -> ${shortLangName(d.source_lang)}`;
    } else {
      gateBanner.classList.add('state-silent');
      gateBannerAction.textContent = `SILENT · ${d.inference_ms} ms`;
      gateBannerHeadline.textContent = `${d.detected_language_name} detected · 0 B sent to Gemini`;
    }
    gateBannerReason.textContent = d.reason || '';

    clearEmptyState();
    const card = document.createElement('div');
    const isTranslated = Boolean(d.should_translate);
    card.className = `turn-card ${isTranslated ? 'card-translated' : 'card-suppressed'}`;

    if (isTranslated) {
      currentTurnAudioChunks = [];
      card.innerHTML = `
        <div class="card-top-row">
          <span class="card-pill pill-translated">
            ${d.detected_language_name} -> ${shortLangName(d.source_lang)}
          </span>
          <span class="card-meta-inline audio-play-status">LID ${d.inference_ms} ms</span>
        </div>
        <div class="source-utterance-line tx-user-slot">Listening to ${d.detected_language_name} speech...</div>
        <div class="translated-utterance-line tx-agent-slot">${isOfflineCheck ? 'Offline gate check passed' : 'Translating...'}</div>
        <div class="card-footer-row">
          <span class="prosody-inline-tag">
            Prosodia: ${pros.emotion || 'neutral'} · ${pros.speaking_rate || 'medium'} pace · ${pros.pitch_direction || 'level'} pitch
          </span>
          <button type="button" class="replay-pill-btn replay-audio-btn" style="display:none;">
            Replay
          </button>
        </div>
      `;
      if (!isOfflineCheck) {
        activeTranslatedCard = card;
      }
    } else {
      card.innerHTML = `
        <div class="card-top-row" style="margin-bottom:4px;">
          <span class="card-pill pill-suppressed">
            ${d.detected_language_name} · Stayed Silent
          </span>
          <span class="card-meta-inline">0 B sent · ${d.inference_ms} ms</span>
        </div>
        <div class="suppressed-note">
          Prosodia recognized ${d.detected_role === 'source' ? 'your Source language' : 'an unselected language'} (${Math.max(srcProb, 100 - tgtProb).toFixed(0)}%) and suppressed translation.
        </div>
      `;
    }

    streamList.prepend(card);
    streamList.scrollTop = 0;
  }

  function connectWebSocket(onReadyCallback = null) {
    if (ws) {
      try {
        ws.close();
      } catch (_) {}
    }

    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${proto}//${window.location.host}/ws/translate`);
    ws.binaryType = 'arraybuffer';
    wsReady = false;
    updateStatusUI('connecting', 'Connecting...');

    ws.onopen = () => {
      ws.send(
        JSON.stringify({
          type: 'setup',
          config: {
            source_lang: sourceSelect.value,
            target_lang: targetSelect.value,
            voice_name: voiceSelect.value,
          },
        })
      );
    };

    ws.onmessage = (event) => {
      if (event.data instanceof ArrayBuffer) {
        playPcm24kChunk(event.data);
        return;
      }
      try {
        const msg = JSON.parse(event.data);
        const mtype = msg.type;
        if (mtype === 'session_ready') {
          wsReady = true;
          updateStatusUI(
            'connected',
            `gemini-3.8-live · ${shortLangName(msg.config.target_lang)} -> ${shortLangName(msg.config.source_lang)}`
          );
          if (typeof onReadyCallback === 'function') {
            const cb = onReadyCallback;
            onReadyCallback = null;
            cb();
          }
        } else if (mtype === 'silero_vad_Prob' || mtype === 'silero_vad_event') {
          vadStateTag.textContent = `VAD: ${msg.state || 'IDLE'}`;
        } else if (mtype === 'gate_decision') {
          handleGateDecision(msg, false);
        } else if (mtype === 'user_transcript') {
          if (activeTranslatedCard) {
            const slot = activeTranslatedCard.querySelector('.tx-user-slot');
            if (slot) slot.textContent = msg.text;
          }
        } else if (mtype === 'agent_transcript') {
          if (activeTranslatedCard) {
            const slot = activeTranslatedCard.querySelector('.tx-agent-slot');
            if (slot) slot.textContent = msg.text;
          }
        } else if (mtype === 'turn_complete') {
          if (activeTranslatedCard) {
            const uSlot = activeTranslatedCard.querySelector('.tx-user-slot');
            const aSlot = activeTranslatedCard.querySelector('.tx-agent-slot');
            if (uSlot && msg.user_transcript) uSlot.textContent = msg.user_transcript;
            if (aSlot && msg.agent_transcript) aSlot.textContent = msg.agent_transcript;

            const savedChunks = currentTurnAudioChunks.slice();
            const replayBtn = activeTranslatedCard.querySelector('.replay-audio-btn');
            const statusBadge = activeTranslatedCard.querySelector('.audio-play-status');
            if (savedChunks.length > 0 && replayBtn) {
              const totalSamples = savedChunks.reduce((acc, c) => acc + c.length, 0);
              const durSec = (totalSamples / 24000).toFixed(1);
              if (statusBadge) statusBadge.textContent = `${durSec}s audio`;
              replayBtn.style.display = 'inline-flex';
              replayBtn.onclick = () => replayCombinedChunks(savedChunks);
            }
          }
        } else if (mtype === 'error') {
          updateStatusUI('disconnected', `Error: ${msg.message}`);
        }
      } catch (err) {
        console.error('WS parse error:', err);
      }
    };

    ws.onclose = () => {
      wsReady = false;
      updateStatusUI('disconnected', 'Offline · Tap to reconnect');
    };
  }

  function pushConfigUpdate() {
    ensurePlaybackContext();
    const srcIso = supportedLanguages[sourceSelect.value]?.iso;
    const tgtIso = supportedLanguages[targetSelect.value]?.iso;
    if (srcIso && tgtIso && srcIso === tgtIso) {
      const fallback = Object.keys(supportedLanguages).find(
        (k) => supportedLanguages[k].iso !== srcIso
      );
      if (fallback) {
        targetSelect.value = fallback;
      }
    }
    updateProbabilityLabels();
    if (ws && ws.readyState === WebSocket.OPEN) {
      wsReady = false;
      updateStatusUI('connecting', 'Updating pair...');
      ws.send(
        JSON.stringify({
          type: 'update_config',
          config: {
            source_lang: sourceSelect.value,
            target_lang: targetSelect.value,
            voice_name: voiceSelect.value,
          },
        })
      );
    } else {
      connectWebSocket();
    }
  }

  async function toggleMicrophone() {
    if (isRecording) {
      stopMicrophone();
      return;
    }

    ensurePlaybackContext();
    if (!ws || ws.readyState !== WebSocket.OPEN || !wsReady) {
      connectWebSocket(() => startMicrophoneCapture());
    } else {
      startMicrophoneCapture();
    }
  }

  function downsampleBuffer(buffer, inRate, outRate) {
    if (inRate === outRate) return buffer;
    const ratio = inRate / outRate;
    const newLen = Math.round(buffer.length / ratio);
    const result = new Float32Array(newLen);
    for (let i = 0; i < newLen; i++) {
      const idx = Math.min(buffer.length - 1, Math.floor(i * ratio));
      result[i] = buffer[idx];
    }
    return result;
  }

  async function startMicrophoneCapture() {
    try {
      micStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 16000,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      micCtx = new AudioCtx({ sampleRate: 16000 });
      const actualRate = micCtx.sampleRate || 16000;
      const source = micCtx.createMediaStreamSource(micStream);

      let pendingSamples = new Float32Array(0);
      const send512Frames = (floatSamples16k) => {
        if (!isRecording || !ws || ws.readyState !== WebSocket.OPEN || !wsReady) return;
        const combined = new Float32Array(pendingSamples.length + floatSamples16k.length);
        combined.set(pendingSamples, 0);
        combined.set(floatSamples16k, pendingSamples.length);

        let offset = 0;
        while (offset + 512 <= combined.length) {
          const frame = combined.subarray(offset, offset + 512);
          offset += 512;
          const pcm16 = new Int16Array(512);
          for (let i = 0; i < 512; i++) {
            const s = Math.max(-1, Math.min(1, frame[i]));
            pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
          }
          ws.send(pcm16.buffer);
        }
        pendingSamples = combined.slice(offset);
      };

      const proc = micCtx.createScriptProcessor(512, 1, 1);
      proc.onaudioprocess = (e) => {
        const input = e.inputBuffer.getChannelData(0);
        const resampled = downsampleBuffer(input, actualRate, 16000);
        send512Frames(resampled);
      };

      source.connect(proc);
      proc.connect(micCtx.destination);
      micNode = proc;
      isRecording = true;
      micBtn.classList.add('recording');
      micBtnLabel.textContent = 'Listening · Tap to Stop';
      gateBannerAction.textContent = 'MIC LIVE';
      gateBannerHeadline.textContent = `Speak ${shortLangName(targetSelect.value)} to translate, or ${shortLangName(sourceSelect.value)} for silence`;
    } catch (err) {
      alert(`Microphone access error: ${err.message}`);
    }
  }

  function stopMicrophone() {
    isRecording = false;
    micBtn.classList.remove('recording');
    micBtnLabel.textContent = 'Tap to Listen';
    if (micNode) {
      micNode.disconnect();
      micNode = null;
    }
    if (micCtx) {
      micCtx.close().catch(() => {});
      micCtx = null;
    }
    if (micStream) {
      micStream.getTracks().forEach((t) => t.stop());
      micStream = null;
    }
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'audio_stream_end' }));
    }
  }

  async function initApp() {
    try {
      const healthResp = await fetch('/api/health');
      const health = await healthResp.json();

      supportedLanguages = health.supported_languages || {};
      geapProjectTag.textContent = `${health.platform || 'Agent Platform (formerly known as Vertex AI)'} · ${health.location}`;

      sourceSelect.innerHTML = '';
      targetSelect.innerHTML = '';
      Object.entries(supportedLanguages).forEach(([code, info]) => {
        const optSrc = document.createElement('option');
        optSrc.value = code;
        optSrc.textContent = info.name;
        sourceSelect.appendChild(optSrc);

        const optTgt = document.createElement('option');
        optTgt.value = code;
        optTgt.textContent = info.name;
        targetSelect.appendChild(optTgt);
      });

      sourceSelect.value = health.default_source_lang || 'en-US';
      targetSelect.value = health.default_target_lang || 'hi-IN';
      updateProbabilityLabels();
      connectWebSocket();
    } catch (err) {
      console.error('Initialization failed:', err);
    }
  }

  sourceSelect.addEventListener('change', pushConfigUpdate);
  targetSelect.addEventListener('change', pushConfigUpdate);
  voiceSelect.addEventListener('change', pushConfigUpdate);

  swapBtn.addEventListener('click', () => {
    const prevSrc = sourceSelect.value;
    sourceSelect.value = targetSelect.value;
    targetSelect.value = prevSrc;
    pushConfigUpdate();
  });

  if (settingsToggleBtn && settingsSheet) {
    settingsToggleBtn.addEventListener('click', () => {
      const isHidden = settingsSheet.classList.toggle('hidden');
      settingsToggleBtn.classList.toggle('active', !isHidden);
      settingsToggleBtn.setAttribute('aria-expanded', String(!isHidden));
    });
  }

  connectBtn.addEventListener('click', () => {
    ensurePlaybackContext();
    connectWebSocket();
  });
  micBtn.addEventListener('click', toggleMicrophone);
  clearStreamBtn.addEventListener('click', () => {
    streamList.innerHTML = `
      <div class="empty-stage" id="empty-stream-state">
        <div class="empty-icon-ring">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z" />
            <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
            <line x1="12" x2="12" y1="19" y2="22" />
          </svg>
        </div>
        <p class="empty-headline">Unidirectional Speech Interpreter</p>
        <p class="empty-subtext">
          Target speech is translated into your Source language in real time. When you speak the Source language, the interpreter stays silent.
        </p>
      </div>
    `;
  });

  initApp();
})();
