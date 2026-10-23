/**
 * Orchestrator: phone UI  ⇄  camera/Vision  ⇄  VAD  ⇄  system ASR/TTS  ⇄  FastVLM server
 *
 * Turn-taking loop (hands free):
 *   VAD speech start ──▶ (barge-in: stop her voice, tell the server to abort)
 *   VAD speech end   ──▶ commit timer
 *   ASR finals       ──▶ buffered into one utterance
 *   commit           ──▶ ws {t:"say"} ──▶ server streams sentences ──▶ TTS queue
 *   TTS end          ──▶ listening again  (half-duplex) / VAD-threshold back to normal
 */

import { MicVad } from './vad.js';
import { SystemTTS, SystemASR, rankVoices } from './speech.js';
import { Vision } from './vision.js';
import * as UI from './ui.js';

const KEY = 'fastvlm-call-settings-v1';

const DEFAULTS = {
  lang: 'de-DE',
  persona: 'mara',
  // system speech
  voiceURI: '', rate: 1.0, pitch: 1.05, volume: 1,
  // hands-free audio
  handsFree: true, neuralVad: true, vadOnset: 0.55, vadOffset: 0.35,
  vadHangover: 620, halfDuplex: true, bargeIn: true,
  // frames handed to FastVLM
  frameSize: 768, ratio: '1:1', intervalMs: 1100, quality: 0.8,
  mirror: true, previewMode: 'exact', pixelExact: false,
  // character
  proactive: true, activationMode: 'hybrid', activationThreshold: 0.62,
  activationCooldown: 22, subtitles: true,
  // optional external dialogue LLM
  llmEnable: false, llmBase: 'http://127.0.0.1:11434/v1', llmModel: 'llama3.1:8b', llmKey: '',
};

const load = () => {
  try { return { ...DEFAULTS, ...(JSON.parse(localStorage.getItem(KEY)) || {}) }; }
  catch { return { ...DEFAULTS }; }
};

class App {
  constructor() {
    this.s = load();
    this.state = 'idle';
    this.personas = [];
    this.persona = null;
    this.recents = [];
    this.server = null;
    this.ws = null;
    this.t0 = 0;
    this.pendingText = '';
    this.commitTimer = null;
    this.herBubble = null;
    this.ring = new UI.RingTone();
    this.lastTurnMs = null;
    this.tts = new SystemTTS({
      onVoices: (v) => this.onVoices(v),
      onStart: () => this.onTtsStart(),
      onEnd: () => this.onTtsEnd(),
      onCancel: () => this.onTtsEnd(true),
      onSentenceEnd: (t) => { this.subHer(t); },
      onBoundary: (e) => this.onBoundary(e),
      onError: (e) => { if (!this._ttsErrAt || Date.now() - this._ttsErrAt > 8000) { this._ttsErrAt = Date.now(); UI.toast('Sprachausgabe: ' + e); } },
    });
    this.asr = new SystemASR({
      lang: this.s.lang,
      onFinal: (t, c) => this.onAsrFinal(t, c),
      onInterim: (t) => this.subYou(t),
      onState: (st) => this.setChip('chip-asr', 'ASR ' + st, st === 'listening'),
      onError: (e) => { if (e !== 'no-speech' && e !== 'aborted') this.setChip('chip-asr', 'ASR ' + e, false, true); },
      onFatal: (e) => UI.toast('Mikro/Spracherkennung blockiert: ' + e),
      onSpeech: () => {},
    });
    this.vad = new MicVad({
      opts: {
        neural: this.s.neuralVad, onset: this.s.vadOnset, offset: this.s.vadOffset,
        hangoverMs: this.s.vadHangover,
      },
      onSpeechStart: () => this.onSpeechStart(),
      onSpeechEnd: (e) => this.onSpeechEnd(e),
      onLevel: (rms, p) => this.onLevel(rms, p),
      onInfo: (i) => this.setChip('chip-vad', 'VAD ' + i.detector + (i.barge ? ' ↑' : ''), i.detector === 'silero'),
      onError: (e) => UI.toast('VAD: ' + e),
    });
    this.vision = new Vision({
      video: UI.$('cam'), grab: UI.$('grab'), view: UI.$('modelview'),
      size: this.s.frameSize, ratio: this.s.ratio, quality: this.s.quality,
      intervalMs: this.s.intervalMs, mirror: this.s.mirror, mode: this.s.previewMode,
      onFrame: (id, buf, meta) => this.sendFrame(id, buf, meta),
      onStats: (st) => this.onVisionStats(st),
      onError: (e) => UI.toast('Kamera: ' + e),
    });
  }

  /* ================= boot ================= */
  async boot() {
    UI.startClock(UI.$('sb-time'));
    this.applyPersisted();
    this.wireStatic();
    this.buildSettingsSchema();
    await this.refreshStatus();
    setInterval(() => this.state === 'idle' && this.refreshStatus(), 6000);
    this.checkSecureContext();
    if ('serviceWorker' in navigator) navigator.serviceWorker.register('sw.js').catch(() => {});
  }

  checkSecureContext() {
    const ok = window.isSecureContext || ['localhost', '127.0.0.1'].includes(location.hostname);
    if (ok) return;
    const gate = UI.el('div', 'gate');
    gate.innerHTML = `<h2>Kamera &amp; Mikro brauchen HTTPS</h2>
      <p>Diese Seite wurde über <code>${location.protocol}//${location.host}</code> geladen.
      Browser erlauben Kamerazugriff nur in einem <i>secure context</i>.</p>
      <p>Starte den Server mit <code>./run.sh</code> (erzeugt ein selbstsigniertes Zertifikat)
      und akzeptiere die Warnung im Browser einmalig, oder öffne die Seite über
      <code>https://&lt;dein-rechner&gt;:8443</code>.</p>
      <button class="round green" id="gate-ok" style="width:auto;padding:0 22px;height:46px">Trotzdem weiter</button>`;
    document.getElementById('phone').appendChild(gate);
    gate.querySelector('#gate-ok').onclick = () => gate.remove();
  }

  async refreshStatus() {
    try {
      const r = await fetch('api/status');
      this.server = await r.json();
      const e = this.server.engine || {};
      const dot = UI.$('engine-dot'), lab = UI.$('engine-label');
      dot.className = 'dot ' + (e.ready ? 'ok' : e.loading ? 'load' : 'err');
      const short = String(e.model || '').split('/').pop();
      lab.textContent = e.ready ? `${e.backend} · ${short}` : e.loading ? `lädt ${short}…` : (e.backend + ' bereit');
      this.setChip('chip-vlm', 'VLM ' + (e.ready ? e.backend : '…'), !!e.ready);
      this.renderEngineInfo();
      if (!this.personas.length && this.server.personas?.length) {
        this.personas = this.server.personas;
        this.renderLists();
        if (this._schema) this.buildSettingsSchema();
      }
    } catch (err) {
      UI.$('engine-label').textContent = 'Server offline';
      UI.$('engine-dot').className = 'dot err';
    }
  }

  renderLists() {
    UI.renderContacts(UI.$('fav-list'), this.personas, (p) => this.call(p));
    UI.renderContacts(UI.$('contact-list'), this.personas, (p) => this.call(p));
    this.renderRecents();
  }
  renderRecents() {
    UI.renderRecents(UI.$('recent-list'), UI.$('recent-empty'), this.recents, (p) => this.call(p));
  }

  wireStatic() {
    UI.wireTabs([...document.querySelectorAll('.tab')], UI.$('dialer-title'),
      { fav: 'Favoriten', key: 'Keypad', recents: 'Verlauf', contacts: 'Kontakte' });
    UI.renderKeypad(UI.$('keypad'), (d) => {
      const n = UI.$('key-num');
      n.textContent = (n.textContent.trim() + d).slice(0, 16) || '\u00a0';
      try {
        const ctx = this._dtmfCtx || (this._dtmfCtx = new (window.AudioContext || window.webkitAudioContext)());
        const freqs = { '1': [697, 1209], '2': [697, 1336], '3': [697, 1477], '4': [770, 1209], '5': [770, 1336], '6': [770, 1477], '7': [852, 1209], '8': [852, 1336], '9': [852, 1477], '*': [941, 1209], '0': [941, 1336], '#': [941, 1477] };
        const f = freqs[d]; if (f) {
          const t = ctx.currentTime, g = ctx.createGain();
          g.gain.setValueAtTime(0.09, t); g.gain.exponentialRampToValueAtTime(0.0001, t + 0.16);
          g.connect(ctx.destination);
          f.forEach((hz) => { const o = ctx.createOscillator(); o.frequency.value = hz; o.connect(g); o.start(t); o.stop(t + 0.17); });
        }
      } catch {}
    });
    UI.$('key-del').onclick = () => {
      const n = UI.$('key-num');
      n.textContent = n.textContent.trim().slice(0, -1) || '\u00a0';
    };
    UI.$('key-call').onclick = () => this.call(this.currentPersona() || this.personas[0]);
    UI.$('btn-settings').onclick = () => this.openSettings();
    UI.$('settings-close').onclick = () => UI.$('settings').classList.remove('on');
    UI.$('btn-engine').onclick = () => this.openSettings('engine');

    // in-call controls
    UI.$('btn-hangup').onclick = () => this.hangup();
    UI.$('ring-cancel').onclick = () => this.hangup(true);
    UI.$('ring-accept').onclick = () => this.connectNow();
    UI.$('btn-mute').onclick = (e) => this.toggle(e.currentTarget, (on) => {
      this.vad.setMuted(!on); if (on) this.asr.start(); else this.asr.stop();
      UI.toast(on ? 'Mikro an' : 'Mikro stumm');
    });
    UI.$('btn-cam').onclick = (e) => this.toggle(e.currentTarget, (on) => {
      UI.$('mine').classList.toggle('off', !on);
      this.vision.setPaused(!on);
      this.note(on ? '(Sein Video ist wieder an, du siehst ihn wieder.)' : '(Sein Video ist jetzt aus, du siehst nichts mehr.)');
      UI.toast(on ? 'Kamera an' : 'Kamera aus – sie sieht nichts mehr');
    });
    UI.$('btn-subs').onclick = (e) => this.toggle(e.currentTarget, (on) => {
      this.s.subtitles = on; this.save();
      document.querySelector('.subs').style.display = on ? '' : 'none';
    });
    UI.$('btn-panel').onclick = (e) => this.toggle(e.currentTarget, (on) => {
      UI.$('transcript').classList.toggle('on', on);
      this.scrollTranscript();
    });
    UI.$('btn-freeze').onclick = () => {
      const p = !this.vision.paused;
      this.vision.setPaused(p);
      UI.$('btn-freeze').textContent = p ? 'Weiter' : 'Pause';
    };
    UI.$('modelview').onclick = () => this.vision.flip().then((f) =>
      UI.toast(f === 'user' ? 'Frontkamera' : 'Hauptkamera'));
    UI.$('sub-you').onclick = () => this.commitNow();

    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'visible' && this.state === 'in-call') UI.keepAwake(true);
    });
    window.addEventListener('beforeunload', () => this.hangup(true));
    UI.$('sb-carrier').textContent = location.hostname === 'localhost' ? 'FastVLM · lokal' : 'FastVLM · ' + location.hostname;
  }

  toggle(btn, fn) {
    const on = btn.dataset.on !== 'true';
    btn.dataset.on = String(on);
    fn(on);
  }

  currentPersona() {
    const id = this.s.persona;
    return this.personas.find(p => p.id === id) || null;
  }

  /* ================= call flow ================= */
  async call(persona) {
    if (this.state !== 'idle') return;
    if (!navigator.mediaDevices?.getUserMedia) {
      UI.toast('Kein Kamerazugriff – HTTPS nötig (siehe Hinweis)'); this.checkSecureContext(); return;
    }
    this.persona = persona || this.currentPersona() || { id: 'mara', name: 'Mara', hue: 326 };
    this.s.persona = this.persona.id; this.save();
    this.state = 'dialing';
    UI.setHue(document.documentElement, this.persona.hue);
    UI.setHue(UI.$('screen-call'), this.persona.hue);
    UI.$('call-name').textContent = this.persona.name;
    UI.$('ring-name').textContent = this.persona.name;
    UI.$('call-avatar').style.setProperty('--hue', this.persona.hue);
    UI.$('call-avatar').innerHTML = `<span>${this.persona.name[0]}</span>`;
    UI.$('ring-avatar').style.setProperty('--hue', this.persona.hue);
    UI.$('ring-avatar').innerHTML = `<span>${this.persona.name[0]}</span>`;
    UI.$('ring-state').textContent = 'FaceTime-Video · klingelt…';
    UI.$('call-state').textContent = 'verbindet…';
    UI.$('transcript').innerHTML = '';
    this.ring.start();
    UI.$('ringing').classList.add('on');
    UI.$('ring-accept').style.display = 'none';
    UI.keepAwake(true);

    // media + speech must be started from this gesture
    this.tts.unlock();
    try {
      await this.vision.start();
    } catch (e) {
      UI.toast('Kamera nicht verfügbar: ' + e.message, 5000);
    }
    try {
      const info = await this.vad.start();
      this.setChip('chip-vad', 'VAD ' + info.detector, info.detector === 'silero');
    } catch (e) {
      UI.toast('Mikro nicht verfügbar: ' + e.message, 5000);
      this.setChip('chip-vad', 'VAD aus', false, true);
    }
    this.connect();
    setTimeout(() => { if (this.state === 'dialing') this.connectNow(); }, 7000);
  }

  connect() {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const url = `${proto}://${location.host}/ws/call`;
    let ws;
    try { ws = new WebSocket(url); } catch (e) { UI.toast('WebSocket: ' + e.message); return; }
    ws.binaryType = 'arraybuffer';
    this.ws = ws;
    ws.onopen = () => {
      this.setChip('chip-vlm', 'VLM verbunden', true);
      ws.send(JSON.stringify({
        t: 'hello', persona: this.persona.id, lang: this.s.lang.slice(0, 2),
        proactive: this.s.proactive, activation_mode: this.s.activationMode,
        activation_threshold: this.s.activationThreshold,
      }));
    };
    ws.onmessage = (ev) => this.onWs(ev.data);
    ws.onerror = () => this.setChip('chip-vlm', 'VLM Fehler', false, true);
    ws.onclose = () => {
      this.setChip('chip-vlm', 'VLM getrennt', false, true);
      if (this.state === 'in-call') { UI.toast('Verbindung zum Server beendet'); this.hangup(true); }
    };
  }

  connectNow() {
    if (this.state !== 'dialing') return;
    this.state = 'in-call';
    this.ring.stop();
    UI.$('ringing').classList.remove('on');
    document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
    UI.$('screen-call').classList.add('active');
    this.t0 = Date.now();
    clearInterval(this._clockTimer);
    this._clockTimer = setInterval(() => {
      UI.$('call-timer').textContent = UI.fmtDur((Date.now() - this.t0) / 1000);
    }, 500);
    if (this.s.handsFree && this.asr.supported) this.asr.start();
    else if (!this.asr.supported) this.enableTextFallback();
    this.addSys(`verbunden · ${this.server?.engine?.backend || '?'} · ${this.server?.engine?.model?.split('/').pop() || ''}`);
  }

  enableTextFallback() {
    if (UI.$('text-input')) return;
    const box = UI.el('div');
    box.id = 'text-input';
    box.style.cssText = 'position:absolute;left:8px;right:8px;bottom:96px;display:flex;gap:6px;z-index:8';
    box.innerHTML = `<input type="text" placeholder="Tippen statt sprechen…" style="flex:1;background:#151a24;border:1px solid #232a38;color:#eef1f7;border-radius:12px;padding:10px 12px;font-size:14px">
      <button style="background:#3b9dff;color:#fff;border-radius:12px;padding:0 16px;font-weight:600">Senden</button>`;
    UI.$('screen-call').appendChild(box);
    const inp = box.querySelector('input'), btn = box.querySelector('button');
    const send = () => { const v = inp.value.trim(); if (!v) return; inp.value = ''; this.pushUser(v); };
    btn.onclick = send;
    inp.onkeydown = (e) => { if (e.key === 'Enter') send(); };
    UI.toast('Spracherkennung nicht verfügbar – Tippen aktiviert', 4500);
  }

  async hangup(silent = false) {
    if (this.state === 'idle') return;
    const seconds = this.t0 ? (Date.now() - this.t0) / 1000 : 0;
    if (this.persona && seconds > 1) {
      this.recents.unshift({ persona: this.persona, seconds, at: Date.now(), missed: seconds < 3 });
      this.recents = this.recents.slice(0, 30);
      this.renderRecents();
    }
    this.state = 'idle';
    this.ring.stop();
    clearInterval(this._clockTimer);
    clearTimeout(this.commitTimer);
    this.tts.cancel();
    this.asr.stop();
    this.vad.stop();
    this.vision.stop();
    UI.keepAwake(false);
    try { this.ws && this.ws.send(JSON.stringify({ t: 'bye' })); } catch {}
    try { this.ws && this.ws.close(); } catch {}
    this.ws = null;
    UI.$('ringing').classList.remove('on');
    UI.$('her').classList.remove('speaking', 'listening');
    UI.$('transcript').classList.remove('on');
    UI.$('btn-panel').dataset.on = 'false';
    document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
    UI.$('screen-dialer').classList.add('active');
    this.pendingText = '';
    this.subHer(''); this.subYou('');
    if (!silent) UI.toast(`Anruf beendet · ${UI.fmtDur(seconds)}`);
    this.refreshStatus();
  }

  /* ================= websocket ================= */
  onWs(raw) {
    let m; try { m = JSON.parse(raw); } catch { return; }
    switch (m.t) {
      case 'welcome':
        this.sid = m.sid; break;
      case 'ready': {
        this.serverReady = m;
        const e = m.engine || {};
        this.setChip('chip-vlm', `VLM ${e.backend}${e.ready ? '' : ' (lädt)'}`, !!e.ready, !e.ready);
        this.setChip('chip-pro', this.s.proactive ? 'Initiative ' + this.s.activationMode : 'Initiative aus', this.s.proactive);
        if (m.persona?.voice) {
          const v = m.persona.voice;
          this.tts.pitch = v.pitch ?? this.s.pitch;
          this.tts.rate = v.rate ?? this.s.rate;
          this.tts.volume = this.s.volume;
          this.tts.lang = this.s.lang;
          this.tts.autoVoice(v.names || [], this.s.lang);
          if (this.s.voiceURI) this.tts.setVoiceURI(this.s.voiceURI);
        }
        if (this.state === 'dialing') this.connectNow();
        break;
      }
      case 'status':
        if (m.state === 'thinking') { UI.$('call-state').textContent = 'denkt nach…'; this.setChip('chip-vlm', 'VLM denkt…', true); }
        else { UI.$('call-state').textContent = this.s.handsFree ? 'hört zu' : 'verbunden'; this.setChip('chip-vlm', 'VLM bereit', true); }
        break;
      case 'delta':
        if (!this.herBubble) this.herBubble = this.addMsg('her', this.persona?.name || 'Sie', '');
        this.herPending = (this.herPending || '') + m.text;
        this.herBubble.querySelector('.txt').textContent = this.herPending.slice(0, 400);
        this.subHer(m.text);
        this.scrollTranscript();
        break;
      case 'sentence':
        this.tts.enqueue(m.text);
        this.subHer(m.text);
        break;
      case 'turn_end': {
        this.lastTurnMs = m.ms;
        if (this.herBubble) this.herBubble.querySelector('.txt').textContent = m.text;
        else if (m.text) this.addMsg('her', this.persona?.name || 'Sie', m.text);
        this.herBubble = null; this.herPending = '';
        this.subHer(m.text);
        const e = this.server?.engine || {};
        this.setChip('chip-vlm', `VLM ${Math.round(m.ttft_ms || m.ms)} ms · ${e.backend || '?'}`, true);
        if (m.reason === 'proactive') this.addSys('sie hat von selbst gesprochen');
        this.scrollTranscript();
        break;
      }
      case 'cancelled':
        this.herBubble = null; this.herPending = ''; break;
      case 'activation':
        this.lastActivation = m;
        this.setChip('chip-pro', m.speak ? 'Initiative: spricht!' :
          `Initiative ${(m.score * 100) | 0}% · ${m.mode}`, m.score > this.s.activationThreshold * 0.85);
        break;
      case 'scene':
        this.lastScene = m.text;
        break;
      case 'probe_vote':
        break;
      case 'ttft_ms':
        this.setChip('chip-vlm', `VLM TTFT ${Math.round(m.ms)} ms`, true);
        break;
      case 'error':
        UI.toast('Server: ' + m.message, 4500);
        this.setChip('chip-vlm', 'VLM Fehler', false, true);
        break;
      case 'log':
        if (m.level !== 'info') console.warn('[server]', m.msg);
        break;
      case 'ack_frame':
        break;
      case 'pong': case 'keepalive':
        break;
    }
  }

  sendFrame(id, buf, meta) {
    if (!this.ws || this.ws.readyState !== 1) return;
    try {
      this.ws.send(JSON.stringify({ t: 'frame_meta', id, ts: meta.ts, w: meta.w, h: meta.h, bytes: meta.bytes }));
      this.ws.send(buf);
    } catch (e) { /* drop frame */ }
  }

  note(text) {
    if (this.ws?.readyState === 1) this.ws.send(JSON.stringify({ t: 'note', text }));
  }

  onVisionStats(st) {
    UI.$('hud-frame').textContent = `frame #${st.id}`;
    UI.$('hud-size').textContent = `${st.w}×${st.h} · ${st.kb} KB`;
    UI.$('hud-rate').textContent = `${st.fps.toFixed(2)} fps → modell`;
    UI.$('chip-vlm').title = `Kamera ${st.src} · ${st.mode} · ${st.mirror ? 'gespiegelt' : 'nicht gespiegelt'}`;
  }

  /* ================= voice loop ================= */
  onLevel(rms, p) {
    if (!this.s.subtitles) return;
    const bars = document.querySelectorAll('#wave i');
    const n = Math.min(bars.length, Math.round(p * bars.length));
    bars.forEach((b, i) => { if (!document.getElementById('her').classList.contains('speaking')) b.style.height = (i < n ? 4 + p * 26 * (0.6 + Math.random() * 0.6) : 4) + 'px'; });
    this.setChip('chip-asr', `ASR ${this.asr.wantRunning ? 'läuft' : 'aus'} · ${(p * 100) | 0}%`, p > 0.4);
  }

  onSpeechStart() {
    if (this.state !== 'in-call') return;
    UI.$('her').classList.add('listening');
    if (this.tts.speaking && this.s.bargeIn) {
      this.tts.cancel();
      this.ws?.readyState === 1 && this.ws.send(JSON.stringify({ t: 'interrupt', reason: 'barge-in' }));
      this.addSys('du hast sie unterbrochen');
      if (this.s.halfDuplex) this.asr.setPausedForTTS(false);
    }
    if (!this.asr.wantRunning && this.s.handsFree && !this.tts.speaking) this.asr.start();
  }

  onSpeechEnd(e) {
    UI.$('her').classList.remove('listening');
    if (e.tooShort) return;
    this.scheduleCommit(560);
  }

  scheduleCommit(ms) {
    clearTimeout(this.commitTimer);
    this.commitTimer = setTimeout(() => this.commitNow(), ms);
  }

  onAsrFinal(text) {
    this.pendingText = (this.pendingText + ' ' + text).trim();
    this.subYou(this.pendingText);
    this.scheduleCommit(this.vad.gate.speaking ? 900 : 620);
  }

  commitNow() {
    clearTimeout(this.commitTimer);
    const text = this.pendingText.trim();
    this.pendingText = '';
    if (!text) { this.subYou(''); return; }
    if (this.state !== 'in-call') return;
    this.pushUser(text);
  }

  pushUser(text) {
    this.addMsg('you', 'Du', text);
    this.subYou('');
    this.scrollTranscript();
    if (this.ws?.readyState === 1) this.ws.send(JSON.stringify({ t: 'say', text, final: true }));
    UI.$('call-state').textContent = 'denkt nach…';
  }

  /* ================= TTS state ================= */
  onTtsStart() {
    UI.$('her').classList.add('speaking');
    UI.$('her').classList.remove('listening');
    this.vad.setSheSpeaks(true);
    if (this.s.halfDuplex) this.asr.setPausedForTTS(true);
    UI.$('call-state').textContent = 'spricht…';
  }
  onTtsEnd(cancelled) {
    UI.$('her').classList.remove('speaking');
    this.vad.setSheSpeaks(false);
    if (this.s.halfDuplex && !cancelled) this.asr.setPausedForTTS(false);
    if (cancelled && this.s.halfDuplex) this.asr.setPausedForTTS(false);
    UI.$('call-state').textContent = this.s.handsFree ? 'hört zu' : 'verbunden';
    document.querySelectorAll('#wave i').forEach(b => b.style.height = '');
  }
  onBoundary(e) {
    const bars = document.querySelectorAll('#wave i');
    const i = Math.floor(Math.random() * bars.length);
    bars.forEach((b, k) => b.style.height = (Math.abs(k - i) < 4 ? 8 + Math.random() * 22 : 4) + 'px');
  }

  /* ================= transcript / subtitles ================= */
  addMsg(who, name, text) {
    const box = UI.$('transcript');
    const d = UI.el('div', 'msg ' + who, `<b>${name}</b><span class="txt"></span>`);
    d.querySelector('.txt').textContent = text;
    box.appendChild(d);
    return d;
  }
  addSys(text) {
    const box = UI.$('transcript');
    box.appendChild(UI.el('div', 'msg sys', text));
  }
  scrollTranscript() {
    const box = UI.$('transcript');
    box.scrollTop = box.scrollHeight;
  }
  subHer(t) {
    if (!this.s.subtitles) return;
    const n = UI.$('sub-her');
    n.textContent = t || '\u00a0';
    n.classList.toggle('blank', !t);
  }
  subYou(t) {
    const n = UI.$('sub-you');
    n.textContent = t ? '🎙 ' + t : '\u00a0';
    n.classList.toggle('blank', !t);
  }
  setChip(id, text, hot = false, warn = false) {
    const c = UI.$(id);
    if (!c || c.textContent === text) { if (c) { c.className = 'chip' + (warn ? ' hot' : hot ? ' live' : ''); } return; }
    c.textContent = text;
    c.className = 'chip' + (warn ? ' hot' : hot ? ' live' : '');
  }

  /* ================= settings ================= */
  save() { localStorage.setItem(KEY, JSON.stringify(this.s)); }
  applyPersisted() {
    document.querySelector('.subs').style.display = this.s.subtitles ? '' : 'none';
    UI.$('btn-subs').dataset.on = String(this.s.subtitles);
    UI.$('btn-cam').dataset.on = 'true';
    UI.$('btn-mute').dataset.on = 'true';
    UI.$('modelview').classList.toggle('nosmooth', this.s.pixelExact);
    this.tts.rate = this.s.rate; this.tts.pitch = this.s.pitch; this.tts.volume = this.s.volume;
    this.tts.lang = this.s.lang;
  }

  buildSettingsSchema() {
    const schema = [
      { group: 'Engine' },
      { type: 'info', key: 'engine', rows: [['Backend', '…'], ['Modell', '…'], ['Bereit', '…']] },
      { type: 'button', key: 'warm', label: 'Modell jetzt laden', buttonLabel: 'Laden' },
      { group: 'Charakter' },
      { type: 'select', key: 'persona', label: 'Kontakt', options: (this.personas.length ? this.personas : [{ id: 'mara', name: 'Mara' }]).map(p => ({ value: p.id, label: `${p.name}, ${p.age || ''} · ${p.city || ''}` })) },
      { type: 'select', key: 'lang', label: 'Sprache', options: [{ value: 'de-DE', label: 'Deutsch' }, { value: 'en-US', label: 'English' }] },
      { type: 'switch', key: 'proactive', label: 'Sie redet von selbst', desc: 'StreamBridge-Aktivierungsmodell: beobachtet dein Video und kommentiert spontan, wenn sich etwas tut oder Pause ist.' },
      { type: 'select', key: 'activationMode', label: 'Aktivierungsmodus', options: [{ value: 'hybrid', label: 'hybrid (CV + VLM-Probe)' }, { value: 'heuristic', label: 'nur Bildänderung (schnell)' }, { value: 'vlm', label: 'nur FastVLM-Entscheid' }, { value: 'off', label: 'aus' }] },
      { type: 'range', key: 'activationThreshold', label: 'Schwelle', attrs: { min: 0.3, max: 0.95, step: 0.01 }, fmt: v => (v * 100).toFixed(0) + '%' },
      { type: 'range', key: 'activationCooldown', label: 'Mindestabstand', attrs: { min: 5, max: 90, step: 1 }, fmt: v => v + ' s' },

      { group: 'Ihre Stimme (System-TTS)' },
      { type: 'select', key: 'voiceURI', label: 'Stimme', options: [{ value: '', label: 'automatisch (weiblich)' }] },
      { type: 'range', key: 'rate', label: 'Tempo', attrs: { min: 0.6, max: 1.6, step: 0.02 }, fmt: v => v.toFixed(2) + '×' },
      { type: 'range', key: 'pitch', label: 'Höhe', attrs: { min: 0.6, max: 1.6, step: 0.02 }, fmt: v => v.toFixed(2) },
      { type: 'range', key: 'volume', label: 'Lautstärke', attrs: { min: 0, max: 1, step: 0.05 }, fmt: v => (v * 100).toFixed(0) + '%' },

      { group: 'Dein Mikro (Hands-free)' },
      { type: 'switch', key: 'handsFree', label: 'Hands-free sprechen', desc: 'Freihändig: Voice-Activity-Detection + System-Spracherkennung. Ohne Spracherkennung erscheint ein Tippfeld.' },
      { type: 'switch', key: 'neuralVad', label: 'Neuronale VAD (Silero)', desc: 'Sonst adaptive Energie-/Nulldurchgangs-VAD. Wird automatisch genutzt, wenn das ONNX-Modell nicht lädt.' },
      { type: 'range', key: 'vadOnset', label: 'VAD-Empfindlichkeit', attrs: { min: 0.15, max: 0.9, step: 0.01 }, fmt: v => v.toFixed(2) },
      { type: 'range', key: 'vadHangover', label: 'Sprechpause bis Antwort', attrs: { min: 200, max: 1500, step: 20 }, fmt: v => v + ' ms' },
      { type: 'switch', key: 'halfDuplex', label: 'Halbduplex', desc: 'Spracherkennung pausiert, während sie spricht – verhindert, dass sie sich selbst hört. Unterbrechen (Barge-in) geht trotzdem.' },
      { type: 'switch', key: 'bargeIn', label: 'Unterbrechen erlaubt', desc: 'Wenn du lossprichst, während sie redet, bricht sie ab.' },

      { group: 'Bild für FastVLM' },
      { type: 'select', key: 'frameSize', label: 'Auflösung', options: [384, 512, 640, 768, 1024].map(v => ({ value: v, label: v + ' px' })) },
      { type: 'select', key: 'ratio', label: 'Ausschnitt', options: [{ value: '1:1', label: 'Quadrat (exakt wie FastVLM croppt)' }, { value: '4:3', label: '4:3' }, { value: '16:9', label: '16:9' }] },
      { type: 'range', key: 'intervalMs', label: 'Frame-Intervall', attrs: { min: 300, max: 4000, step: 100 }, fmt: v => v + ' ms (' + (1000 / v).toFixed(1) + ' fps)' },
      { type: 'range', key: 'quality', label: 'JPEG-Qualität', attrs: { min: 0.4, max: 0.95, step: 0.05 }, fmt: v => (v * 100).toFixed(0) + '%' },
      { type: 'switch', key: 'mirror', label: 'Selfie-Spiegelung', desc: 'Spiegelt Aufnahme UND Anzeige identisch – die Vorschau bleibt exakt das, was das Modell bekommt. Aus = das Modell kann Text besser lesen.' },
      { type: 'select', key: 'previewMode', label: 'Vorschau unten', options: [{ value: 'exact', label: 'exakte Modell-Frames (Standard)' }, { value: 'fluid', label: 'flüssiges Kamerabild + Frame-Marker' }] },
      { type: 'switch', key: 'pixelExact', label: 'Pixelhart skalieren', desc: 'Deaktiviert die Glättung der Vorschau, damit du 1:1 siehst, was im JPEG steht.' },

      { group: 'Dialog-Gehirn (optional)' },
      { type: 'switch', key: 'llmEnable', label: 'Externes LLM fürs Reden', desc: 'FastVLM liefert dann nur noch, was sie sieht; ein OpenAI-kompatibles Modell (z. B. Ollama) formuliert die Antworten. Deutlich bessere Gespräche als mit dem 0,5B-VLM allein.' },
      { type: 'text', key: 'llmBase', label: 'Base-URL', placeholder: 'http://127.0.0.1:11434/v1' },
      { type: 'text', key: 'llmModel', label: 'Modell', placeholder: 'llama3.1:8b' },
      { type: 'text', key: 'llmKey', label: 'API-Key', password: true, placeholder: 'leer = lokal' },
      { group: 'Hinweis' },
      { type: 'warn', html: 'Alles bleibt lokal: Frames, Audio und Text gehen nur an deinen eigenen Server. Die Sprachausgabe nutzt die im System installierten Stimmen.' },
    ];
    this._schema = schema;
    UI.buildSettings(UI.$('settings-body'), schema, this.s, (k, v, item) => this.onSetting(k, v, item));
    this.refreshVoiceOptions();
    this.renderEngineInfo();
  }

  onSetting(k, v) {
    if (k === 'warm') { fetch('api/warm', { method: 'POST' }).then(r => r.json()).then(() => this.refreshStatus()); return; }
    this.s[k] = v;
    this.save();
    switch (k) {
      case 'rate': this.tts.rate = v; break;
      case 'pitch': this.tts.pitch = v; break;
      case 'volume': this.tts.volume = v; break;
      case 'voiceURI': this.tts.setVoiceURI(v) || this.tts.autoVoice(); break;
      case 'lang':
        this.tts.lang = v; this.asr.setLang(v); this.tts.autoVoice([], v);
        if (this.ws?.readyState === 1) this.ws.send(JSON.stringify({ t: 'cfg', lang: v.slice(0, 2) }));
        break;
      case 'persona':
        if (this.ws?.readyState === 1) this.ws.send(JSON.stringify({ t: 'cfg', persona: v }));
        break;
      case 'proactive': case 'activationMode': case 'activationThreshold': case 'activationCooldown':
        if (this.ws?.readyState === 1) this.ws.send(JSON.stringify({
          t: 'cfg', proactive: this.s.proactive, activation_mode: this.s.activationMode,
          activation_threshold: this.s.activationThreshold, activation_cooldown: this.s.activationCooldown,
        }));
        break;
      case 'neuralVad': this.vad.opts.neural = v; UI.toast('VAD-Wechsel gilt ab dem nächsten Anruf'); break;
      case 'vadOnset': this.vad.tune({ onset: v, offset: Math.max(0.15, v - 0.2) }); break;
      case 'vadHangover': this.vad.tune({ hangoverMs: v }); break;
      case 'handsFree': v ? this.asr.start() : this.asr.stop(); break;
      case 'subtitles': document.querySelector('.subs').style.display = v ? '' : 'none'; break;
      case 'pixelExact': UI.$('modelview').classList.toggle('nosmooth', v); break;
      case 'mirror': this.vision.mirror = v; break;
      case 'previewMode': this.vision.setMode(v); break;
      case 'frameSize': this.vision.size = v; break;
      case 'ratio': this.vision.ratio = v; break;
      case 'quality': this.vision.quality = v; break;
      case 'intervalMs': this.vision.intervalMs = v; break;
      case 'llmEnable': case 'llmBase': case 'llmModel': case 'llmKey':
        UI.toast('LLM-Einstellungen gelten ab Server-Neustart (.env) – oder nutze /api/status');
        break;
    }
  }

  onVoices(list) {
    this._voices = list;
    this.refreshVoiceOptions();
  }

  refreshVoiceOptions() {
    if (!this._voices) return;
    const ranked = rankVoices(this._voices, this.s.lang);
    const item = this._schema.find(x => x.key === 'voiceURI');
    if (!item) return;
    item.options = [{ value: '', label: 'automatisch (weiblich)' }].concat(
      ranked.slice(0, 40).map(v => ({ value: v.voiceURI, label: `${v.name} · ${v.lang}${v.localService ? '' : ' · cloud'}` }))
    );
    UI.buildSettings(UI.$('settings-body'), this._schema, this.s, (k, v, i) => this.onSetting(k, v, i));
    this.renderEngineInfo();
  }

  renderEngineInfo() {
    const e = this.server?.engine || {}, l = this.server?.llm || {};
    UI.updateInfoBlock(UI.$('settings-body'), 'engine', [
      ['Backend', e.backend || '–'],
      ['Modell', (e.model || '–').toString().split('/').slice(-1)[0]],
      ['Status', e.ready ? 'bereit' : e.loading ? 'lädt…' : (e.error ? 'Fehler → Mock' : 'wartet')],
      ['Ladezeit', e.load_seconds ? e.load_seconds + ' s' : '–'],
      ['Dialog-LLM', l.enabled ? `${l.backend}:${l.model}` : 'aus (FastVLM spricht selbst)'],
      ['Letzte Szene', (this.lastScene || '–').slice(0, 60)],
      ['Antwort Ø', this.lastTurnMs ? Math.round(this.lastTurnMs) + ' ms' : '–'],
      ['Frames gesendet', String(this.vision.frameId || 0)],
      ['VAD', this.vad.info?.detector || '–'],
    ]);
  }

  openSettings(scrollTo) {
    this.renderEngineInfo();
    UI.$('settings').classList.add('on');
    if (scrollTo === 'engine') UI.$('settings-body').scrollTop = 0;
  }
}

const app = new App();
window.__app = app;
app.boot();
