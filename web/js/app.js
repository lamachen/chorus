/**
 * CHORUS · FastVLM Live-Vision-Sprachchat - komplett on-device.
 *
 * Es gibt genau eine Wahrheit in dieser App:
 *   - FastVLM-0.5B-ONNX (transformers.js/WebGPU, Fallback WASM) sieht dein Video
 *     und schreibt die Antworten - frei generiert, nichts geskriptet.
 *   - System-Spracherkennung hört dich, System-Sprachausgabe spricht sie.
 *   - Silero-VAD + Mikro-Pegel steuern Turn-Taking und Barge-in.
 *   - Alle Pegel im UI kommen aus echten Signalen (Mik-RMS, TTS-Word-Boundaries).
 * Kein Server, kein Mock, kein Regel-Charakter, kein Demo-Modus, kein
 * "laden"-Button: Mit dem Antippen eines Kontakts lädt das Modell sofort
 * (Fortschritt sichtbar im Klingel-Screen) und sie meldet sich von selbst.
 */

import * as UI from './ui.js';
import { MicVad } from './vad.js';
import { SystemTTS, SystemASR, rankVoices } from './speech.js';
import { Vision } from './vision.js';
import { OnDeviceVLM, onDeviceSupported, PRESET_MB, splitSentences } from './vlm_webgpu.js';
import { PERSONAS, getPersona, characterSystem, directives } from './persona.js';
import { Activation } from './activation.js';

const KEY = 'chorus-settings-v2';
const DEFAULTS = {
  lang: 'de-DE', persona: 'mara',
  voiceURI: '', rate: 1.0, pitch: 1.05, volume: 1,
  handsFree: true, vadOnset: 0.55, vadHangover: 620, halfDuplex: true, bargeIn: true,
  frameSize: 768, ratio: '1:1', intervalMs: 1100, quality: 0.8,
  previewMode: 'exact', pixelExact: false, cameraFacing: 'environment', mirrorMode: 'auto',
  proactive: true, activationThreshold: 0.62, activationCooldown: 22, subtitles: true,
  onDevicePreset: 'compact', turbo: false,
};

const load = () => { try { return { ...DEFAULTS, ...(JSON.parse(localStorage.getItem(KEY)) || {}) }; } catch { return { ...DEFAULTS }; } };

class App {
  constructor() {
    this.s = load();
    this.state = 'idle';
    this.persona = getPersona(this.s.persona);
    this.recents = [];
    this.pendingText = '';
    this.commitTimer = null;
    this.herBubble = null;
    this.herPending = '';
    this.ring = new UI.RingTone();
    this.vlm = null;
    this.vlmHistory = [];
    this.vlmBusy = false;
    this.greeted = false;
    this.perf = { ttfa: [] };
    this.commitAt = 0;
    this.lastExchange = 0;
    this.buildInfo = null;
    this.onDevice = null;
    this.act = new Activation({ threshold: this.s.activationThreshold, cooldown: this.s.activationCooldown, enabled: this.s.proactive });

    this.tts = new SystemTTS({
      onVoices: (v) => this.onVoices(v),
      onStart: (t) => this.onTtsStart(t),
      onEnd: () => this.onTtsEnd(false),
      onCancel: () => this.onTtsEnd(true),
      onSentenceEnd: (t) => this.subHer(t),
      onBoundary: (e) => this.onBoundary(e),
      onError: (e) => this.toastOnce('tts', 'Sprachausgabe: ' + e),
    });
    this.asr = new SystemASR({
      lang: this.s.lang,
      onFinal: (t) => this.onAsrFinal(t),
      onInterim: (t) => this.subYou(t),
      onState: (st) => this.setChip('chip-asr', 'ASR ' + st, st === 'listening'),
      onError: (e) => { if (e !== 'no-speech' && e !== 'aborted') this.setChip('chip-asr', 'ASR ' + e, false, true); },
      onFatal: (e) => UI.toast('Spracherkennung blockiert: ' + e, 5000),
    });
    this.vad = new MicVad({
      opts: { onset: this.s.vadOnset, hangoverMs: this.s.vadHangover },
      onSpeechStart: () => this.onSpeechStart(),
      onSpeechEnd: (e) => this.onSpeechEnd(e),
      onLevel: (rms, p) => this.onLevel(rms, p),
      onInfo: (i) => this.setChip('chip-vad', 'VAD ' + i.detector, i.detector === 'silero'),
      onError: (m) => this.setChip('chip-vad', 'VAD Fehler', false, true),
    });
    this.vision = new Vision({
      video: UI.$('cam'), grab: UI.$('grab'), view: UI.$('modelview'),
      size: this.s.frameSize, ratio: this.s.ratio, quality: this.s.quality,
      intervalMs: this.s.intervalMs, mirrorMode: this.s.mirrorMode,
      facing: this.s.cameraFacing, mode: this.s.previewMode,
      onFrame: () => {},                       // Frames bleiben on-device
      onStats: (st) => this.onVisionStats(st),
      onError: (e) => UI.toast('Kamera: ' + e, 4000),
    });
  }

  /* ================= Boot ================= */
  async boot() {
    UI.startClock(UI.$('sb-time'));
    this.buildWaves();
    this.applyPersisted();
    this.wireStatic();
    this.buildSettingsSchema();
    this.checkSecureContext();
    fetch('build_info.json').then(r => r.ok ? r.json() : null).then(j => { this.buildInfo = j; }).catch(() => {});
    this.probeOnDeviceInfo();
    if ('serviceWorker' in navigator) {
      navigator.serviceWorker.register('sw.js').then((reg) => {
        reg.addEventListener('updatefound', () => {
          const ni = reg.installing;
          ni && ni.addEventListener('statechange', () => {
            if (ni.state === 'installed' && navigator.serviceWorker.controller) UI.toast('Update geladen – Seite wechselt gleich automatisch');
          });
        });
      }).catch(() => {});
    }
  }

  probeOnDeviceInfo() { this.onDevice = onDeviceSupported(); this.renderInfo(); }

  checkSecureContext() {
    if (window.isSecureContext || ['localhost', '127.0.0.1'].includes(location.hostname)) return;
    const gate = UI.el('div', 'gate');
    gate.innerHTML = `<h2>Kamera, Mikro & On-Device-KI brauchen HTTPS</h2>
      <p>Diese Seite läuft über <code>${location.protocol}//${location.host}</code>. Browser geben
      Kamera/Mikro nur im secure context frei. Nutze <code>https://lamachen.github.io/chorus/web/index.html</code>
      oder einen lokalen HTTPS-Server.</p>`;
    document.getElementById('phone').appendChild(gate);
  }

  buildWaves() {
    const mk = (id, n) => { const w = UI.$(id); w.innerHTML = ''; for (let i = 0; i < n; i++) w.appendChild(UI.el('i')); };
    mk('wave-you', 10); mk('wave-her', 14);
  }

  /* ================= statische UI ================= */
  wireStatic() {
    UI.renderContacts(UI.$('fav-list'), PERSONAS, (p) => this.call(p));
    UI.renderContacts(UI.$('contact-list'), PERSONAS, (p) => this.call(p));
    this.renderRecents();
    UI.wireTabs([...document.querySelectorAll('.tab')], UI.$('dialer-title'),
      { fav: 'Favoriten', key: 'Keypad', recents: 'Verlauf', contacts: 'Kontakte' });
    UI.renderKeypad(UI.$('keypad'), (d) => {
      const n = UI.$('key-num');
      n.textContent = (n.textContent.trim() + d).slice(0, 16) || '\u00a0';
      UI.dtmf && UI.dtmf(d);
    });
    UI.$('key-del').onclick = () => { const n = UI.$('key-num'); n.textContent = n.textContent.trim().slice(0, -1) || '\u00a0'; };
    UI.$('key-call').onclick = () => this.call(this.persona);
    UI.$('btn-settings').onclick = () => this.openSettings();
    UI.$('btn-engine').onclick = () => this.openSettings();
    UI.$('settings-close').onclick = () => UI.$('settings').classList.remove('on');

    UI.$('btn-hangup').onclick = () => this.hangup();
    UI.$('ring-cancel').onclick = () => this.hangup(true);
    UI.$('btn-mute').onclick = (e) => this.toggle(e.currentTarget, (on) => {
      this.vad.setMuted(!on);
      if (on) { if (this.s.handsFree && this.asr.supported) this.asr.start(); } else this.asr.stop();
      UI.toast(on ? 'Mikro an' : 'Mikro stumm');
    });
    UI.$('btn-cam').onclick = (e) => this.toggle(e.currentTarget, (on) => {
      UI.$('mine').classList.toggle('off', !on);
      this.vision.setPaused(!on);
      UI.toast(on ? 'Kamera an – sie sieht dich wieder' : 'Kamera aus – sie sieht nichts mehr');
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
    UI.$('btn-flip').onclick = () => this.vision.flip().then((f) => {
      UI.toast(f === 'user' ? 'Frontkamera' : 'Rückkamera');
      UI.$('btn-torch').hidden = !this.vision.hasTorch;
      UI.$('btn-torch').dataset.on = 'false';
    });
    UI.$('btn-torch').onclick = () => this.vision.setTorch(!this.vision.torch).then((ok) => {
      UI.$('btn-torch').dataset.on = String(this.vision.torch);
      if (!ok) UI.toast('Kein Blitz an dieser Kamera verfügbar');
    });
    document.addEventListener('visibilitychange', () => { if (document.visibilityState === 'visible' && this.state === 'in-call') UI.keepAwake(true); });
    window.addEventListener('beforeunload', () => this.hangup(true));
    UI.$('sb-carrier').textContent = location.hostname === 'localhost' ? 'CHORUS · on-device' : 'CHORUS · ' + location.hostname;
  }

  toggle(btn, fn) { const on = btn.dataset.on !== 'true'; btn.dataset.on = String(on); fn(on); }
  renderRecents() { UI.renderRecents(UI.$('recent-list'), UI.$('recent-empty'), this.recents, (p) => this.call(p)); }

  /* ================= Anruf ================= */
  async call(persona) {
    if (this.state !== 'idle') return;
    if (!navigator.mediaDevices?.getUserMedia) { this.checkSecureContext(); return; }
    this.persona = persona || this.persona;
    this.s.persona = this.persona.id; this.save();
    this.state = 'dialing';
    this.greeted = false;
    this.vlmHistory = [];
    this.act.reset();
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
    UI.keepAwake(true);
    this.tts.unlock();
    this.tts.lang = this.s.lang;
    this.tts.autoVoice(this.persona.voiceNames || [], this.s.lang);
    if (this.s.voiceURI) this.tts.setVoiceURI(this.s.voiceURI);
    this.tts.pitch = this.persona.voicePitch ?? this.s.pitch;
    this.tts.rate = this.persona.voiceRate ?? this.s.rate;
    this.tts.volume = this.s.volume;

    const t0 = Date.now();
    const media = (async () => {
      try { await this.vision.start(); } catch (e) { UI.toast('Kamera nicht verfügbar: ' + e.message, 5000); }
      try { await this.vad.start(); } catch (e) {
        this.setChip('chip-vad', 'VAD nicht verfügbar', false, true);
        UI.toast('VAD nicht verfügbar: ' + e.message + ' – Spracherkennung übernimmt die Turn-Enden', 5000);
      }
    })();

    // On-Device-VLM: sofort, ohne Rückfrage, mit ehrlichem Fortschritt
    this.vlm = this.vlm && (this.vlm.loading || this.vlm.ready) ? this.vlm : new OnDeviceVLM({ preset: this.s.onDevicePreset });
    const prog = (msg, p) => {
      UI.$('ring-state').textContent = 'FastVLM lädt · ' + msg;
      const bar = UI.$('ring-prog');
      if (bar) bar.style.width = Math.round((p || 0) * 100) + '%';
      this._vlmProg = msg;
      this.renderInfo();
    };
    prog('Tokenizer + Vorverarbeitung…', 0);

    const failLoad = (msg) => {
      this._vlmError = msg;
      UI.$('ring-state').textContent = 'FastVLM konnte nicht geladen werden';
      const bar = UI.$('ring-prog'); if (bar) bar.style.width = '0%';
      UI.toast('On-Device-VLM: ' + msg, 6000);
      this.setChip('chip-vlm', 'VLM Fehler', false, true);
      this.ring.stop();
      clearInterval(this._stall);
    };
    this._vlmError = null;
    let lastMsg = '', lastMove = Date.now();
    clearInterval(this._stall);
    this._stall = setInterval(() => {
      if (!this.vlm || this.vlm.ready || this.state === 'idle') { clearInterval(this._stall); return; }
      if (this._vlmProg !== lastMsg) { lastMsg = this._vlmProg || ''; lastMove = Date.now(); }
      else if (Date.now() - lastMove > 15000) {
        failLoad('Ladung bleibt hängen – offline oder CDN blockiert?');
      }
    }, 1000);
    const vlmReady = this.vlm.load(prog).catch((e) => failLoad(String(e && e.message || e)));

    await media;
    await vlmReady;
    if (this.state !== 'dialing') return;                 // abgebrochen
    if (!this.vlm.ready) return;                          // ehrlich stehen bleiben
    if (Date.now() - t0 < 1600) await new Promise(r => setTimeout(r, 1600 - (Date.now() - t0)));
    this.enterCall();
  }

  enterCall() {
    this.state = 'in-call';
    this.ring.stop();
    UI.$('ringing').classList.remove('on');
    document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
    UI.$('screen-call').classList.add('active');
    this.t0 = Date.now();
    clearInterval(this._clockTimer);
    this._clockTimer = setInterval(() => { UI.$('call-timer').textContent = UI.fmtDur((Date.now() - this.t0) / 1000); }, 500);
    this.setChip('chip-vlm', `VLM ${this.vlm.device} · ${this.vlm.preset}`, true);
    this.setChip('chip-pro', this.s.proactive ? 'Initiative an' : 'Initiative aus', this.s.proactive);
    if (this.s.handsFree && this.asr.supported) this.asr.start();
    else this.setChip('chip-asr', 'ASR nicht verfügbar', false, true);
    this.addSys(`verbunden · FastVLM-0.5B on-device (${this.vlm.device})`);
    this.renderAudio();
    clearInterval(this._actTimer);
    this._actTimer = setInterval(() => this.proactiveTick(), 2500);
    this.lastExchange = performance.now();
    if (!this.greeted) { this.greeted = true; this.vlmTurn('greeting', ''); }
  }

  async hangup(silent = false) {
    if (this.state === 'idle') return;
    const seconds = this.t0 ? (Date.now() - this.t0) / 1000 : 0;
    if (seconds > 1) { this.recents.unshift({ persona: this.persona, seconds, at: Date.now(), missed: seconds < 3 }); this.recents = this.recents.slice(0, 30); this.renderRecents(); }
    this.state = 'idle';
    this.ring.stop();
    clearInterval(this._clockTimer);
    clearInterval(this._actTimer);
    clearInterval(this._stall);
    clearTimeout(this.commitTimer);
    cancelAnimationFrame(this._rafId);
    this.tts.cancel();
    this.asr.stop();
    this.vad.stop();
    this.vision.stop();
    this.vlm && this.vlm.cancel();
    this.vlmBusy = false;
    UI.keepAwake(false);
    UI.$('ringing').classList.remove('on');
    UI.$('her').classList.remove('speaking', 'listening');
    UI.$('transcript').classList.remove('on');
    UI.$('btn-panel').dataset.on = 'false';
    document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
    UI.$('screen-dialer').classList.add('active');
    this.pendingText = '';
    this.subHer(''); this.subYou('');
    if (!silent) UI.toast(`Anruf beendet · ${UI.fmtDur(seconds)}`);
  }

  /* ================= proactive (StreamBridge-Scorer) ================= */
  proactiveTick() {
    if (this.state !== 'in-call' || !this.s.proactive) return;
    if (this.vlmBusy || this.tts.speaking || this.vad?.gate?.speaking) return;
    const st = this.act.observe(this.vision.grab, {
      now: performance.now(),
      userSpeaking: this.vad?.gate?.speaking || false,
      sheSpeaking: this.tts.speaking,
      lastExchange: this.lastExchange,
    });
    this.setChip('chip-pro', st.speak ? 'Initiative: spricht!' : `Initiative ${(st.score * 100) | 0}%`, st.score > this.s.activationThreshold * 0.85);
    if (st.speak) { this.act.fire(); this.vlmTurn('proactive', ''); }
  }

  /* ================= der eine, echte Turn-Pfad ================= */
  async vlmTurn(reason, text) {
    if (!this.vlm || !this.vlm.ready || this.vlmBusy) return;
    this.vlmBusy = true;
    this._si = 0;
    this.herBubble = null; this.herPending = '';
    this.dispatch({ t: 'status', state: 'thinking', reason });
    const dir = directives(this.s.lang);
    const userText = reason === 'proactive' ? dir.proactive : reason === 'greeting' ? dir.greet : text;
    let sentBuf = '', all = '';
    const t0 = performance.now();
    try {
      const full = await this.vlm.generate({
        system: characterSystem(this.persona, this.s.lang),
        history: this.vlmHistory.slice(-6),
        userText,
        canvas: this.vision.grab,                 // exakt der angezeigte Frame
        maxTokens: this.s.turbo ? 48 : 96,
        onDelta: (tk) => {
          all += tk;
          this.dispatch({ t: 'delta', text: tk });
          sentBuf += tk;
          const r = splitSentences(sentBuf);
          r.out.forEach((snt) => { if (snt) this.dispatch({ t: 'sentence', text: snt, i: ++this._si }); });
          sentBuf = r.rest;
          if (this._si >= (this.s.turbo ? 2 : 3)) this.vlm.cancel();
        },
      });
      if (sentBuf.trim()) this.dispatch({ t: 'sentence', text: sentBuf.trim(), i: ++this._si });
      const clean = (full || all).trim();
      if (reason === 'user' && text) this.vlmHistory.push(['user', text]);
      else if (reason !== 'user') this.vlmHistory.push(['user', '(Stille)']);
      if (clean) this.vlmHistory.push(['assistant', clean]);
      this.dispatch({ t: 'turn_end', text: clean, reason, ms: performance.now() - t0 });
    } catch (e) {
      const msg = String(e && e.message || e);
      if (!/__cancelled__/.test(msg)) {
        this.dispatch({ t: 'error', message: 'VLM: ' + msg });
        this.setChip('chip-vlm', 'VLM Fehler', false, true);
      } else {
        this.dispatch({ t: 'turn_end', text: all.trim(), reason, ms: performance.now() - t0 });
      }
    }
    this.vlmBusy = false;
    this.lastExchange = performance.now();
    this.dispatch({ t: 'status', state: 'idle' });
  }

  dispatch(m) {
    switch (m.t) {
      case 'status':
        if (m.state === 'thinking') { UI.$('call-state').textContent = 'denkt nach…'; this.setChip('chip-vlm', 'VLM denkt…', true); }
        else { UI.$('call-state').textContent = this.s.handsFree ? 'hört zu' : 'verbunden'; if (this.vlm?.ready) this.setChip('chip-vlm', `VLM ${this.vlm.device}`, true); }
        break;
      case 'delta':
        if (!this.herBubble) this.herBubble = this.addMsg('her', this.persona.name, '');
        this.herPending += m.text;
        this.herBubble.querySelector('.txt').textContent = this.herPending.slice(0, 400);
        this.subHer(m.text);
        this.scrollTranscript();
        break;
      case 'sentence':
        this.tts.enqueue(m.text);
        this.subHer(m.text);
        break;
      case 'turn_end': {
        if (this.herBubble) this.herBubble.querySelector('.txt').textContent = m.text;
        else if (m.text) this.addMsg('her', this.persona.name, m.text);
        this.herBubble = null; this.herPending = '';
        this.subHer(m.text);
        if (m.ms) { this.perf.ttft = m.ms; this.setChip('chip-vlm', `VLM ${Math.round(m.ms)} ms · ${this.vlm?.device}`, true); }
        if (m.reason === 'proactive') this.addSys('sie hat von selbst gesprochen');
        this.renderInfo();
        this.scrollTranscript();
        break;
      }
      case 'error':
        UI.toast(m.message, 5000);
        break;
    }
  }

  /* ================= Hören: ASR + VAD + Barge-in ================= */
  onAsrFinal(text) {
    this.pendingText = (this.pendingText + ' ' + text).trim();
    this.subYou(this.pendingText);
    clearTimeout(this.commitTimer);
    this.commitTimer = setTimeout(() => this.commitNow(), this.vad?.gate?.speaking ? 900 : 620);
  }
  commitNow() {
    clearTimeout(this.commitTimer);
    const text = this.pendingText.trim();
    this.pendingText = '';
    this.subYou('');
    if (!text || this.state !== 'in-call') return;
    this.addMsg('you', 'Du', text);
    this.scrollTranscript();
    this.commitAt = performance.now();
    UI.$('call-state').textContent = 'denkt nach…';
    this.vlmTurn('user', text);
  }
  onSpeechStart(fromSpike = false) {
    if (this.state !== 'in-call') return;
    UI.$('her').classList.add('listening');
    if (this.tts.speaking && this.s.bargeIn) {
      if (fromSpike) this.addSys('barge-in über Mikro-Pegel');
      this.tts.cancel();
      this.vlm && this.vlm.cancel();
      this.vlmBusy = false;
      this.herBubble = null; this.herPending = '';
      if (this.s.halfDuplex) this.asr.setPausedForTTS(false);
    }
    if (!this.asr.wantRunning && this.s.handsFree && this.asr.supported && !this.tts.speaking) this.asr.start();
  }
  onSpeechEnd(e) {
    this.lastExchange = performance.now();
    UI.$('her').classList.remove('listening');
    if (e.tooShort) return;
    clearTimeout(this.commitTimer);
    this.commitTimer = setTimeout(() => this.commitNow(), 560);
  }
  onLevel(rms, p) {
    this.setChip('chip-asr', `ASR ${this.asr.wantRunning ? (this.asr.pausedForTTS ? 'wartet' : 'läuft') : 'aus'} · Mic ${(p * 100) | 0}%`, p > 0.4);
  }

  /* ================= echte Pegel ================= */
  renderAudio() {
    cancelAnimationFrame(this._rafId);
    const loop = () => {
      this._rafId = requestAnimationFrame(loop);
      if (this.state !== 'in-call') return;
      const now = performance.now();

      const lvl = this.vad?.running ? this.vad.level : 0;
      this._micHist = (this._micHist || []).concat(lvl).slice(-10);
      const yb = UI.$('wave-you').children;
      for (let i = 0; i < yb.length; i++) {
        const v = this._micHist[this._micHist.length - yb.length + i] || 0;
        yb[i].style.height = Math.max(2, Math.min(30, Math.pow(Math.min(1, v * 9), 0.7) * 30)) + 'px';
        yb[i].classList.toggle('idle', v < 0.004);
      }

      if (this.tts.speaking && !this._boundCount) this.herEnv = Math.max(this.herEnv || 0, 0.3);
      this.herEnv = (this.herEnv || 0) * 0.90;
      const hb = UI.$('wave-her').children;
      const shape = [0.45, 0.7, 1, 0.8, 0.6, 0.9, 1, 0.75, 0.55, 0.85, 0.95, 0.65, 0.5, 0.4];
      for (let i = 0; i < hb.length; i++) {
        hb[i].style.height = Math.max(2, Math.min(30, (this.herEnv || 0) * 30 * shape[i])) + 'px';
        hb[i].classList.toggle('idle', (this.herEnv || 0) < 0.03);
      }
      const ring = UI.$('voice-ring');
      if (ring) { ring.style.opacity = Math.min(0.85, (this.herEnv || 0) * 0.9); ring.style.transform = `scale(${0.6 + (this.herEnv || 0) * 0.5})`; }
      const prog = UI.$('herprog').firstElementChild;
      if (prog) prog.style.width = (this.tts.speaking && this._sentEst ? Math.min(1, (now - this._sentStart) / this._sentEst) * 100 : 0).toFixed(1) + '%';

      // Barge-in-Kanal 2: echter Pegel-Spike über dem adaptiven Noisefloor
      if (this.tts.speaking && this.s.bargeIn && this.vad?.running) {
        const floor = this.vad.noiseFloor || 0.012;
        if (lvl > Math.max(0.05, floor * 6)) this._spikeMs = (this._spikeMs || 0) + 16; else this._spikeMs = 0;
        if (this._spikeMs > 180 && now - (this._lastBarge || 0) > 1500) { this._lastBarge = now; this._spikeMs = 0; this.onSpeechStart(true); }
      } else this._spikeMs = 0;

      if (this.s.handsFree && this.asr.supported && this.asr.healthTick()) {
        this.setChip('chip-asr', 'ASR neu gestartet', false, true);
      }
    };
    loop();
  }

  onTtsStart(text) {
    if (this.commitAt) {
      const ttfa = performance.now() - this.commitAt;
      this.perf.ttfa.push(ttfa); this.perf.ttfa = this.perf.ttfa.slice(-12);
      this.commitAt = 0;
      this.setChip('chip-vlm', `Stimme nach ${(ttfa / 1000).toFixed(1)} s`, true);
      this.renderInfo();
    }
    this._boundCount = 0;
    this._sentStart = performance.now();
    this._sentEst = 600 + ((text || '').length / Math.max(0.4, this.tts.rate)) * 62;
    this.herEnv = 0.7;
    UI.$('her').classList.add('speaking');
    UI.$('her').classList.remove('listening');
    this.vad.setSheSpeaks(true);
    if (this.s.halfDuplex) this.asr.setPausedForTTS(true);
    UI.$('call-state').textContent = 'spricht…';
  }
  onTtsEnd() {
    this.herEnv = 0;
    const prog = UI.$('herprog') && UI.$('herprog').firstElementChild;
    if (prog) prog.style.width = '0%';
    UI.$('her').classList.remove('speaking');
    this.vad.setSheSpeaks(false);
    if (this.s.halfDuplex) this.asr.setPausedForTTS(false);
    UI.$('call-state').textContent = this.s.handsFree ? 'hört zu' : 'verbunden';
  }
  onBoundary(e) {
    this._boundCount = (this._boundCount || 0) + 1;
    this.herEnv = 1;
  }

  /* ================= Transcript / Untertitel / Chips ================= */
  addMsg(who, name, text) {
    const d = UI.el('div', 'msg ' + who, `<b>${name}</b><span class="txt"></span>`);
    d.querySelector('.txt').textContent = text;
    UI.$('transcript').appendChild(d);
    return d;
  }
  addSys(text) { UI.$('transcript').appendChild(UI.el('div', 'msg sys', text)); }
  scrollTranscript() { const b = UI.$('transcript'); b.scrollTop = b.scrollHeight; }
  subHer(t) { if (!this.s.subtitles) return; const n = UI.$('sub-her'); n.textContent = t || '\u00a0'; }
  subYou(t) { const n = UI.$('sub-you'); n.textContent = t ? '🎙 ' + t : '\u00a0'; }
  setChip(id, text, hot = false, warn = false) {
    const c = UI.$(id);
    if (!c) return;
    c.textContent = text;
    c.className = 'chip' + (warn ? ' hot' : hot ? ' live' : '');
  }
  toastOnce(key, msg) {
    const now = Date.now();
    if (this['_to_' + key] && now - this['_to_' + key] < 8000) return;
    this['_to_' + key] = now;
    UI.toast(msg, 4000);
  }
  onVisionStats(st) {
    UI.$('hud-frame').textContent = `frame #${st.id}`;
    UI.$('hud-size').textContent = `${st.w}×${st.h} · live`;
    UI.$('hud-rate').textContent = `${st.fps.toFixed(2)} fps → VLM`;
    const tb = UI.$('btn-torch');
    if (tb) { tb.hidden = !st.hasTorch; tb.dataset.on = String(st.torch); }
    UI.$('chip-vlm').title = `Kamera ${st.src} · ${st.facing} · ${st.mirror ? 'gespiegelt' : 'ungespiegelt'} · torch ${st.torch ? 'an' : 'aus'}`;
  }

  /* ================= Settings ================= */
  save() { localStorage.setItem(KEY, JSON.stringify(this.s)); }
  applyPersisted() {
    document.querySelector('.subs').style.display = this.s.subtitles ? '' : 'none';
    UI.$('btn-subs').dataset.on = String(this.s.subtitles);
    UI.$('modelview').classList.toggle('nosmooth', this.s.pixelExact);
    this.tts.rate = this.s.rate; this.tts.pitch = this.s.pitch; this.tts.volume = this.s.volume;
    this.tts.lang = this.s.lang;
  }

  buildSettingsSchema() {
    this._schema = [
      { group: 'Charakter' },
      { type: 'select', key: 'persona', label: 'Kontakt', options: PERSONAS.map(p => ({ value: p.id, label: `${p.name}, ${p.age} · ${p.city}` })) },
      { type: 'select', key: 'lang', label: 'Sprache', options: [{ value: 'de-DE', label: 'Deutsch' }, { value: 'en-US', label: 'English' }] },
      { type: 'switch', key: 'proactive', label: 'Sie redet von selbst', desc: 'Aktivierungs-Scorer (StreamBridge-Konzept): beobachtet die echten Frames und die Gesprächspause. WAS sie sagt, generiert immer das VLM.' },
      { type: 'range', key: 'activationThreshold', label: 'Initiative-Schwelle', attrs: { min: 0.3, max: 0.95, step: 0.01 }, fmt: v => (v * 100).toFixed(0) + '%' },
      { type: 'range', key: 'activationCooldown', label: 'Mindestabstand', attrs: { min: 5, max: 90, step: 1 }, fmt: v => v + ' s' },

      { group: 'Ihre Stimme (System-TTS)' },
      { type: 'select', key: 'voiceURI', label: 'Stimme', options: [{ value: '', label: 'automatisch (weiblich)' }] },
      { type: 'range', key: 'rate', label: 'Tempo', attrs: { min: 0.6, max: 1.6, step: 0.02 }, fmt: v => v.toFixed(2) + '×' },
      { type: 'range', key: 'pitch', label: 'Höhe', attrs: { min: 0.6, max: 1.6, step: 0.02 }, fmt: v => v.toFixed(2) },
      { type: 'range', key: 'volume', label: 'Lautstärke', attrs: { min: 0, max: 1, step: 0.05 }, fmt: v => (v * 100).toFixed(0) + '%' },

      { group: 'Dein Mikro' },
      { type: 'switch', key: 'handsFree', label: 'Hands-free (System-ASR)' },
      { type: 'range', key: 'vadOnset', label: 'VAD-Empfindlichkeit', attrs: { min: 0.15, max: 0.9, step: 0.01 }, fmt: v => v.toFixed(2) },
      { type: 'range', key: 'vadHangover', label: 'Pause bis Turn-Ende', attrs: { min: 200, max: 1500, step: 20 }, fmt: v => v + ' ms' },
      { type: 'switch', key: 'halfDuplex', label: 'Halbduplex', desc: 'Erkennung pausiert, während sie spricht. Barge-in bleibt über VAD und Pegel-Spike aktiv.' },
      { type: 'switch', key: 'bargeIn', label: 'Unterbrechen erlaubt' },

      { group: 'Bild (exakt das, was das VLM sieht)' },
      { type: 'select', key: 'frameSize', label: 'Auflösung', options: [384, 512, 640, 768, 1024].map(v => ({ value: v, label: v + ' px' })) },
      { type: 'select', key: 'ratio', label: 'Ausschnitt', options: [{ value: '1:1', label: 'Quadrat (wie das VLM croppt)' }, { value: '4:3', label: '4:3' }, { value: '16:9', label: '16:9' }] },
      { type: 'range', key: 'intervalMs', label: 'Frame-Intervall', attrs: { min: 300, max: 4000, step: 100 }, fmt: v => v + ' ms' },
      { type: 'range', key: 'quality', label: 'JPEG-Qualität (Anzeige)', attrs: { min: 0.4, max: 0.95, step: 0.05 }, fmt: v => (v * 100).toFixed(0) + '%' },
      { type: 'select', key: 'previewMode', label: 'Vorschau', options: [{ value: 'exact', label: 'exakt gesendete Frames' }, { value: 'fluid', label: 'flüssig + Frame-Marker' }] },
      { type: 'switch', key: 'pixelExact', label: 'Pixelhart skalieren' },
      { type: 'select', key: 'cameraFacing', label: 'Standard-Kamera', options: [{ value: 'environment', label: 'Rückkamera (Standard)' }, { value: 'user', label: 'Frontkamera' }] },
      { type: 'select', key: 'mirrorMode', label: 'Spiegelung', options: [{ value: 'auto', label: 'auto (Front ja, Rück nein)' }, { value: 'on', label: 'immer' }, { value: 'off', label: 'nie' }] },

      { group: 'On-Device-VLM' },
      { type: 'select', key: 'onDevicePreset', label: 'Modellgröße', options: [
        { value: 'compact', label: 'compact ≈ 676 MB' }, { value: 'balanced', label: 'balanced ≈ 812 MB' },
        { value: 'apple', label: 'apple-Mix ≈ 1,1 GB' }, { value: 'fp16', label: 'fp16 ≈ 1,5 GB' }] },
      { type: 'switch', key: 'turbo', label: 'Turbo (minimale Latenz)', desc: '48 statt 96 Tokens, max. 2 Sätze. Gilt ab dem nächsten Anruf.' },
      { type: 'info', key: 'vlm', rows: [['Status', '–']] },
      { type: 'info', key: 'perf', rows: [['TTFA', '–']] },
      { type: 'warn', html: 'Alles läuft on-device in deinem Browser: Frames, Audio und Text verlassen das Gerät nicht. Modell-Download einmalig vom Hugging-Face-CDN, danach Browser-Cache.' },
    ];
    UI.buildSettings(UI.$('settings-body'), this._schema, this.s, (k, v, i) => this.onSetting(k, v, i));
    this.refreshVoiceOptions();
    this.renderInfo();
  }

  onSetting(k, v) {
    this.s[k] = v;
    this.save();
    switch (k) {
      case 'rate': this.tts.rate = v; break;
      case 'pitch': this.tts.pitch = v; break;
      case 'volume': this.tts.volume = v; break;
      case 'voiceURI': this.tts.setVoiceURI(v) || this.tts.autoVoice(); break;
      case 'lang': this.tts.lang = v; this.asr.setLang(v); this.tts.autoVoice([], v); break;
      case 'persona': this.persona = getPersona(v); break;
      case 'proactive': this.act.tune({ enabled: v }); this.setChip('chip-pro', v ? 'Initiative an' : 'Initiative aus', v); break;
      case 'activationThreshold': this.act.tune({ threshold: v }); break;
      case 'activationCooldown': this.act.tune({ cooldown: v }); break;
      case 'vadOnset': this.vad.tune({ onset: v, offset: Math.max(0.15, v - 0.2) }); break;
      case 'vadHangover': this.vad.tune({ hangoverMs: v }); break;
      case 'handsFree': v ? (this.asr.supported && this.asr.start()) : this.asr.stop(); break;
      case 'subtitles': document.querySelector('.subs').style.display = v ? '' : 'none'; break;
      case 'pixelExact': UI.$('modelview').classList.toggle('nosmooth', v); break;
      case 'mirrorMode': this.vision.setMirrorMode(v); break;
      case 'cameraFacing': this.vision.facing = v; this.vision.restart().then(() => { UI.$('btn-torch').hidden = !this.vision.hasTorch; }); break;
      case 'previewMode': this.vision.setMode(v); break;
      case 'frameSize': this.vision.size = v; break;
      case 'ratio': this.vision.ratio = v; break;
      case 'quality': this.vision.quality = v; break;
      case 'intervalMs': this.vision.intervalMs = v; break;
      case 'onDevicePreset': UI.toast('Gilt ab dem nächsten Anruf'); break;
    }
  }

  onVoices(list) { this._voices = list; this.refreshVoiceOptions(); }
  refreshVoiceOptions() {
    if (!this._voices) return;
    const item = this._schema.find(x => x.key === 'voiceURI');
    if (!item) return;
    item.options = [{ value: '', label: 'automatisch (weiblich)' }].concat(
      rankVoices(this._voices, this.s.lang).slice(0, 40).map(v => ({ value: v.voiceURI, label: `${v.name} · ${v.lang}${v.localService ? '' : ' · cloud'}` })));
    UI.buildSettings(UI.$('settings-body'), this._schema, this.s, (k, v, i) => this.onSetting(k, v, i));
    this.renderInfo();
  }

  renderInfo() {
    const v = this.vlm;
    UI.updateInfoBlock(UI.$('settings-body'), 'vlm', [
      ['Modell', 'FastVLM-0.5B-ONNX'],
      ['Status', v ? (v.ready ? `bereit · ${v.device} · ${v.preset}` : v.loading ? 'lädt: ' + (this._vlmProg || '') : 'Fehler: ' + (v.error || this._vlmError || '')) : 'nicht geladen'],
      ['Ladezeit', v && v.ready ? (v.loadMs / 1000).toFixed(1) + ' s' : '–'],
      ['WebGPU', this.onDevice ? (this.onDevice.webgpu ? 'ja' : 'nein → WASM') : '…'],
      ['Gerät', this.onDevice ? `${this.onDevice.ua} · ${this.onDevice.cores} Kerne` : '…'],
    ]);
    const avg = (a) => a && a.length ? Math.round(a.reduce((x, y) => x + y, 0) / a.length) : null;
    UI.updateInfoBlock(UI.$('settings-body'), 'perf', [
      ['Ø TTFA (Redeende → Stimme)', avg(this.perf.ttfa) ? avg(this.perf.ttfa) + ' ms' : '–'],
      ['Letzte Antwort', this.perf.ttft ? Math.round(this.perf.ttft) + ' ms' : '–'],
      ['Frames erfasst', String(this.vision.frameId || 0)],
      ['Build', this.buildInfo ? `${this.buildInfo.stamp || ''} · ${this.buildInfo.note || ''}`.slice(0, 60) : '–'],
    ]);
  }

  openSettings() { this.renderInfo(); UI.$('settings').classList.add('on'); }
}

const app = new App();
window.__app = app;
app.boot();
