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
import { OnDeviceVLM, onDeviceSupported, PRESET_MB, splitSentences, looksGibberish } from './vlm_webgpu.js';
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
  onDevicePreset: 'apple', onDeviceDevice: 'auto', turbo: false,
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
    this.cameraOn = true;
    this.memo = { summary: '', rounds: 0 };
    this.memoBusy = false;
    this.thinkStart = 0;
    this._gotFirst = false;
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
    window.addEventListener('resize', () => { if (this.state === 'in-call') this.setupWaveform(); });
    this.applyPersisted();
    this.wireStatic();
    this.buildSettingsSchema();
    this.checkSecureContext();
    fetch('build_info.json').then(r => r.ok ? r.json() : null).then(j => { this.buildInfo = j; this.renderInfo(); }).catch(() => {});
    this.probeOnDeviceInfo();
    this.preloadVlm();
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

  /** On-Device-VLM sofort beim Seitenstart laden - kein Button, kein Dialog. */
  preloadVlm() {
    this.vlm = this.vlm || new OnDeviceVLM({ preset: this.s.onDevicePreset, device: this.s.onDeviceDevice || 'auto' });
    const label = (txt, cls) => {
      UI.$('engine-label').textContent = txt;
      UI.$('engine-dot').className = 'dot ' + cls;
    };
    if (this.vlm.ready) { label('VLM bereit', 'ok'); return; }
    label('VLM lädt…', 'load');
    this.vlm.load((msg, p) => {
      label(`VLM ${(p * 100) | 0}% · ${String(msg).slice(0, 26)}`, 'load');
      this._vlmProg = msg;
      this._vlmP = p || 0;
    }).then(() => {
      label(`VLM bereit · ${this.vlm.device}`, 'ok');
      this.setChip('chip-vlm', `VLM ${this.vlm.device} · ${this.vlm.preset}`, true);
      this.vlm.warmup && this.vlm.warmup();
      this.renderInfo();
    }).catch((e) => {
      this._vlmError = String(e && e.message || e);
      label('VLM Fehler', 'err');
      this.renderInfo();
    });
  }

  checkSecureContext() {
    if (window.isSecureContext || ['localhost', '127.0.0.1'].includes(location.hostname)) return;
    const gate = UI.el('div', 'gate');
    gate.innerHTML = `<h2>Kamera, Mikro & On-Device-KI brauchen HTTPS</h2>
      <p>Diese Seite läuft über <code>${location.protocol}//${location.host}</code>. Browser geben
      Kamera/Mikro nur im secure context frei. Nutze <code>https://lamachen.github.io/chorus/web/index.html</code>
      oder einen lokalen HTTPS-Server.</p>`;
    document.getElementById('phone').appendChild(gate);
  }

  setupWaveform() {
    const c = UI.$('waveform');
    if (!c) return;
    const dpr = Math.min(1.5, window.devicePixelRatio || 1);
    c.width = Math.max(1, Math.round(c.clientWidth * dpr));
    c.height = Math.max(1, Math.round(c.clientHeight * dpr));
    this._wf = c;
    this._wfBars = Math.max(28, Math.floor(c.clientWidth / 6));
    this.micHist = new Array(this._wfBars).fill(0);
    this.herHist = new Array(this._wfBars).fill(0);
  }

  /** gespiegelte Echtzeit-Waveform: oben = ihre TTS-Hüllkurve, unten = dein Mikro-RMS */
  drawWave() {
    const c = this._wf;
    if (!c || !c.width) return;
    const ctx = c.getContext('2d');
    const w = c.width, h = c.height, mid = h / 2;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = 'rgba(255,255,255,.18)';
    ctx.fillRect(0, mid - 0.75, w, 1.5);
    const n = this.micHist.length, bw = w / n;
    const hue = this.persona?.hue ?? 326;
    for (let i = 0; i < n; i++) {
      const mv = Math.min(1, (this.micHist[i] || 0) * 9);
      const hv = Math.min(1, this.herHist[i] || 0);
      const mh = Math.pow(mv, 0.7) * (mid - 6);
      const hh = hv * (mid - 6);
      const rr = (x, y, ww, h2) => { ctx.beginPath(); if (ctx.roundRect) ctx.roundRect(x, y, ww, h2, 2); else ctx.rect(x, y, ww, h2); ctx.fill(); };
      ctx.fillStyle = mv < 0.004 ? 'rgba(59,157,255,.25)' : `rgba(59,157,255,${0.45 + mv * 0.55})`;
      rr(i * bw + 1, mid + 2, bw - 2, Math.max(1.5, mh));
      ctx.fillStyle = `hsl(${hue} 85% 66% / ${0.3 + hv * 0.7})`;
      rr(i * bw + 1, mid - 2 - Math.max(1.5, hh), bw - 2, Math.max(1.5, hh));
    }
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
      this.cameraOn = on;
      UI.$('mine').classList.toggle('off', !on);
      this.vision.setPaused(!on);
      UI.toast(on ? 'Kamera an – sie sieht dich wieder' : 'Kamera aus – reines Sprachgespräch läuft weiter');
    });
    UI.$('call-state').onclick = () => {
      if (!this.thinkStart) return;
      this.vlm && this.vlm.cancel();
      this.tts.cancel();
      this.vlmBusy = false;
      this.thinkStart = 0;
      this.dispatch({ t: 'status', state: 'idle' });
      UI.toast('Turn abgebrochen');
    };
    UI.$('btn-subs').onclick = (e) => this.toggle(e.currentTarget, (on) => {
      this.s.subtitles = on; this.save();
      this.setSubsVisible(on);
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
    UI.$('btn-keys').onclick = (e) => this.toggle(e.currentTarget, (on) => {
      UI.$('keyrow').hidden = !on;
      if (on) UI.$('key-input').focus();
    });
    const sendKey = () => {
      const v = UI.$('key-input').value.trim();
      if (!v) return;
      UI.$('key-input').value = '';
      this.addMsg('you', 'Du (Tastatur)', v);
      this.scrollTranscript();
      this.commitAt = performance.now();
      this.vlmTurn('user', v);
    };
    UI.$('key-send').onclick = sendKey;
    UI.$('key-input').onkeydown = (e) => { if (e.key === 'Enter') sendKey(); };
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
    UI.$('transcript').querySelectorAll('.msg').forEach(m => m.remove());
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

    // On-Device-VLM: läuft i. d. R. schon seit dem Seitenstart; sonst jetzt starten
    this.vlm = this.vlm || new OnDeviceVLM({ preset: this.s.onDevicePreset, device: this.s.onDeviceDevice || 'auto' });
    const prog = (msg, p) => {
      UI.$('ring-state').textContent = 'FastVLM lädt · ' + msg;
      const bar = UI.$('ring-prog');
      if (bar) bar.style.width = Math.round((p || 0) * 100) + '%';
      this._vlmProg = msg;
      this._vlmP = p || 0;
      this.renderInfo();
    };
    prog('Tokenizer + Vorverarbeitung…', 0);

    const failLoad = (msg) => {
      this._vlmError = msg;
      try { this.vlm && this.vlm.killSwitch(); } catch {}
      UI.$('ring-state').textContent = 'FastVLM konnte nicht geladen werden';
      const bar = UI.$('ring-prog'); if (bar) bar.style.width = '0%';
      UI.toast('On-Device-VLM: ' + msg, 6000);
      this.setChip('chip-vlm', 'VLM Fehler', false, true);
      this.ring.stop();
      clearInterval(this._stall);
    };
    this._vlmError = null;
    let maxP = -1, lastMove = Date.now();
    clearInterval(this._stall);
    this._stall = setInterval(() => {
      if (!this.vlm || this.vlm.ready || this.state === 'idle') { clearInterval(this._stall); return; }
      const p = this._vlmP || 0;
      if (p > maxP) { maxP = p; lastMove = Date.now(); }
      else if (Date.now() - lastMove > 12000) {
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
    this.vlm.warmup && this.vlm.warmup();
    if (this.s.handsFree && this.asr.supported) this.asr.start();
    else {
      this.setChip('chip-asr', this.asr.supported ? 'ASR aus' : 'ASR nicht verfügbar', false, true);
      UI.$('keyrow').hidden = false;
      UI.$('btn-keys').dataset.on = 'true';
      if (!this.asr.supported) UI.toast('Keine System-Spracherkennung in diesem Browser – Tastatur-Eingang ist offen', 5000);
    }
    this.addSys(`verbunden · FastVLM-0.5B on-device (${this.vlm.device})`);
    this.setupWaveform();
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
    this._gibAcc = ''; this._gibDone = false; this._gibHit = false;
    this.thinkStart = performance.now();
    this._gotFirst = false;
    this.herBubble = null; this.herPending = '';
    this.dispatch({ t: 'status', state: 'thinking', reason });
    const dir = directives(this.s.lang);
    const userText = reason === 'proactive' ? dir.proactive : reason === 'greeting' ? dir.greet : text;
    let sentBuf = '', all = '';
    const t0 = performance.now();
    try {
      const full = await this.vlm.generate({
        system: characterSystem(this.persona, this.s.lang, { cameraOn: this.cameraOn, memo: this.memo.summary }),
        history: this.vlmHistory.slice(-6),
        userText,
        // Kamera aus => reines Sprachgespräch (text-only); sonst exakt der angezeigte Frame
        canvas: this.cameraOn ? this.vision.grab : null,
        maxTokens: this.s.turbo ? 48 : 96,
        onDelta: (tk) => {
          if (!this._gotFirst) {
            this._gotFirst = true;
            const ttft = performance.now() - this.thinkStart;
            this.perf.ttft = ttft;
            this.setChip('chip-vlm', `1. Token ${Math.round(ttft / 1000)}s · ${this.vlm.device}`, true);
          }
          all += tk;
          this._gibAcc += tk;
          if (!this._gibDone && this._gibAcc.length >= 24) {
            this._gibDone = true;
            if (looksGibberish(this._gibAcc)) {
              this._gibHit = true;
              this.vlm.cancel();
              this.tts.cancel();
              this.setChip('chip-vlm', 'Ausgabe = Buchstabensalat', false, true);
              UI.toast('On-Device-VLM hat Buchstabensalat erzeugt – Turn abgebrochen, nichts gesprochen. Preset „Standard“ oder „fp16“ wählen.', 7000);
              return;
            }
          }
          if (this._gibHit) return;
          this.dispatch({ t: 'delta', text: tk });
          sentBuf += tk;
          const r = splitSentences(sentBuf);
          r.out.forEach((snt) => { if (snt) this.dispatch({ t: 'sentence', text: snt, i: ++this._si }); });
          sentBuf = r.rest;
          if (this._si >= (this.s.turbo ? 2 : 3)) this.vlm.cancel();
        },
      });
      if (this._gibHit) { this.vlmBusy = false; this.thinkStart = 0; this.dispatch({ t: 'status', state: 'idle' }); return; }
      if (sentBuf.trim()) this.dispatch({ t: 'sentence', text: sentBuf.trim(), i: ++this._si });
      const clean = (full || all).trim();
      if (looksGibberish(clean)) {
        this.tts.cancel();
        this.setChip('chip-vlm', 'Ausgabe = Buchstabensalat', false, true);
        UI.toast('VLM-Ausgabe unbrauchbar – nicht gesprochen. Preset wechseln.', 6000);
        this.vlmBusy = false; this.thinkStart = 0;
        this.dispatch({ t: 'status', state: 'idle' });
        return;
      }
      if (reason === 'user' && text) this.vlmHistory.push(['user', text]);
      else if (reason !== 'user') this.vlmHistory.push(['user', '(Stille)']);
      if (clean) this.vlmHistory.push(['assistant', clean]);
      this.memo.rounds++;
      this.maybeCompress();
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
    this.thinkStart = 0;
    this.lastExchange = performance.now();
    this.dispatch({ t: 'status', state: 'idle' });
  }

  /** StreamBridge-Round-Decay on-device: alle 4 Runden backt das VLM nebenher
   *  aus altem Merkzettel + letzten Runden einen neuen Merkzettel. */
  maybeCompress() {
    if (this.memo.rounds < 4 || this.memoBusy || !this.vlm?.ready || this.vlmBusy) return;
    this.memo.rounds = 0;
    const rounds = this.vlmHistory.slice(-8).map(([r, t]) => (r === 'user' ? 'ER: ' : 'SIE: ') + t).join('\n');
    this.memoBusy = true;
    this.vlm.generate({
      system: 'Fasse diesen Gesprächsausschnitt in maximal zwei kurzen Sätzen zusammen. Nur Fakten, die man später noch braucht (Namen, Orte, Vorlieben, Absprachen). Keine Anrede.',
      history: [], userText: (this.memo.summary ? 'ALTER MERKZETTEL: ' + this.memo.summary + '\n' : '') + rounds,
      canvas: null, maxTokens: 48,
    }).then((t) => {
      const sum = (t || '').trim().replace(/\s+/g, ' ');
      if (sum && !looksGibberish(sum)) this.memo.summary = sum.slice(0, 220);
    }).catch(() => {}).finally(() => { this.memoBusy = false; this.renderInfo(); });
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
        if (!this._subCleared) { this._subCleared = true; this.subYou(''); }
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
    this.subYou('✓ gehört: ' + text);
    this._subCleared = false;
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

  /* ================= echte Pegel -> Waveform ================= */
  renderAudio() {
    cancelAnimationFrame(this._rafId);
    const loop = () => {
      this._rafId = requestAnimationFrame(loop);
      if (this.state !== 'in-call') return;
      const now = performance.now();
      this._wfTick = (this._wfTick || 0) + 1;
      const draw = this._wfTick % 2 === 0;

      const lvl = this.vad?.running ? this.vad.level : 0;
      this.micHist = (this.micHist || []).concat(lvl).slice(-(this._wfBars || 48));
      if (this.tts.speaking && !this._boundCount) this.herEnv = Math.max(this.herEnv || 0, 0.3);
      this.herEnv = (this.herEnv || 0) * 0.90;
      this.herHist = (this.herHist || []).concat(this.herEnv || 0).slice(-(this._wfBars || 48));
      if (draw) this.drawWave();

      // Barge-in-Kanal 2: echter Pegel-Spike ueber dem adaptiven Noisefloor
      if (this.tts.speaking && this.s.bargeIn && this.vad?.running) {
        const floor = this.vad.noiseFloor || 0.012;
        if (lvl > Math.max(0.05, floor * 6)) this._spikeMs = (this._spikeMs || 0) + 16; else this._spikeMs = 0;
        if (this._spikeMs > 180 && now - (this._lastBarge || 0) > 1500) { this._lastBarge = now; this._spikeMs = 0; this.onSpeechStart(true); }
      } else this._spikeMs = 0;

      if (this.s.handsFree && this.asr.supported && this.asr.healthTick()) {
        this.setChip('chip-asr', 'ASR neu gestartet', false, true);
      }

      // Think-Watchdog: sichtbare Sekunden + ehrlicher Abbruch statt "denkt unendlich"
      if (this.thinkStart && !this._gotFirst) {
        const el = (now - this.thinkStart) / 1000;
        const sec = Math.round(el);
        if (sec !== this._thinkSec) {
          this._thinkSec = sec;
          this.setChip('chip-vlm', `denkt … ${sec}s (${this.vlm?.device || '?'})`, true);
          UI.$('call-state').textContent = `denkt nach … ${sec}s (tippen = abbrechen)`;
        }
        const lim = (this.vlm?.device === 'wasm' ? 150 : 60) * (this.s.turbo ? 0.6 : 1);
        if (el > lim) {
          this.thinkStart = 0;
          this.vlm && this.vlm.cancel();
          this.vlmBusy = false;
          this.dispatch({ t: 'status', state: 'idle' });
          this.setChip('chip-vlm', 'Turn: kein 1. Token', false, true);
          UI.toast(`Kein erstes Token in ${Math.round(lim)}s (${this.vlm?.device}). Turbo an oder Preset fp16 versuchen.`, 7000);
        }
      } else if (this.thinkStart && this._gotFirst) {
        UI.$('call-state').textContent = 'spricht…';
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
    this.herEnv = 0.9;
    UI.$('her').classList.add('speaking');
    UI.$('her').classList.remove('listening');
    this.vad.setSheSpeaks(true);
    if (this.s.halfDuplex) this.asr.setPausedForTTS(true);
    UI.$('call-state').textContent = 'spricht…';
  }
  onTtsEnd() {
    this.herEnv = 0;
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
    const cv = UI.$('chip-vlm');
    if (cv) cv.title = `Kamera ${st.src} · ${st.facing} · ${st.mirror ? 'gespiegelt' : 'ungespiegelt'} · torch ${st.torch ? 'an' : 'aus'}`;
  }

  /* ================= Settings ================= */
  save() { localStorage.setItem(KEY, JSON.stringify(this.s)); }
  setSubsVisible(v) {
    const a = document.querySelector('.her-sub-wrap'), b = document.querySelector('.you-sub-wrap');
    if (a) a.style.display = v ? '' : 'none';
    if (b) b.style.display = v ? '' : 'none';
  }
  applyPersisted() {
    this.setSubsVisible(this.s.subtitles);
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
      { type: 'select', key: 'onDevicePreset', label: 'Modell-Mix', options: [
        { value: 'apple', label: 'Standard (von Apple validiert) ≈ 1,1 GB' },
        { value: 'fp16', label: 'fp16 ≈ 1,5 GB (max. Qualität)' },
        { value: 'compact', label: 'compact ≈ 676 MB (EXPERIMENTELL: kann Buchstabensalat erzeugen)' }] },
      { type: 'select', key: 'onDeviceDevice', label: 'Inferenz-Gerät', options: [
        { value: 'auto', label: 'auto (WebGPU, sonst WASM)' }, { value: 'webgpu', label: 'WebGPU erzwingen' }, { value: 'wasm', label: 'WASM/CPU erzwingen' }] },
      { type: 'switch', key: 'turbo', label: 'Turbo (minimale Latenz)', desc: '48 statt 96 Tokens, max. 2 Sätze. Gilt ab dem nächsten Anruf.' },
      { type: 'info', key: 'vlm', rows: [['Status', '–']] },
      { type: 'button', key: 'systest', label: 'Systemtest: gesamte Signal-Kette messen', buttonLabel: 'Test' },
      { type: 'info', key: 'test', rows: [['Systemtest', 'noch nicht gelaufen']] },
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
      case 'subtitles': this.setSubsVisible(v); break;
      case 'pixelExact': UI.$('modelview').classList.toggle('nosmooth', v); break;
      case 'mirrorMode': this.vision.setMirrorMode(v); break;
      case 'cameraFacing': this.vision.facing = v; this.vision.restart().then(() => { UI.$('btn-torch').hidden = !this.vision.hasTorch; }); break;
      case 'previewMode': this.vision.setMode(v); break;
      case 'frameSize': this.vision.size = v; break;
      case 'ratio': this.vision.ratio = v; break;
      case 'quality': this.vision.quality = v; break;
      case 'intervalMs': this.vision.intervalMs = v; break;
      case 'onDevicePreset': UI.toast('Gilt ab dem nächsten Anruf'); break;
      case 'systest': this.runSystemTest(); break;
    }
  }

  onVoices(list) { this._voices = list; this.refreshVoiceOptions(); }
  refreshVoiceOptions() {
    if (!this._voices) return;
    const sel = document.getElementById('set-voiceURI');
    if (!sel) return;
    const opts = [{ value: '', label: 'automatisch (weiblich)' }].concat(
      rankVoices(this._voices, this.s.lang).slice(0, 40).map(v => ({ value: v.voiceURI, label: `${v.name} · ${v.lang}${v.localService ? '' : ' · cloud'}` })));
    sel.innerHTML = '';
    opts.forEach((o) => { const op = UI.el('option', null, o.label); op.value = o.value; if (o.value === (this.s.voiceURI || '')) op.selected = true; sel.appendChild(op); });
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

  /** Systemtest: misst jede Stufe der echten Signal-Kette und zeigt sie als Liste. */
  async runSystemTest() {
    const R = [];
    const add = (name, ok, detail) => { R.push([name, ok, detail]); this.renderTest(R); };
    const wait = (ms) => new Promise(r => setTimeout(r, ms));
    this.renderTest([['Test läuft …', 'warn', '']]);

    add('Secure Context', window.isSecureContext ? 'pass' : 'fail', location.protocol + '//' + location.host);

    try {
      if (!this.vision.running) await this.vision.start();
      await wait(800);
      const g = this.vision.grab.getContext('2d', { willReadFrequently: true });
      const d = g.getImageData(0, 0, 48, 48).data;
      let mean = 0, varr = 0;
      const lum = [];
      for (let i = 0; i < d.length; i += 4) { const l = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2]; lum.push(l); mean += l; }
      mean /= lum.length;
      for (const l of lum) varr += (l - mean) * (l - mean);
      varr /= lum.length;
      add('Kamera → Frame', varr > 15 ? 'pass' : 'warn', `Varianz ${varr | 0} (schwarz = 0)`);
    } catch (e) { add('Kamera → Frame', 'fail', String(e.message || e).slice(0, 60)); }

    try {
      if (!this.vad.running) await this.vad.start();
      let mx = 0;
      for (let i = 0; i < 25; i++) { mx = Math.max(mx, this.vad.level); await wait(100); }
      add('Mikro → Pegel', mx > 0.01 ? 'pass' : mx > 0.002 ? 'warn' : 'fail', `max ${mx.toFixed(3)} (Stille = !)`);
      add('VAD (Silero)', this.vad.info.detector === 'silero' ? 'pass' : 'fail', this.vad.info.detector);
    } catch (e) { add('Mikro/VAD', 'fail', String(e.message || e).slice(0, 60)); }

    if (!this.asr.supported) {
      add('System-ASR (Transkript)', 'fail', 'Browser ohne Spracherkennung → Tastatur-Eingang');
    } else {
      this.asr.start();
      await wait(1800);
      add('System-ASR (Transkript)', this.asr.wantRunning ? 'pass' : 'warn', 'sprich kurz: Gehörtes muss im Protokoll erscheinen');
    }

    const ttsRes = await new Promise((res) => {
      const to = setTimeout(() => res('timeout'), 5000);
      try {
        const u = new SpeechSynthesisUtterance('Test.');
        u.onstart = () => { clearTimeout(to); res('start'); };
        u.onerror = (e) => { clearTimeout(to); res(e.error || 'error'); };
        speechSynthesis.speak(u);
      } catch (e) { clearTimeout(to); res('throw'); }
    });
    try { speechSynthesis.cancel(); } catch {}
    add('System-TTS (Stimme)', ttsRes === 'start' ? 'pass' : 'fail', String(ttsRes));

    if (this.vlm && this.vlm.ready) {
      const t0 = performance.now();
      let out = '';
      try {
        await this.vlm.generate({
          system: 'Antworte mit genau einem Wort.', history: [], userText: 'Sag ein Wort.',
          canvas: this.vision.grab, maxTokens: 6, onDelta: (t) => { out += t; },
        });
        add('VLM-Mini-Inferenz', out.trim() ? 'pass' : 'warn', `"${out.trim().slice(0, 24)}" · ${(performance.now() - t0) | 0} ms`);
      } catch (e) { add('VLM-Mini-Inferenz', 'fail', String(e.message || e).slice(0, 60)); }
    } else {
      add('On-Device-VLM', 'fail', this._vlmError || 'nicht geladen (Dialer-Chip ansehen)');
    }

    const fails = R.filter(r => r[1] === 'fail').length;
    UI.toast(fails ? `Systemtest: ${fails} Stufe(n) mit ✗` : 'Systemtest: alle Stufen ✓', 5000);
    return R;
  }

  renderTest(R) {
    const sym = { pass: '✓', warn: '!', fail: '✗' };
    UI.updateInfoBlock(UI.$('settings-body'), 'test', R.map(([n, ok, d]) => [`${sym[ok] || '?'} ${n}`, d || ok]));
  }
}

const app = new App();
window.__app = app;
app.boot();
