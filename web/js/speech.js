/**
 * "System speech": the browser's own speech engine on both directions.
 *
 *   SystemTTS  - window.speechSynthesis, sentence-queued (so streaming tokens
 *                can be spoken before the model finished), with the usual
 *                mobile workarounds (iOS gesture unlock, missing onend,
 *                Android voice list that arrives late).
 *   SystemASR  - window.(webkit)SpeechRecognition, continuous + interim,
 *                auto-restarting, half-duplex aware.
 */

/* ------------------------------------------------------------------ */
/*  voices                                                             */
/* ------------------------------------------------------------------ */
const FEMALE_HINTS = [
  'anna', 'helena', 'lena', 'vicki', 'vicky', 'petra', 'marlene', 'hedda', 'clara',
  'samantha', 'victoria', 'zira', 'susan', 'karen', 'tessa', 'fiona', 'moira', 'tilda',
  'amelie', 'julie', 'aude', 'google deutsch', 'female', 'frau', 'weiblich', 'yelda',
  'nora', 'julia', 'laura', 'sara', 'eva', 'mila', 'siri',
];
const MALE_HINTS = ['stefan', 'markus', 'david', 'mark', 'daniel', 'alex', 'male', 'mann', 'herr', 'fred', 'ralf'];

const norm = (s) => String(s || '').toLowerCase().replace(/_/g, '-');

export function pickVoice(voices, lang = 'de-DE', names = [], gender = 'f') {
  if (!voices || !voices.length) return null;
  const want = norm(lang);
  const base = want.split('-')[0];
  let pool = voices.filter(v => norm(v.lang) === want);
  if (!pool.length) pool = voices.filter(v => norm(v.lang).startsWith(base + '-'));
  if (!pool.length) pool = voices.filter(v => norm(v.lang).startsWith(base));
  if (!pool.length) pool = voices.slice();

  const hints = [...names.map(norm), ...(gender === 'f' ? FEMALE_HINTS : MALE_HINTS)];
  for (const h of hints) {
    const hit = pool.find(v => norm(v.name).includes(h));
    if (hit) return hit;
  }
  // avoid obviously male voices when we want a woman
  const anti = gender === 'f' ? MALE_HINTS : FEMALE_HINTS;
  const notAnti = pool.filter(v => !anti.some(h => norm(v.name).includes(h)));
  if (notAnti.length) return notAnti[0];
  return pool[0];
}

export function rankVoices(voices, lang = 'de-DE') {
  const base = norm(lang).split('-')[0];
  return [...voices].sort((a, b) => {
    const sa = norm(a.lang) === norm(lang) ? 0 : norm(a.lang).startsWith(base) ? 1 : 2;
    const sb = norm(b.lang) === norm(lang) ? 0 : norm(b.lang).startsWith(base) ? 1 : 2;
    if (sa !== sb) return sa - sb;
    const fa = FEMALE_HINTS.some(h => norm(a.name).includes(h)) ? 0 : 1;
    const fb = FEMALE_HINTS.some(h => norm(b.name).includes(h)) ? 0 : 1;
    if (fa !== fb) return fa - fb;
    return (a.localService === b.localService) ? a.name.localeCompare(b.name) : (a.localService ? -1 : 1);
  });
}

/* ------------------------------------------------------------------ */
/*  TTS                                                                */
/* ------------------------------------------------------------------ */
export class SystemTTS {
  constructor(cb = {}) {
    this.cb = cb;
    this.synth = window.speechSynthesis;
    this.supported = !!this.synth;
    this.queue = [];
    this.current = null;
    this.busy = false;
    this.voice = null;
    this.lang = 'de-DE';
    this.rate = 1.0;
    this.pitch = 1.0;
    this.volume = 1.0;
    this.unlocked = false;
    this.voices = [];
    this.cancelled = false;
    this._watchdog = null;
    this._sentCount = 0;
    if (this.supported) {
      this._refreshVoices();
      this.synth.addEventListener?.('voiceschanged', () => this._refreshVoices());
      // Safari fires voiceschanged only sometimes
      setTimeout(() => this._refreshVoices(), 400);
      setTimeout(() => this._refreshVoices(), 1600);
    }
  }

  _refreshVoices() {
    const list = this.synth.getVoices() || [];
    if (!list.length) return;
    this.voices = list;
    this.cb.onVoices && this.cb.onVoices(list);
    if (!this.voice) this.autoVoice();
  }

  autoVoice(names = [], lang = this.lang) {
    this.lang = lang || this.lang;
    this.voice = pickVoice(this.voices, this.lang, names, 'f');
    this.cb.onVoice && this.cb.onVoice(this.voice);
    return this.voice;
  }

  setVoiceURI(uri) {
    const v = this.voices.find(x => x.voiceURI === uri);
    if (v) { this.voice = v; this.lang = v.lang; this.cb.onVoice && this.cb.onVoice(v); }
    return this.voice;
  }

  /** must be called from a user gesture (tap on "call") - unlocks iOS */
  unlock() {
    if (!this.supported || this.unlocked) return;
    this.unlocked = true;
    try {
      const u = new SpeechSynthesisUtterance(' ');
      u.volume = 0.01; u.rate = 1; u.voice = this.voice || undefined;
      u.onend = () => {};
      this.synth.speak(u);
      setTimeout(() => { try { this.synth.cancel(); } catch {} }, 120);
    } catch {}
  }

  /** queue one sentence (used while the model streams) */
  enqueue(text) {
    const clean = String(text || '').trim();
    if (!clean || !this.supported) return;
    this.queue.push(clean);
    this._pump();
  }

  /** replace the queue and speak immediately (greetings, fallbacks) */
  speakNow(text) {
    this.queue = [];
    this.enqueue(text);
  }

  _pump() {
    if (!this.supported || this.busy || !this.queue.length) return;
    const text = this.queue.shift();
    this.busy = true;
    this.cancelled = false;
    this._sentCount++;
    const u = new SpeechSynthesisUtterance(text);
    if (this.voice) { u.voice = this.voice; u.lang = this.voice.lang; }
    else u.lang = this.lang;
    u.rate = Math.max(0.5, Math.min(2, this.rate));
    u.pitch = Math.max(0, Math.min(2, this.pitch));
    u.volume = this.volume;
    this.current = u;

    let done = false;
    const finish = (why) => {
      if (done) return;
      done = true;
      clearTimeout(this._watchdog);
      this.busy = false;
      this.current = null;
      this.cb.onSentenceEnd && this.cb.onSentenceEnd(text, why);
      if (this.cancelled || !this.queue.length) {
        if (this.cancelled) this.cb.onCancel && this.cb.onCancel();
        else this.cb.onEnd && this.cb.onEnd();
      }
      if (!this.cancelled) this._pump();
    };

    u.onstart = () => { this.cb.onStart && this.cb.onStart(text); };
    u.onboundary = (e) => { this.cb.onBoundary && this.cb.onBoundary(e, text); };
    u.onend = () => finish('end');
    u.onerror = (e) => { if (e && e.error !== 'interrupted' && e.error !== 'canceled') this.cb.onError && this.cb.onError(e.error); finish('error'); };

    // watchdog: some Android builds never fire onend
    const est = 500 + (text.length / Math.max(0.4, u.rate)) * 62;
    this._watchdog = setTimeout(() => finish('watchdog'), Math.min(20000, est + 2500));

    try { this.synth.speak(u); }
    catch (e) { finish('throw'); this.cb.onError && this.cb.onError(String(e)); }

    // Chrome desktop/Android sometimes freezes long queues
    if (!this._keepAlive) {
      this._keepAlive = setInterval(() => {
        if (this.synth.speaking && !this.synth.paused) { try { this.synth.resume(); } catch {} }
      }, 6000);
    }
  }

  cancel() {
    this.cancelled = true;
    this.queue = [];
    clearTimeout(this._watchdog);
    if (this.supported) { try { this.synth.cancel(); } catch {} }
    if (this.busy) { this.busy = false; this.current = null; this.cb.onCancel && this.cb.onCancel(); }
  }

  get speaking() { return this.busy || this.queue.length > 0; }
  get pending() { return this.queue.length; }
}

/* ------------------------------------------------------------------ */
/*  ASR                                                                */
/* ------------------------------------------------------------------ */
export class SystemASR {
  constructor(cb = {}) {
    this.cb = cb;
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    this.SR = SR;
    this.supported = !!SR;
    this.lang = 'de-DE';
    this.wantRunning = false;
    this.rec = null;
    this.interim = '';
    this.restartDelay = 220;
    this.restarts = 0;
    this.lastFinalAt = 0;
    this.lastActivityAt = performance.now();
    this.startedAt = 0;
    this.pausedForTTS = false;
  }

  setLang(lang) { this.lang = lang || this.lang; if (this.rec) { this.rec.lang = this.lang; } }

  start() {
    if (!this.supported) { this.cb.onError && this.cb.onError('unsupported'); return false; }
    this.wantRunning = true;
    this._create();
    try { this.rec.start(); this.cb.onState && this.cb.onState('starting'); }
    catch (e) { /* already started */ }
    return true;
  }

  _create() {
    const r = new this.SR();
    r.continuous = true;
    r.interimResults = true;
    r.maxAlternatives = 1;
    r.lang = this.lang;

    r.onstart = () => { this.startedAt = performance.now(); this.lastActivityAt = this.startedAt; this.cb.onState && this.cb.onState('listening'); };
    r.onaudiostart = () => { this.lastActivityAt = performance.now(); this.cb.onState && this.cb.onState('listening'); };
    r.onspeechstart = () => { this.lastActivityAt = performance.now(); this.cb.onSpeech && this.cb.onSpeech(true); };
    r.onspeechend = () => { this.cb.onSpeech && this.cb.onSpeech(false); };

    r.onresult = (e) => {
      let interim = '';
      for (let i = e.resultIndex; i < e.results.length; i++) {
        const res = e.results[i];
        const txt = res[0].transcript;
        if (res.isFinal) {
          const t = txt.trim();
          this.lastActivityAt = performance.now();
          if (t) {
            this.lastFinalAt = performance.now();
            this.cb.onFinal && this.cb.onFinal(t, res[0].confidence || 0);
          }
        } else interim += txt;
      }
      this.interim = interim.trim();
      this.cb.onInterim && this.cb.onInterim(this.interim);
    };

    r.onerror = (e) => {
      const err = e.error || 'unknown';
      this.cb.onError && this.cb.onError(err);
      if (err === 'not-allowed' || err === 'service-not-allowed') { this.wantRunning = false; this.cb.onFatal && this.cb.onFatal(err); }
      if (err === 'audio-capture') { this.wantRunning = false; this.cb.onFatal && this.cb.onFatal(err); }
    };

    r.onend = () => {
      this.cb.onState && this.cb.onState('ended');
      if (!this.wantRunning) return;
      this.restarts++;
      const delay = this.pausedForTTS ? 120 : Math.min(1800, this.restartDelay + (this.restarts > 6 ? 350 : 0));
      clearTimeout(this._rt);
      this._rt = setTimeout(() => {
        if (!this.wantRunning) return;
        try { this._create(); this.rec.start(); }
        catch (err) { this.cb.onError && this.cb.onError('restart-failed'); }
      }, delay);
    };
    this.rec = r;
  }

  /** half-duplex: stop listening while she talks, restart when she is done */
  setPausedForTTS(on) {
    this.pausedForTTS = on;
    if (!this.supported) return;
    if (on) { this.stop(true); }
    else if (this.wantRunning) { this.start(); }
  }

  stop(temporary = false) {
    if (!temporary) this.wantRunning = false;
    clearTimeout(this._rt);
    if (this.rec) { try { this.rec.stop(); } catch {} }
    if (!temporary) this.rec = null;
    this.cb.onState && this.cb.onState(temporary ? 'paused' : 'stopped');
  }

  /** Health-Watchdog: erkannt nichts mehr, obwohl er laufen soll? Hart neu bauen. */
  healthTick() {
    if (!this.supported || !this.wantRunning || this.pausedForTTS) return null;
    const idle = performance.now() - this.lastActivityAt;
    if (idle > 25000) {
      this.lastActivityAt = performance.now();
      try { this.rec && this.rec.abort && this.rec.abort(); } catch {}
      try { this._create(); this.rec.start(); } catch {}
      return 'recreated';
    }
    return null;
  }

  flushInterim() { const i = this.interim; this.interim = ''; return i; }
}
