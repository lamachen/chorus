/**
 * Hands-free voice activity detection.
 *
 * Two interchangeable detectors behind one state machine:
 *
 *   1. Silero VAD  (neural, vendored ONNX + onnxruntime-web/wasm)  <- default
 *   2. DSP VAD     (adaptive noise floor + SNR + zero-crossing)    <- fallback
 *
 * Both emit a 0..1 speech probability per 32 ms frame, which is fed through
 * the same hysteresis state machine (onset streak, hang-over, min/max speech
 * length).  While she is talking the thresholds are raised so the phone's own
 * loudspeaker does not trigger barge-in (browser AEC helps, but is not perfect).
 */

const ORT_BASE = new URL('../vendor/ort/', import.meta.url).href;
const MODEL_URL = new URL('../vendor/silero_vad.onnx', import.meta.url).href;

const FRAME = 512;          // samples @16 kHz = 32 ms
const FRAME_MS = 32;

/* ------------------------------------------------------------------ */
/*  state machine                                                      */
/* ------------------------------------------------------------------ */
class SpeechGate {
  constructor(opts = {}) {
    this.onset = opts.onset ?? 0.55;         // probability to start
    this.offset = opts.offset ?? 0.35;       // probability to stay in speech
    this.onsetFrames = opts.onsetFrames ?? 3;    // 96 ms
    this.bargeOnsetFrames = opts.bargeOnsetFrames ?? 7;   // 224 ms while she talks
    this.hangoverMs = opts.hangoverMs ?? 620;
    this.minSpeechMs = opts.minSpeechMs ?? 240;
    this.maxSpeechMs = opts.maxSpeechMs ?? 22000;
    this.boost = 1.0;                        // >1 while she is speaking
    this.reset();
  }
  reset() {
    this.speaking = false;
    this.streak = 0;
    this.t0 = 0;
    this.tLastVoice = 0;
  }
  setBoost(v) { if ((v > 1) !== (this.boost > 1)) this.streak = 0; this.boost = v; }

  feed(p, now) {
    const on = this.onset * this.boost;
    const off = this.offset * this.boost;
    let ev = null;
    if (!this.speaking) {
      this.streak = p >= on ? this.streak + 1 : 0;
      const need = this.boost > 1 ? this.bargeOnsetFrames : this.onsetFrames;
      if (this.streak >= need) {
        this.speaking = true;
        this.t0 = now - this.streak * FRAME_MS;
        this.tLastVoice = now;
        ev = { type: 'start', at: this.t0 };
      }
    } else {
      if (p >= off) this.tLastVoice = now;
      const voiced = now - this.tLastVoice;
      const total = now - this.t0;
      if (voiced > this.hangoverMs || total > this.maxSpeechMs) {
        this.speaking = false;
        const dur = this.tLastVoice - this.t0;
        this.streak = 0;
        ev = { type: 'end', at: this.tLastVoice, ms: dur, tooShort: dur < this.minSpeechMs };
      }
    }
    return ev;
  }
}

/* ------------------------------------------------------------------ */
/*  DSP fallback                                                       */
/* ------------------------------------------------------------------ */
class DspVad {
  constructor() {
    this.name = 'dsp';
    this.floor = -60;
    this.smooth = 0;
    this.initialised = false;
  }
  prob(frame) {
    let sum = 0, zc = 0, prev = 0;
    for (let i = 0; i < frame.length; i++) {
      const v = frame[i];
      sum += v * v;
      if ((v >= 0) !== (prev >= 0)) zc++;
      prev = v;
    }
    const rms = Math.sqrt(sum / frame.length);
    const db = 20 * Math.log10(rms + 1e-7);
    const zcr = zc / frame.length;            // 0..1, voice ~0.02-0.15, hiss >0.3

    if (!this.initialised) { this.floor = db; this.initialised = true; }
    else if (db < this.floor) this.floor = 0.82 * this.floor + 0.18 * db;
    else this.floor = Math.min(db - 3, this.floor + 0.0035);

    const snr = db - this.floor;
    let p = (snr - 4) / 13;
    p = Math.max(0, Math.min(1, p));
    if (zcr > 0.34) p *= 0.35;                // broadband noise, not voice
    else if (zcr < 0.02 && snr < 9) p *= 0.5; // very low-frequency rumble
    this.smooth = 0.55 * p + 0.45 * this.smooth;
    return this.smooth;
  }
}

/* ------------------------------------------------------------------ */
/*  Silero (ONNX)                                                      */
/* ------------------------------------------------------------------ */
class SileroVad {
  constructor(ort, session) {
    this.name = 'silero';
    this.ort = ort;
    this.session = session;
    this.state = new Float32Array(2 * 128);
    this.sr = new ort.Tensor('int64', new BigInt64Array([16000n]), []);
    this.busy = false;
    this.queue = [];
    this.last = 0;
    this.smooth = 0;
  }
  prob(frame) {
    if (this.busy) { this.queue.push(frame); if (this.queue.length > 4) this.queue.shift(); return this.last; }
    this.busy = true;
    this._run(frame).catch(() => {});
    return this.last;
  }
  async _run(frame) {
    const ort = this.ort;
    try {
      const out = await this.session.run({
        input: new ort.Tensor('float32', frame, [1, FRAME]),
        state: new ort.Tensor('float32', this.state, [2, 1, 128]),
        sr: this.sr,
      });
      this.state = Float32Array.from(out.stateN.data);
      const p = out.output.data[0];
      this.smooth = 0.6 * p + 0.4 * this.smooth;
      this.last = this.smooth;
      if (this.onProb) this.onProb(this.last);
    } catch (e) {
      this.error = String(e && e.message || e);
    } finally {
      this.busy = false;
      const next = this.queue.shift();
      if (next) this.prob(next);
    }
  }
  reset() { this.state = new Float32Array(2 * 128); this.smooth = 0; this.last = 0; }
}

/* ------------------------------------------------------------------ */
/*  public class                                                       */
/* ------------------------------------------------------------------ */
export class MicVad {
  /**
   * @param {object} o
   *   onSpeechStart(), onSpeechEnd(ms), onLevel(rms, prob), onInfo(obj), onError(e)
   */
  constructor(o = {}) {
    this.cb = o;
    this.opts = {
      neural: true,
      onset: 0.55, offset: 0.35, hangoverMs: 620, minSpeechMs: 240,
      bargeBoost: 1.55, bargeOnsetFrames: 7, ...o.opts,
    };
    this.gate = new SpeechGate(this.opts);
    this.running = false;
    this.muted = false;
    this.level = 0;
    this.prob = 0;
    this.info = { detector: 'none', sampleRate: 0 };
  }

  async start() {
    if (this.running) return this.info;
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true, noiseSuppression: true, autoGainControl: true,
        channelCount: 1, sampleRate: { ideal: 16000 },
      },
      video: false,
    });
    this.stream = stream;

    let ctx;
    try { ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 }); }
    catch { ctx = new (window.AudioContext || window.webkitAudioContext)(); }
    this.ctx = ctx;
    if (ctx.state === 'suspended') await ctx.resume();

    const src = ctx.createMediaStreamSource(stream);
    let detector = null;

    if (this.opts.neural) {
      try { detector = await this._loadSilero(); }
      catch (e) { this.cb.onInfo && this.cb.onInfo({ detector: 'dsp', note: 'Silero nicht ladbar: ' + (e.message || e) }); }
    }
    if (!detector) detector = new DspVad();
    this.detector = detector;

    // ---- audio graph ----
    const useWorklet = typeof ctx.audioWorklet !== 'undefined';
    if (useWorklet) {
      try {
        await ctx.audioWorklet.addModule(new URL('./vad-worklet.js', import.meta.url).href);
        this.node = new AudioWorkletNode(ctx, 'vad-processor', {
          numberOfInputs: 1, numberOfOutputs: 0, channelCount: 1,
          processorOptions: { targetRate: 16000, frameSize: FRAME },
        });
        this.node.port.onmessage = (e) => this._onMessage(e.data);
        src.connect(this.node);
      } catch (e) {
        this.node = null;
      }
    }
    if (!this.node) {
      // ScriptProcessor fallback (older Safari)
      this.sp = ctx.createScriptProcessor(2048, 1, 1);
      this._spBuf = new Float32Array(0);
      this.sp.onaudioprocess = (e) => this._onScript(e.inputBuffer.getChannelData(0));
      src.connect(this.sp);
      this.sp.connect(ctx.destination);   // required in some browsers; muted gain below
      const g = ctx.createGain(); g.gain.value = 0; this.sp.disconnect(); this.sp.connect(g); g.connect(ctx.destination);
    }

    this.running = true;
    this.info = { detector: detector.name, sampleRate: ctx.sampleRate, worklet: !!this.node };
    this.cb.onInfo && this.cb.onInfo(this.info);
    return this.info;
  }

  async _loadSilero() {
    const ort = await import(/* @vite-ignore */ ORT_BASE + 'ort.wasm.min.mjs');
    ort.env.wasm.wasmPaths = ORT_BASE;
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.proxy = false;
    ort.env.wasm.simd = true;
    const session = await ort.InferenceSession.create(MODEL_URL, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
      logSeverityLevel: 3,
    });
    const v = new SileroVad(ort, session);
    v.onProb = (p) => this._probability(p);
    return v;
  }

  _onMessage(msg) {
    if (!this.running) return;
    if (msg.type === 'level') { this.level = msg.rms; return; }
    if (msg.type !== 'frame') return;
    if (this.muted) { this._probability(0); return; }
    if (this.detector.name === 'silero') {
      this.detector.prob(msg.data);            // async: probability arrives via onProb
      this._fallbackTick(msg.data);            // keep a DSP estimate for the meter
    } else {
      this._probability(this.detector.prob(msg.data));
    }
  }

  _fallbackTick(frame) {
    if (!this._dsp) this._dsp = new DspVad();
    this._dspShadow = this._dsp.prob(frame);
    // if the neural net is lagging behind (queue full), fall back to the estimate
    if (this.detector.busy && this.detector.queue.length >= 3) this._probability(this._dspShadow);
  }

  _onScript(ch) {
    if (!this.running) return;
    const prev = this._spBuf;
    const buf = new Float32Array(prev.length + ch.length);
    buf.set(prev); buf.set(ch, prev.length);
    let sum = 0;
    for (let i = 0; i < ch.length; i++) sum += ch[i] * ch[i];
    this.level = Math.sqrt(sum / ch.length);
    let off = 0;
    while (buf.length - off >= FRAME) {
      const frame = buf.slice(off, off + FRAME);
      off += FRAME;
      if (this.muted) { this._probability(0); continue; }
      if (this.detector.name === 'silero') { this.detector.prob(frame); this._fallbackTick(frame); }
      else this._probability(this.detector.prob(frame));
    }
    this._spBuf = buf.slice(off);
  }

  _probability(p) {
    this.prob = p;
    const now = performance.now();
    this.cb.onLevel && this.cb.onLevel(this.level, p, now);
    const ev = this.gate.feed(p, now);
    if (!ev) return;
    if (ev.type === 'start') this.cb.onSpeechStart && this.cb.onSpeechStart(ev);
    else this.cb.onSpeechEnd && this.cb.onSpeechEnd(ev);
  }

  /** raise thresholds while she is talking (barge-in sensitivity) */
  setSheSpeaks(on) {
    this.gate.setBoost(on ? this.opts.bargeBoost : 1.0);
    this.cb.onInfo && this.cb.onInfo({ ...this.info, barge: on });
  }

  tune(o = {}) {
    Object.assign(this.opts, o);
    if (o.onset) this.gate.onset = o.onset;
    if (o.offset) this.gate.offset = o.offset;
    if (o.hangoverMs) this.gate.hangoverMs = o.hangoverMs;
    if (o.minSpeechMs) this.gate.minSpeechMs = o.minSpeechMs;
  }

  setMuted(m) { this.muted = m; this.gate.reset(); }

  stop() {
    this.running = false;
    try { this.node && this.node.disconnect(); } catch {}
    try { this.sp && this.sp.disconnect(); } catch {}
    try { this.stream && this.stream.getTracks().forEach(t => t.stop()); } catch {}
    try { this.ctx && this.ctx.close(); } catch {}
    this.node = this.sp = this.stream = this.ctx = null;
    this.gate.reset();
  }
}

export { SpeechGate, DspVad, SileroVad, FRAME, FRAME_MS };
