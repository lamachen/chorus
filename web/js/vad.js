/**
 * Voice Activity Detection: Silero VAD (ONNX) über onnxruntime-web/wasm.
 *
 * Eine einzige, echte Implementierung - kein Ersatz-Modell, keine Skripte:
 *   AudioWorklet (Resample auf 16 kHz, 512-Frame=32 ms)
 *     -> Silero-ONNX  -> Sprachwahrscheinlichkeit
 *     -> Hysterese-Automat (Onset-Streak, Hang-over, Min/Max-Dauer)
 *
 * Kann das ONNX-Modell nicht geladen werden, gibt es bewusst KEIN verstecktes
 * Ersatzverfahren: `start()` wirft und die App zeigt das offen an.
 * Die System-Spracherkennung hat eigenständig genug Endpunktung, um
 * weiterhin gehört zu werden; Barge-in über den Mikro-Pegel bleibt aktiv.
 */

const ORT_BASE = new URL('../vendor/ort/', import.meta.url).href;
const MODEL_URL = new URL('../vendor/silero_vad.onnx', import.meta.url).href;

const FRAME = 512;          // 32 ms @16 kHz
const FRAME_MS = 32;

/* ---------------- Hysterese-Automat ---------------- */
class SpeechGate {
  constructor(o = {}) {
    this.onset = o.onset ?? 0.55;
    this.offset = o.offset ?? 0.35;
    this.onsetFrames = o.onsetFrames ?? 3;
    this.bargeOnsetFrames = o.bargeOnsetFrames ?? 5;
    this.hangoverMs = o.hangoverMs ?? 620;
    this.minSpeechMs = o.minSpeechMs ?? 240;
    this.maxSpeechMs = o.maxSpeechMs ?? 22000;
    this.boost = 1.0;
    this.reset();
  }
  reset() { this.speaking = false; this.streak = 0; this.t0 = 0; this.tLastVoice = 0; }
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
      if (voiced > this.hangoverMs || now - this.t0 > this.maxSpeechMs) {
        this.speaking = false;
        const dur = this.tLastVoice - this.t0;
        this.streak = 0;
        ev = { type: 'end', at: this.tLastVoice, ms: dur, tooShort: dur < this.minSpeechMs };
      }
    }
    return ev;
  }
}

/* ---------------- Silero (ONNX, sequenziell befüllt) ---------------- */
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
    this._run(frame);
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
      if (this.onError) this.onError(String(e && e.message || e));
    } finally {
      this.busy = false;
      const next = this.queue.shift();
      if (next) this.prob(next);
    }
  }
  reset() { this.state = new Float32Array(2 * 128); this.smooth = 0; this.last = 0; }
}

/* ---------------- öffentliche Klasse ---------------- */
export class MicVad {
  constructor(o = {}) {
    this.cb = o;
    this.opts = { onset: 0.55, offset: 0.35, hangoverMs: 620, minSpeechMs: 240, bargeBoost: 1.25, bargeOnsetFrames: 5, ...o.opts };
    this.gate = new SpeechGate(this.opts);
    this.running = false;
    this.muted = false;
    this.level = 0;
    this.prob = 0;
    this.noiseFloor = 0.012;
    this.info = { detector: 'none', sampleRate: 0 };
  }

  async start() {
    if (this.running) return this.info;
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1, sampleRate: { ideal: 16000 } },
      video: false,
    });
    this.stream = stream;

    let ctx;
    try { ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 }); }
    catch { ctx = new (window.AudioContext || window.webkitAudioContext)(); }
    this.ctx = ctx;
    if (ctx.state === 'suspended') await ctx.resume();

    const detector = await this._loadSilero();      // wirft, wenn nicht ladbar -> ehrlich
    this.detector = detector;
    detector.onProb = (p) => this._probability(p);
    detector.onError = (m) => this.cb.onError && this.cb.onError(m);

    await ctx.audioWorklet.addModule(new URL('./vad-worklet.js', import.meta.url).href);
    this.node = new AudioWorkletNode(ctx, 'vad-processor', {
      numberOfInputs: 1, numberOfOutputs: 0, channelCount: 1,
      processorOptions: { targetRate: 16000, frameSize: FRAME },
    });
    this.node.port.onmessage = (e) => this._onMessage(e.data);
    ctx.createMediaStreamSource(stream).connect(this.node);

    this.running = true;
    this.info = { detector: 'silero', sampleRate: ctx.sampleRate, worklet: true };
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
      executionProviders: ['wasm'], graphOptimizationLevel: 'all', logSeverityLevel: 3,
    });
    return new SileroVad(ort, session);
  }

  _onMessage(msg) {
    if (!this.running) return;
    if (msg.type === 'level') { this.level = msg.rms; return; }
    if (msg.type !== 'frame') return;
    if (this.muted) { this._probability(0); return; }
    this.detector.prob(msg.data);
  }

  _probability(p) {
    this.prob = p;
    const now = performance.now();
    if (!this.gate.speaking) this.noiseFloor = 0.985 * this.noiseFloor + 0.015 * Math.max(this.level, 0.002);
    this.cb.onLevel && this.cb.onLevel(this.level, p, now);
    const ev = this.gate.feed(p, now);
    if (!ev) return;
    if (ev.type === 'start') this.cb.onSpeechStart && this.cb.onSpeechStart();
    else this.cb.onSpeechEnd && this.cb.onSpeechEnd(ev);
  }

  setSheSpeaks(on) { this.gate.setBoost(on ? this.opts.bargeBoost : 1.0); }
  tune(o = {}) {
    Object.assign(this.opts, o);
    if (o.onset) this.gate.onset = o.onset;
    if (o.offset) this.gate.offset = o.offset;
    if (o.hangoverMs) this.gate.hangoverMs = o.hangoverMs;
    if (o.minSpeechMs) this.gate.minSpeechMs = o.minSpeechMs;
    if (o.bargeBoost) this.gate.bargeBoost = this.opts.bargeBoost = o.bargeBoost;
    if (o.bargeOnsetFrames) this.gate.bargeOnsetFrames = o.bargeOnsetFrames;
  }
  setMuted(m) { this.muted = m; this.gate.reset(); }
  stop() {
    this.running = false;
    try { this.node && this.node.disconnect(); } catch {}
    try { this.stream && this.stream.getTracks().forEach(t => t.stop()); } catch {}
    try { this.ctx && this.ctx.close(); } catch {}
    this.node = this.stream = this.ctx = null;
    this.gate.reset();
  }
}

export { SpeechGate, FRAME, FRAME_MS };
