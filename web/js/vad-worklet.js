/**
 * AudioWorklet: resamples the microphone to 16 kHz and emits fixed
 * 512-sample frames (32 ms) - exactly what Silero VAD expects.
 *
 * Also works when the AudioContext already runs at 16 kHz (ratio === 1).
 */
class VadProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opts = (options && options.processorOptions) || {};
    this.target = opts.targetRate || 16000;
    this.frameSize = opts.frameSize || 512;      // silero v5/v6 @16k
    this.ratio = sampleRate / this.target;

    this.carry = new Float32Array(0);
    this.pos = 0;                                 // fractional read position
    this.frame = new Float32Array(this.frameSize);
    this.filled = 0;

    this.peak = 0;
    this.sum = 0;
    this.n = 0;
    this.blocks = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input.length || !input[0] || input[0].length === 0) return true;
    const ch = input[0];

    // ---- level meter (source rate) ----
    for (let i = 0; i < ch.length; i++) {
      const v = ch[i];
      const a = v < 0 ? -v : v;
      if (a > this.peak) this.peak = a;
      this.sum += v * v;
      this.n++;
    }

    // ---- resample (linear interpolation, carry across blocks) ----
    let buf = ch;
    if (this.carry.length) {
      buf = new Float32Array(this.carry.length + ch.length);
      buf.set(this.carry, 0);
      buf.set(ch, this.carry.length);
    }
    let p = this.pos;
    while (p + 1 < buf.length) {
      const i = p | 0;
      const f = p - i;
      this.push(buf[i] * (1 - f) + buf[i + 1] * f);
      p += this.ratio;
    }
    const k = p | 0;
    this.carry = k > 0 ? buf.slice(k) : buf.slice();
    this.pos = p - k;

    if (++this.blocks >= 8) {                     // ~ every 21 ms @128/48k
      this.blocks = 0;
      const rms = this.n ? Math.sqrt(this.sum / this.n) : 0;
      this.port.postMessage({ type: 'level', rms, peak: this.peak });
      this.peak = 0; this.sum = 0; this.n = 0;
    }
    return true;
  }

  push(sample) {
    this.frame[this.filled++] = sample;
    if (this.filled < this.frameSize) return;
    this.filled = 0;
    const out = new Float32Array(this.frame);
    this.port.postMessage({ type: 'frame', data: out }, [out.buffer]);
  }
}

registerProcessor('vad-processor', VadProcessor);
