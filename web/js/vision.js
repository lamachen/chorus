/**
 * The vision pipeline - and the guarantee that the lower half of the call UI
 * shows *exactly* the pixels that are handed to FastVLM.
 *
 *   camera ──▶ <video> ──▶ [cover-crop + optional mirror] ──▶ grab-canvas
 *                                                                  │
 *                                                    toBlob('image/jpeg')
 *                                                                  │
 *                            ┌─────────────────────────────────────┴────────────┐
 *                            ▼                                                  ▼
 *                   WebSocket (binary JPEG)                          createImageBitmap(blob)
 *                   → server → FastVLM                               → visible <canvas>
 *
 * The visible canvas is painted from the *decoded bytes of the JPEG that was
 * just sent*, i.e. including compression artefacts - not from the camera.
 * Its backing store has exactly the capture resolution, so 1 canvas pixel ==
 * 1 model pixel.  FastVLM's own preprocessor only resizes the shortest edge to
 * 1024 and centre-crops a square; since we already send a square, the framing
 * is identical to what the model sees.
 *
 * Preview mode "fluid" is an opt-in comfort mode: it draws the live camera with
 * the same crop, and marks every frame that actually goes to the model.
 */

export class Vision {
  constructor(o = {}) {
    this.video = o.video;
    this.grab = o.grab;
    this.view = o.view;
    this.onFrame = o.onFrame || (() => {});
    this.onStats = o.onStats || (() => {});
    this.onError = o.onError || (() => {});

    this.size = o.size || 768;
    this.ratio = o.ratio || '1:1';
    this.quality = o.quality || 0.8;
    this.intervalMs = o.intervalMs || 1100;
    this.mirrorMode = o.mirrorMode || 'auto';          // auto | on | off
    this.mode = o.mode || 'exact';
    this.facing = o.facing || 'environment';            // Default: RÜCKkamera
    this.mirror = this._mirrorFor(this.facing, o.mirror);
    this.torch = false;
    this.hasTorch = false;
    this.paused = false;
    this.running = false;

    this.frameId = 0;
    this.lastBytes = 0;
    this.sentTimes = [];
    this.gctx = this.grab.getContext('2d', { alpha: false, willReadFrequently: false });
    this.vctx = this.view.getContext('2d', { alpha: false });
    this._flash = 0;
  }

  _mirrorFor(facing, o) {
    if (o && o.mirrorMode && o.mirrorMode !== 'auto') return o.mirrorMode === 'on';
    if (this && this.mirrorMode && this.mirrorMode !== 'auto') return this.mirrorMode === 'on';
    return facing === 'user';      // Front = Selfie-Spiegel, Rück = ungespiegelt
  }

  async restart() {
    const wasRunning = this.running;
    this.stop();
    if (wasRunning) await this.start();
  }

  setMirrorMode(m) {
    this.mirrorMode = m;
    this.mirror = m === 'auto' ? this.facing === 'user' : m === 'on';
  }

  async setTorch(on) {
    const track = this.stream && this.stream.getVideoTracks()[0];
    if (!track || !this.hasTorch) return false;
    try {
      await track.applyConstraints({ advanced: [{ torch: !!on }] });
      this.torch = !!on;
      return true;
    } catch (e) { this.onError('torch: ' + e.message); return false; }
  }

  dims() {
    const s = this.size;
    if (this.ratio === '4:3') return { w: s, h: Math.round(s * 0.75) };
    if (this.ratio === '3:4') return { w: Math.round(s * 0.75), h: s };
    if (this.ratio === '16:9') return { w: s, h: Math.round(s * 9 / 16) };
    return { w: s, h: s };
  }

  async start() {
    if (this.running) return;
    const { w, h } = this.dims();
    this.grab.width = w; this.grab.height = h;
    this.view.width = w; this.view.height = h;

    const stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: { ideal: this.facing },
        width: { ideal: Math.max(w, h) },
        height: { ideal: Math.max(w, h) },
        frameRate: { ideal: 30, max: 30 },
      },
    });
    this.stream = stream;
    this._probeTrack();
    this.video.srcObject = stream;
    this.video.setAttribute('playsinline', '');
    try { await this.video.play(); } catch (e) { this.onError('video.play: ' + e.message); }

    this.running = true;
    this.fit();
    window.addEventListener('resize', () => this.fit());
    this._loop();
    if (this.mode === 'fluid') this._raf();
  }

  /** letterbox the canvas inside its container, keeping the pixel aspect */
  fit() {
    const box = this.view.parentElement;
    if (!box) return;
    const cw = box.clientWidth, ch = box.clientHeight;
    if (!cw || !ch) return;
    const s = Math.min(cw / this.view.width, ch / this.view.height);
    const w = Math.round(this.view.width * s), h = Math.round(this.view.height * s);
    Object.assign(this.view.style, {
      width: w + 'px', height: h + 'px',
      left: ((cw - w) / 2) + 'px', top: ((ch - h) / 2) + 'px',
      position: 'absolute',
    });
  }

  setMode(mode) {
    this.mode = mode;
    if (mode === 'fluid') this._raf();
    else { clearTimeout(this._rafId); this._drawLast(); }
  }

  setPaused(p) {
    this.paused = p;
    if (!p) this._tick();
  }

  _probeTrack() {
    const track = this.stream && this.stream.getVideoTracks()[0];
    this.track = track || null;
    try { this.hasTorch = !!(track && track.getCapabilities && track.getCapabilities().torch); }
    catch { this.hasTorch = false; }
    this.torch = false;
    const actual = track && track.getSettings ? track.getSettings().facingMode : null;
    if (actual) this.facing = actual;
    if (this.mirrorMode === 'auto') this.mirror = this.facing === 'user';
  }

  async flip() {
    this.facing = this.facing === 'user' ? 'environment' : 'user';
    if (this.mirrorMode === 'auto') this.mirror = this.facing === 'user';
    if (!this.stream) return this.facing;
    try {
      const s = await navigator.mediaDevices.getUserMedia({
        audio: false, video: { facingMode: { ideal: this.facing }, width: { ideal: Math.max(this.view.width, this.view.height) } },
      });
      this.stream.getTracks().forEach(t => t.stop());
      this.stream = s;
      this.video.srcObject = s;
      await this.video.play();
      this._probeTrack();
    } catch (e) { this.onError('flip: ' + e.message); }
    return this.facing;
  }

  stop() {
    this.running = false;
    clearTimeout(this._tid);
    cancelAnimationFrame(this._rafId);
    try { this.stream && this.stream.getTracks().forEach(t => t.stop()); } catch {}
    this.stream = null;
    if (this.video) this.video.srcObject = null;
  }

  /* ---------------- capture ---------------- */
  _paintGrab() {
    const v = this.video;
    if (!v || v.readyState < 2 || !v.videoWidth) return false;
    const { w, h } = this.dims();
    if (this.grab.width !== w) { this.grab.width = w; this.grab.height = h; this.view.width = w; this.view.height = h; this.fit(); }
    const g = this.gctx;
    const vw = v.videoWidth, vh = v.videoHeight;
    const scale = Math.max(w / vw, h / vh);          // "cover"
    const sw = w / scale, sh = h / scale;
    const sx = (vw - sw) / 2, sy = (vh - sh) / 2;
    g.save();
    if (this.mirror) { g.translate(w, 0); g.scale(-1, 1); }
    g.fillStyle = '#000';
    g.fillRect(0, 0, w, h);
    g.drawImage(v, sx, sy, sw, sh, 0, 0, w, h);
    g.restore();
    return true;
  }

  _loop() {
    if (!this.running) return;
    clearTimeout(this._tid);
    this._tid = setTimeout(() => this._tick(), this.intervalMs);
  }

  async _tick() {
    if (!this.running) return;
    this._loop();
    if (this.paused || !this._paintGrab()) return;

    const id = ++this.frameId;
    const now = performance.now();
    this.sentTimes.push(now);
    while (this.sentTimes.length && now - this.sentTimes[0] > 5000) this.sentTimes.shift();

    // Anzeige = exakt dieselben Pixel, die das VLM bekommt (derselbe Canvas)
    if (this.mode === 'exact') {
      this.vctx.drawImage(this.grab, 0, 0);
      this._flash = 3;
    }
    this.onFrame(id, null, {
      w: this.grab.width, h: this.grab.height,
      fps: this.sentTimes.length / 5, mode: this.mode, ts: Date.now(),
    });
    this._stats();
  }

  _raf() {
    if (!this.running || this.mode !== 'fluid') return;
    this._rafId = requestAnimationFrame(() => {
      if (this._paintGrab()) {
        this.vctx.drawImage(this.grab, 0, 0);
        if (this._flash > 0) {
          this._flash--;
          this.vctx.strokeStyle = 'rgba(255,80,70,.85)';
          this.vctx.lineWidth = Math.max(3, this.view.width / 140);
          this.vctx.strokeRect(2, 2, this.view.width - 4, this.view.height - 4);
        }
      }
      this._raf();
    });
  }

  _stats() {
    this.onStats({
      id: this.frameId, w: this.grab.width, h: this.grab.height,
      fps: this.sentTimes.length / 5,
      src: this.video ? `${this.video.videoWidth}×${this.video.videoHeight}` : '-',
      paused: this.paused, mode: this.mode, mirror: this.mirror, facing: this.facing,
      torch: this.torch, hasTorch: this.hasTorch,
    });
  }
}
