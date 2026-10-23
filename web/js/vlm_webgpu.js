/**
 * Echtes On-Device-VLM im Browser: FastVLM-0.5B als ONNX über
 * transformers.js (>=3.7, dieselbe Bibliothek wie Apples eigenes
 * `apple/fastvlm-webgpu`-Demo) mit WebGPU (Fallback: WASM).
 *
 * Modell: onnx-community/FastVLM-0.5B-ONNX
 *   vision_encoder + embed_tokens + decoder_model_merged (KV-Cache, merged)
 *
 * Download-Größen (einmalig, danach im Cache-API des Browsers):
 *   compact  ≈ 0,7 GB   (uint8 / uint8 / q4)      <- Default für Handys
 *   apple    ≈ 1,1 GB   (fp16 / q4  / q4)        <- Apples Demo-Mix
 *   balanced ≈ 0,8 GB   (fp16 / uint8 / q4)
 *   fp16     ≈ 1,5 GB
 *
 * Die Bildquelle ist derselbe Grab-Canvas wie im Server-Modus: Das On-Device-
 * VLM sieht also exakt die Frames, die unten im UI angezeigt werden.
 */

const CDN = [
  'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.5',
  'https://unpkg.com/@huggingface/transformers@3.7.5',
];

export const MODEL_ID = 'onnx-community/FastVLM-0.5B-ONNX';

/* NUR von Apple validierte Mischungen als Standard: uint8-Vision/Embeddings
 * erzeugen nachweislich Buchstabensalat - deshalb nicht mehr Default. */
/* Nur öffentlich validierte Mixe:
 *   q4f16  = Talk2FastVLM-Default (deren Speed/Qualitäts-Empfehlung, ~807 MB)
 *   apple  = Mix aus Apples WebGPU-Demo & ONNX-Card (~1095 MB)
 *   fp16   = Qualitätsempfehlung (~1516 MB)
 * uint8/int8: ersatzlos gestrichen (nur als Notlagen-Stufe dokumentiert, Salat-Risiko). */
export const DTYPE_PRESETS = {
  q4f16: { embed_tokens: 'q4f16', vision_encoder: 'q4f16', decoder_model_merged: 'q4f16' },
  apple: { embed_tokens: 'fp16', vision_encoder: 'q4', decoder_model_merged: 'q4' },
  fp16: { embed_tokens: 'fp16', vision_encoder: 'fp16', decoder_model_merged: 'fp16' },
};

export const PRESET_MB = { q4f16: 807, apple: 1095, fp16: 1516 };

/** Buchstabensalat-Erkennung: verhindert, dass Müll in die System-TTS geht. */
export function looksGibberish(text) {
  const t = String(text || '').replace(/[^\p{L}\s]/gu, '').toLowerCase().trim();
  if (t.length < 12) return false;
  const vowels = (t.match(/[aeiouäöüáéíóúàèìòù]/g) || []).length;
  const spaces = (t.match(/\s/g) || []).length;
  const letters = (t.match(/\p{L}/gu) || []).length;
  if (!letters) return false;
  if (vowels / letters < 0.09) return true;          // keine Vokale -> Salat
  if (spaces / letters < 0.04) return true;          // keine Wortgrenzen
  if (/(.)\1{6,}/.test(t)) return true;             // Zeichenwiederholung
  if (/(..\1{0,1})\1{3,}/.test(t) && spaces < 3) return true;
  return false;
}

let _T = null;
async function loadTransformers() {
  if (_T) return _T;
  let lastErr;
  for (const url of CDN) {
    try {
      _T = await import(/* @vite-ignore */ url);
      // keine endlosen Retry-Stürme bei Netzabbruch/Block: schnell ehrlich scheitern
      try { _T.env.maxRetry = 1; _T.env.retryDelay = 300; } catch {}
      return _T;
    } catch (e) { lastErr = e; }
  }
  throw new Error('transformers.js nicht ladbar: ' + (lastErr && lastErr.message));
}

export function onDeviceSupported() {
  return {
    webgpu: !!navigator.gpu,
    wasm: typeof WebAssembly === 'object',
    memoryGB: navigator.deviceMemory || 0,
    cores: navigator.hardwareConcurrency || 0,
    storage: !!navigator.storage,
  };
}

export class OnDeviceVLM {
  constructor({ preset = 'compact', device = 'auto', model = MODEL_ID } = {}) {
    this.preset = preset;
    this.devicePref = device;      // auto | webgpu | wasm
    this.model = model;
    this.ready = false;
    this.loading = false;
    this.busy = false;
    this.cancelled = false;
    this.device = null;
    this.error = null;
    this.loadMs = 0;
  }

  async deviceToUse() {
    if (this.devicePref !== 'auto') return this.devicePref;
    if (navigator.gpu) {
      try {
        const ad = await navigator.gpu.requestAdapter();
        if (ad) return 'webgpu';
      } catch {}
    }
    return 'wasm';
  }

  /** Erreichbarkeit des Modell-CDNs VOR dem Load prüfen - sonst ehrlich sofort scheitern
   *  statt in transformers.js' endloser Retry-Schleife zu hängen. */
  async _preflight() {
    if (this._preflightOk) return true;
    const url = `https://huggingface.co/${this.model}/resolve/main/config.json`;
    const ctl = new AbortController();
    const to = setTimeout(() => ctl.abort(), 8000);
    try {
      const r = await fetch(url, { signal: ctl.signal, cache: 'force-cache' });
      if (!r.ok) throw new Error('HTTP ' + r.status);
      this._preflightOk = true;
      return true;
    } catch (e) {
      throw new Error('Modell-CDN nicht erreichbar (offline oder blockiert) – On-Device-VLM nicht geladen');
    } finally {
      clearTimeout(to);
    }
  }

  async load(onProgress) {
    if (this.ready) return true;
    if (this.loading) return this._loadPromise;
    this.loading = true;
    this._loadPromise = (async () => {
      const t0 = performance.now();
      await this._preflight();
      const T = await loadTransformers();
      onProgress && onProgress('Processor (Tokenizer + Bild-Vorverarbeitung)…', 0);
      this.processor = await T.AutoProcessor.from_pretrained(this.model, {
        progress_callback: (m) => this._prog(onProgress, m),
      });
      this.device = await this.deviceToUse();
      onProgress && onProgress(`Modell (${this.preset}, ${this.device})…`, 0);
      this.model_ = await T.AutoModelForImageTextToText.from_pretrained(this.model, {
        dtype: DTYPE_PRESETS[this.preset] || DTYPE_PRESETS.compact,
        device: this.device,
        progress_callback: (m) => this._prog(onProgress, m),
      });
      this.T = T;
      this.ready = true;
      this.loadMs = performance.now() - t0;
      return true;
    })();
    try {
      return await this._loadPromise;
    } catch (e) {
      this.error = String(e && e.message || e);
      throw e;
    } finally {
      this.loading = false;
    }
  }

  _prog(onProgress, m) {
    if (!onProgress || !m) return;
    if (m.status === 'progress' && m.file) {
      this._files = this._files || {};
      this._files[m.file] = [m.loaded || 0, m.total || 0];
    }
    const f = this._files || {};
    let loaded = 0, total = 0;
    for (const k in f) { loaded += f[k][0]; total += f[k][1]; }
    const p = total ? loaded / total : 0;
    this._pMax = Math.max(this._pMax || 0, p);          // monoton: Balken zappelt nicht zurück
    const cur = f[this._curFile || ''] ;
    const name = Object.keys(f).slice(-1)[0] || '…';
    onProgress(`${(loaded / 1e6).toFixed(0)} / ${(total / 1e6).toFixed(0)} MB · ${String(name).split('/').pop()}`,
      this._pMax);
  }

  /**
   * @param {object} o system, history:[[role,text]], userText, canvas,
   *                   maxTokens, temperature? (nur do_sample), onDelta(text)
   * @returns {string} kompletter Antworttext
   */
  /** Dokumentkonform: single-turn (System + ein User-Turn mit <image>),
   *  kein Abbruch mitten in generate (Apple-Demo: skippen statt canceln),
   *  serieller Zugriff über Mutex statt busy-throw. */
  async generate({ system, history = [], userText, canvas, maxTokens = 96, onDelta }) {
    if (!this.ready) await this.load();
    while (this._mutex) await this._mutex;
    let release;
    this._mutex = new Promise((r) => { release = r; });
    this.busy = true;
    this.discard = false;
    const T = this.T;
    try {
      const messages = [];
      if (system) messages.push({ role: 'system', content: system });
      messages.push({ role: 'user', content: `<image>${userText}` });

      const prompt = this.processor.apply_chat_template(messages, { add_generation_prompt: true });

      let rawImg = null;
      if (canvas) {
        const ctx = canvas.getContext('2d', { willReadFrequently: true });
        const d = ctx.getImageData(0, 0, canvas.width, canvas.height);
        rawImg = new T.RawImage(d.data, d.width, d.height, 4);
      }
      const inputs = await this.processor(rawImg, prompt, { add_special_tokens: false });

      let streamed = '';
      const self = this;
      const streamer = new T.TextStreamer(this.processor.tokenizer, {
        skip_prompt: true,
        skip_special_tokens: true,      // wie Apples WebGPU-Demo
        callback_function: (t) => {
          streamed += t;
          if (!self.discard) onDelta && onDelta(t);   // discard = verwerfen, NICHT abbrechen
        },
      });

      await this.model_.generate({
        ...inputs,
        max_new_tokens: maxTokens,
        do_sample: false,               // greedy, wie Card & Apple-Demo
        repetition_penalty: 1.2,        // exakt Apples Demo-Wert
        streamer,
      });
      return streamed.trim();
    } finally {
      this.busy = false;
      release();
      this._mutex = null;
    }
  }

  /** Ausgabe verwerfen (TTS schweigt), Inferenz aber fertig laufen lassen. */
  discardTurn() { this.discard = true; }
  cancel() { this.discard = true; }

  /** 1-Token-Warmup: kompiliert Kernel/Shader, damit der erste echte Turn
   *  nicht minutenlang "denkt". Laeuft im Hintergrund, Fehler werden ignoriert. */
  async warmup() {
    if (!this.ready || this._warmed) return;
    this._warmed = true;
    try {
      await Promise.race([
        this.generate({ system: 'Antworte mit einem Wort.', history: [], userText: 'Hi', canvas: null, maxTokens: 2 }),
        new Promise((_, rej) => setTimeout(() => rej(new Error('warmup timeout')), 25000)),
      ]);
    } catch { /* warmup ist optional */ }
  }

  /** Retry-Schleife von transformers.js sofort beenden (offline/Block). */
  killSwitch() { if (this.T) { try { this.T.env.allowRemoteModels = false; } catch {} } }
}

/* Satz-Splitter für die TTS-Queue (Gegenstück zur serverseitigen Logik) */
export function splitSentences(buf) {
  const out = [];
  let rest = buf;
  let m;
  while ((m = rest.match(/[.!?…]+[\s"')\]]*/))) {
    const head = rest.slice(0, m.index + m[0].length).trim();
    rest = rest.slice(m.index + m[0].length);
    if (head.length >= 12 || !rest.trim()) out.push(head);
    else if (out.length) out[out.length - 1] += ' ' + head;
    else out.push(head);
  }
  while (rest.trim().length > 64) {          // harte Kappe: TTS bekommt nie Zeilen
    const cut = rest.lastIndexOf(' ', 64);
    const at = cut > 24 ? cut : 64;
    out.push(rest.slice(0, at).trim());
    rest = rest.slice(at);
  }
  return { out, rest };
}
