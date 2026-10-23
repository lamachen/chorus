/**
 * Reines Messinstrument. Keine App-Logik, keine Defaults, keine Fallbacks:
 * jeder Schritt schreibt Rohwerte (auch Fehler) in das Log. Ziel: erstmals
 * WAHRE Daten von echtem Geraet ueber diesen ONNX-Export + WebGPU/WASM.
 */
const log = document.getElementById('log');
const line = (txt, cls = '') => {
  const t = new Date().toISOString().slice(11, 19);
  log.innerHTML += `<span class="${cls}">[${t}] ${txt}</span>\n`;
  log.scrollTop = log.scrollHeight;
};
const report = () => log.innerText;

let T = null, processor = null, model = null, gpuDevice = null, camCanvas = null;

const MODEL = 'onnx-community/FastVLM-0.5B-ONNX';
const PRESET = { embed_tokens: 'q4f16', vision_encoder: 'q4f16', decoder_model_merged: 'q4f16' };

function fixedCanvas() {
  const c = document.createElement('canvas');
  c.width = 256; c.height = 256;
  const x = c.getContext('2d');
  x.fillStyle = '#204080'; x.fillRect(0, 0, 256, 256);
  x.fillStyle = '#e08020'; x.fillRect(40, 60, 120, 90);
  x.fillStyle = '#ffffff'; x.beginPath(); x.arc(190, 190, 40, 0, 7); x.fill();
  return c;
}

async function run(label, canvas, extra = {}, maxTokens = 32) {
  line(`── Run ${label} · canvas=${canvas ? 'ja' : 'NEIN (text-only)'} · max=${maxTokens} ${JSON.stringify(extra)}`);
  const messages = [
    { role: 'system', content: 'Antworte in genau einem kurzen Satz.' },
    { role: 'user', content: (canvas ? '<image>' : '') + 'Beschreibe in einem Satz, was zu sehen ist, oder sage HALLO, wenn kein Bild da ist.' },
  ];
  const prompt = processor.apply_chat_template(messages, { add_generation_prompt: true });
  const image = canvas
    ? new T.RawImage(canvas.getContext('2d', { willReadFrequently: true }).getImageData(0, 0, canvas.width, canvas.height).data, canvas.width, canvas.height, 4)
    : null;
  const inputs = await processor(image, prompt, { add_special_tokens: false });
  line(`   inputs: ${Object.keys(inputs).join(', ')} · ids ${JSON.stringify(inputs.input_ids.dims)} · px ${inputs.pixel_values ? JSON.stringify(inputs.pixel_values.dims) : 'FEHLEN'}`, 'dim');
  let out = '', first = null;
  const t0 = performance.now();
  const streamer = new T.TextStreamer(processor.tokenizer, {
    skip_prompt: true, skip_special_tokens: true,
    callback_function: (t) => { if (first === null) first = performance.now() - t0; out += t; },
  });
  try {
    await model.generate({ ...inputs, max_new_tokens: maxTokens, do_sample: false, streamer, ...extra });
  } catch (e) {
    line(`   GENERATE-FEHLER: ${e && e.message ? e.message : e}`, 'err');
  }
  const dt = ((performance.now() - t0) / 1000).toFixed(1);
  line(`   TTFT ${first === null ? '–' : (first / 1000).toFixed(1) + 's'} · gesamt ${dt}s · ${out.trim().length} Zeichen`);
  line(`   ROH-OUTPUT: »${out.trim()}«`, out.trim() ? 'ok' : 'warn');
  return out;
}

async function main() {
  log.innerHTML = '';
  line(`UA: ${navigator.userAgent}`);
  line(`secure=${window.isSecureContext} · cores=${navigator.hardwareConcurrency} · deviceMemory=${navigator.deviceMemory || '?'}GB · ua-platform=${navigator.platform || '?'}`);

  line('── WebGPU-Prüfung');
  if (!navigator.gpu) line('   navigator.gpu FEHLT → kein WebGPU', 'err');
  else {
    line('   navigator.gpu vorhanden');
    try {
      const ad = await navigator.gpu.requestAdapter();
      if (!ad) line('   requestAdapter() = null → kein nutzbarer Adapter', 'err');
      else {
        try { const i = await ad.info; line(`   Adapter: vendor=${i.vendor || '?'} arch=${i.architecture || '?'} desc=${i.description || '?'}`); }
        catch { line('   Adapter: ja (info nicht lesbar)', 'warn'); }
        try {
          gpuDevice = await ad.requestDevice();
          gpuDevice.lost.then((info) => line(`   !! GPU-DEVICE VERLOREN: ${info.reason} ${info.message} ← das ist die Crash-Wahrheit`, 'err'));
          line('   requestDevice() ok');
        } catch (e) { line(`   requestDevice() FEHLER: ${e.message}`, 'err'); }
      }
    } catch (e) { line(`   requestAdapter() FEHLER: ${e.message}`, 'err'); }
  }

  line('── transformers.js laden');
  try {
    T = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.5');
    line('   Import ok, Version ' + (T.env?.version || '?'), 'ok');
  } catch (e) { line('   Import FEHLER: ' + e.message, 'err'); return; }

  line('── Processor laden');
  const t1 = performance.now();
  try {
    processor = await T.AutoProcessor.from_pretrained(MODEL);
    line(`   ok in ${((performance.now() - t1) / 1000).toFixed(1)}s`, 'ok');
  } catch (e) { line('   FEHLER: ' + e.message, 'err'); return; }

  const device = gpuDevice ? 'webgpu' : 'wasm';
  line(`── Modell laden (q4f16, device=${device})`);
  const t2 = performance.now();
  try {
    model = await T.AutoModelForImageTextToText.from_pretrained(MODEL, { dtype: PRESET, device });
    line(`   ok in ${((performance.now() - t2) / 1000).toFixed(1)}s`, 'ok');
  } catch (e) {
    line(`   FEHLER auf ${device}: ${e.message}`, 'err');
    if (device === 'webgpu') {
      line('   Retry auf wasm …', 'warn');
      try {
        model = await T.AutoModelForImageTextToText.from_pretrained(MODEL, { dtype: PRESET, device: 'wasm' });
        line('   wasm-Load ok', 'ok');
      } catch (e2) { line('   wasm auch FEHLER: ' + e2.message, 'err'); return; }
    } else return;
  }

  await run('A · Card-Konvention (greedy, ohne Penalty)', fixedCanvas());
  await run('B · Apple-Demo-Konvention (greedy + rep 1.2)', fixedCanvas(), { repetition_penalty: 1.2 });
  await run('C · text-only (Kamera-aus-Pfad)', null);
  line('── Ende. Bericht bitte kopieren und senden.', 'ok');
}

document.getElementById('run').onclick = () => { main().catch((e) => line('UNERWARTETER FEHLER: ' + (e && e.stack || e), 'err')); };
document.getElementById('copy').onclick = async () => {
  try { await navigator.clipboard.writeText(report()); line('Bericht in Zwischenablage.', 'ok'); }
  catch { line('Clipboard blockiert – Text manuell markieren.', 'warn'); }
};
document.getElementById('cam').onclick = async () => {
  try {
    const st = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: 'environment' } } });
    const v = document.createElement('video');
    v.srcObject = st; await v.play();
    await new Promise((r) => setTimeout(r, 700));
    camCanvas = document.createElement('canvas');
    camCanvas.width = 384; camCanvas.height = 384;
    const x = camCanvas.getContext('2d');
    const vw = v.videoWidth, vh = v.videoHeight, sc = Math.max(384 / vw, 384 / vh);
    x.drawImage(v, (vw - 384 / sc) / 2, (vh - 384 / sc) / 2, 384 / sc, 384 / sc, 0, 0, 384, 384);
    st.getTracks().forEach((t) => t.stop());
    if (processor && model) await run('D · echter Kamera-Frame', camCanvas);
    else line('Kamera-Frame gespeichert – erst Modell laden, dann läuft Run D automatisch nicht; bitte Messreihe neu starten.', 'warn');
  } catch (e) { line('Kamera FEHLER: ' + e.message, 'err'); }
};
