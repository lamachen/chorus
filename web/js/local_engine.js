/**
 * On-Device-Fallback-Engine ("Demo-Modus").
 *
 * Wenn kein eigener Server erreichbar ist (z. B. direkt von GitHub Pages),
 * laeuft der komplette Charakter lokal im Browser: Bildanalyse ueber Canvas,
 * regelbasierte Antworten in Character, Wort-Streaming wie ein Token-Stream,
 * proaktives Sprechen ueber einen vereinfachten Aktivierungs-Scorer.
 *
 * Spracherkennung, VAD und Sprachausgabe bleiben dieselben System-APIs -
 * es faellt also nur der Netzwerk-Hop weg (Latenz: ~0 ms statt RTT).
 *
 * Fuer echtes On-Device-VLM siehe README ("On-Device & Latenz"):
 * FastVLM-0.5B als ONNX (onnx-community/FastVLM-0.5B-ONNX) laeuft ueber
 * onnxruntime-web/WebGPU - Apples eigenes Echtzeit-Demo macht genau das.
 */

const DE = [
  [["hallo", "hi ", "hey", "servus", "moin", "guten"], [
    "Hey! Na? Sag mal was, ich seh dich die ganze Zeit.",
    "Hallo du. Bissl komisch, dass meine Kamera aus ist, oder? Aber ich seh dich.",
  ]],
  [["wie heißt du", "heißt du", "dein name", "wer bist"], [
    "Ich bin {name}. Und du? Und nein, meine Kamera geht nicht, frag nicht.",
    "{name}. {age}, aus {city}. Deine Kamera funktioniert zum Glück, das reicht für uns beide.",
  ]],
  [["kamera", "video aus", "bild aus", "warum siehst"], [
    "Ja, ich weiß, ich weiß. Meine Kamera ist im Eimer, seit sie runtergefallen ist. Nervt mich selbst am meisten.",
    "Meine Kamera ist halt kaputt. Ich beschreib dir mich einfach, wenn du willst.",
  ]],
  [["was siehst du", "siehst du", "wie seh ich aus", "guck mal", "schaust"], [
    "{scene} Mehr kann ich dir ehrlich nicht sagen, so genau ist mein Blick nicht.",
    "{scene} Was willst du hören, ein Gedicht dazu?",
  ]],
  [["wo bist", "wo kommst", "wo wohnst", "welche stadt"], [
    "{city}. {fact0} Und du?",
    "Ich sitz hier in {city} rum. {fact1}",
  ]],
  [["was machst du", "was geht", "wie geht", "alles gut", "wie läuft"], [
    "Nicht viel. {fact2} Also eigentlich genau der richtige Moment für so einen Chat.",
    "Ganz gut. {fact3} Bei dir? Du siehst nicht aus, als würdest du gleich schlafen.",
  ]],
  [["arbeit", "job", "studierst", "studium", "schule", "beruf"], [
    "{fact1} Klingt unspektakulär, ist es auch meistens.",
  ]],
  [["musik", "hörst du", "lied", "song", "band"], [
    "{fact4} Sag nicht, du hörst was Vernünftiges.",
  ]],
  [["tier", "katze", "hund", "haustier"], [
    "{fact3} Und ja, sie liegt gerade wieder auf meinem Arm.",
  ]],
  [["alt", "alter", "jahre"], [
    "{age}. Und bevor du fragst: ja, ich sitze abends mit Fremden im Videochat. Läuft bei mir.",
  ]],
  [["langweilig", "müde", "schlafen", "gute nacht"], [
    "Dann geh schlafen, ich halt dich nicht auf. …oder bleib noch fünf Minuten.",
  ]],
  [["witz", "lach", "lustig", "spaß"], [
    "Ich bin schlecht in Witzen. Wirklich schlecht. Soll ich trotzdem?",
  ]],
  [["ki ", "künstliche", "bot", "roboter", "bist du echt", "chatgpt"], [
    "Hä? Was soll das denn jetzt. Ich sitz hier mit kaputter Kamera rum und du fragst, ob ich echt bin.",
  ]],
  [["wetter", "regnet", "sonne", "draußen"], [
    "Hier in {city} ist es grau, wie immer. {fact6}",
  ]],
  [["?"], [
    "Gute Frage. Ehrlich? Keine Ahnung. Was denkst du denn?",
  ]],
];

const GENERIC = [
  "Hm. {scene} Sag mehr, ich hör zu.",
  "Okay, warte, das musst du mir erklären. Ich mein, ich sitz hier nur mit Ton.",
  "Echt? Das hätte ich jetzt nicht gedacht. Erzähl weiter.",
  "Haha, okay. Und was ist daraus geworden?",
  "Weißt du was, genau darüber hab ich heute auch nachgedacht. Bissl zumindest.",
];

const ASK = [
  "Und, was machst du da gerade?", "Bist du auch so ein Nachtmensch?",
  "Wo kommst du eigentlich her?", "Was läuft bei dir so, erzähl mal.",
  "Und, guter Tag heute oder eher so naja?",
];

const PROACTIVE = [
  "{scene} Sag mal, ist das bei dir immer so?",
  "Du, mir ist gerade aufgefallen: {sceneLow} Stimmt das?",
  "Hm, bei mir ist gerade alles ruhig. {scene} Erzähl mal was.",
  "Ich muss kurz was erzählen: {fact5} Aber sorry, raus damit: {sceneLow}",
];

const pick = (a) => a[(Math.random() * a.length) | 0];

export class LocalEngine {
  constructor(app, persona) {
    this.app = app;
    this.p = persona || { name: 'Mara', age: 24, city: 'Berlin', hue: 326 };
    this.facts = persona?.facts || [
      'wohnt in einer WG, 4. Stock ohne Aufzug',
      'arbeitet drei Tage die Woche in einer Kaffeebar',
      'hat eine getigerte Katze namens Piko',
      'kommt ursprünglich aus Kiel',
      'hört gerade viel UK-Garage und alten Deutschrap',
      'trinkt zu viel Filterkaffee, isst Müsli um 23 Uhr',
      'ihr Fenster geht auf einen Hinterhof, nachts hört sie die U-Bahn',
    ];
    this.cancel = false;
    this.busy = false;
    this.prev = null;          // Uint8ClampedArray 48x48
    this.anchor = null;
    this.lastSpeech = performance.now();
    this.lastPro = performance.now();
    this._timer = null;
  }

  /* ---------------- Bildmerkmale (rein client-seitig) ---------------- */
  analyse() {
    const src = this.app.vision?.grab;
    if (!src || !src.width) return null;
    const c = this._c || (this._c = document.createElement('canvas'));
    c.width = 48; c.height = 48;
    const x = c.getContext('2d', { willReadFrequently: true });
    x.drawImage(src, 0, 0, 48, 48);
    const d = x.getImageData(0, 0, 48, 48).data;
    let sum = 0, sum2 = 0, rs = 0, gs = 0, bs = 0, edge = 0;
    const hues = new Array(8).fill(0);
    for (let i = 0; i < d.length; i += 4) {
      const r = d[i], g = d[i + 1], b = d[i + 2];
      const l = 0.299 * r + 0.587 * g + 0.114 * b;
      sum += l; sum2 += l * l;
      rs += r * r; gs += g * g; bs += b * b;
      if (i >= 4) edge += Math.abs(l - (0.299 * d[i - 4] + 0.587 * d[i - 3] + 0.114 * d[i - 2]));
      const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
      if (mx - mn > 24) {
        let h;
        if (mx === r) h = ((g - b) / (mx - mn) + 6) % 6;
        else if (mx === g) h = (b - r) / (mx - mn) + 2;
        else h = (r - g) / (mx - mn) + 4;
        hues[(h * 8 / 6) | 0]++;
      }
    }
    const n = d.length / 4;
    const mean = sum / n;
    const bright = mean / 255;
    const colorfulness = Math.min(1, Math.sqrt(rs / n + gs / n + bs / n) / 255 * 0.62);
    const edges = Math.min(1, edge / n / 60);
    const hueNames = ['Rot', 'Orange', 'Gelb', 'Grün', 'Türkis', 'Blau', 'Violett', 'Pink'];
    const dom = hues.indexOf(Math.max(...hues));
    const changed = this.prev ? this._diff(d, this.prev) : 0;
    this.prev = d;
    return {
      bright, colorfulness, edges, changed,
      hue: Math.max(...hues) > n * 0.06 ? hueNames[dom] : null,
      warm: (rs - bs) / n / 255 > 0.06,
    };
  }
  _diff(a, b) {
    let s = 0;
    for (let i = 0; i < a.length; i += 16) s += Math.abs(a[i] - b[i]);
    return Math.min(1, s / (a.length / 16) / 60);
  }
  scene(f) {
    if (!f) return 'Ich seh dich gerade nicht so gut.';
    const light = f.bright < 0.18 ? 'sehr dunkel' : f.bright < 0.34 ? 'dämmrig'
      : f.bright < 0.62 ? 'normal beleuchtet' : f.bright < 0.8 ? 'hell' : 'sehr hell';
    let s = `Das Bild ist ${light}`;
    if (f.colorfulness > 0.3) s += ' und ziemlich bunt';
    else if (f.colorfulness < 0.12) s += ' und fast farblos, eher grau';
    if (f.hue) s += `, dominierend ${f.hue}`;
    if (f.edges > 0.22) s += '. Viele Kanten, sieht unruhig aus';
    else if (f.edges < 0.07) s += '. Sehr glatte Flächen, wahrscheinlich Wand oder Himmel';
    if (f.warm) s += '. Warmes Licht, eher abends oder eine Lampe';
    return s + '.';
  }

  fill(tpl, f) {
    return tpl
      .replaceAll('{name}', this.p.name).replaceAll('{age}', this.p.age ?? 24)
      .replaceAll('{city}', this.p.city ?? 'Berlin')
      .replaceAll('{scene}', this.scene(f))
      .replaceAll('{sceneLow}', (this.scene(f) || '').replace(/^./, (c) => c.toLowerCase()))
      .replaceAll('{fact0}', this.facts[0]).replaceAll('{fact1}', this.facts[1])
      .replaceAll('{fact2}', this.facts[2]).replaceAll('{fact3}', this.facts[3])
      .replaceAll('{fact4}', this.facts[4]).replaceAll('{fact5}', this.facts[5])
      .replaceAll('{fact6}', this.facts[6]);
  }

  /* ---------------- Protokoll-Schnittstelle (wie der Server) ---------------- */
  post(o) {
    switch (o.t) {
      case 'hello':
        this.emit({ t: 'ready', engine: { backend: 'on-device (Browser)', model: 'demo-regeln', ready: true }, llm: { enabled: false } });
        setTimeout(() => this.turn('greeting', ''), 500);
        this._proTimer = setInterval(() => this.probe(), 3000);
        break;
      case 'say': if (o.final) this.turn('user', o.text || ''); break;
      case 'probe': this.turn('probe', ''); break;
      case 'interrupt': this.cancel = true; this.emit({ t: 'cancelled', reason: o.reason }); break;
      case 'vad': if (!o.speaking) this.lastSpeech = performance.now(); break;
      case 'speaking': if (!o.value) this.lastSpeech = performance.now(); break;
      case 'bye': clearInterval(this._proTimer); break;
    }
  }
  observe() { this._f = this.analyse(); }

  probe() {
    if (this.busy || this.app.tts.speaking || this.app.vad?.gate?.speaking) return;
    const f = this._f || this.analyse();
    const lull = (performance.now() - this.lastSpeech) / 1000;
    const novelty = f?.changed ?? 0;
    const score = Math.min(1, novelty * 2.6 + Math.min(1, lull / 12) * 0.9);
    const speak = score > 0.62 && lull > 6 && performance.now() - this.lastPro > 22000;
    this.emit({ t: 'activation', score: +score.toFixed(3), speak, mode: 'on-device', reason: speak ? 'Bild/Pause' : 'wartet' });
    if (speak) { this.lastPro = performance.now(); this.turn('proactive', ''); }
  }

  emit(o) { this.app.dispatch(o); }

  /* ---------------- Turn: Worte streamen wie Tokens ---------------- */
  async turn(reason, userText) {
    if (this.busy) return;
    this.busy = true; this.cancel = false;
    const f = this._f || this.analyse();
    const t = (userText || '').toLowerCase();
    let reply;
    if (reason === 'greeting') reply = pick(this.p.greetings || this.app.serverReady?.persona?.greetings || []) ||
      `Hey! Kurzer Hinweis: meine Kamera ist tot, die bleibt schwarz. Aber ich hör dich und ich seh dich.`;
    else if (reason === 'probe') reply = this.scene(f);
    else if (reason === 'proactive') reply = this.fill(pick(PROACTIVE), f);
    else {
      const hit = DE.find(([keys]) => keys.some((k) => t.includes(k)));
      reply = this.fill(hit ? pick(hit[1]) : pick(GENERIC), f);
      if (Math.random() < 0.4) reply += ' ' + pick(ASK);
    }
    this.emit({ t: 'status', state: 'thinking', reason });
    const words = reply.split(' ');
    const buf = [];
    let sentCount = 0;
    for (let i = 0; i < words.length; i++) {
      if (this.cancel) break;
      const w = words[i] + (i < words.length - 1 ? ' ' : '');
      buf.push(w);
      this.emit({ t: 'delta', text: w });
      const acc = buf.join('');
      if (/[.!?]$/.test(words[i]) && acc.length > 14) {
        sentCount++;
        this.emit({ t: 'sentence', text: acc.trim(), i: sentCount });
        buf.length = 0;
        if (sentCount >= 3) break;
      }
      await new Promise((r) => setTimeout(r, 22 + Math.random() * 45));
    }
    if (buf.join('').trim() && !this.cancel) {
      sentCount++;
      this.emit({ t: 'sentence', text: buf.join('').trim(), i: sentCount });
    }
    this.busy = false;
    this.lastSpeech = performance.now();
    this.emit({ t: 'turn_end', text: reply, reason, ms: 0, ttft_ms: 0, frames_seen: this.app.vision?.frameId || 0 });
    this.emit({ t: 'status', state: 'idle', reason: 'turn_end' });
  }
}

/* ------------------------------------------------------------------ */
/*  On-Device-Faehigkeits-Probe (WebGPU etc.)                          */
/* ------------------------------------------------------------------ */
export async function probeOnDevice() {
  const out = {
    webgpu: !!navigator.gpu,
    cores: navigator.hardwareConcurrency || 0,
    memGB: navigator.deviceMemory || 0,
    ua: /iPhone|iPad/.test(navigator.userAgent) ? 'iOS' : /Android/.test(navigator.userAgent) ? 'Android' : 'Desktop',
    secure: window.isSecureContext,
    wasm: typeof WebAssembly === 'object',
    simd: false,
  };
  try { out.simd = await WebAssembly.validate(new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11])); } catch {}
  const gpuOk = out.webgpu && out.wasm;
  out.verdict = gpuOk
    ? `WebGPU vorhanden (${out.ua}, ${out.cores} Kerne) → On-Device-Pfad mit FastVLM-0.5B-ONNX über onnxruntime-web ist prinzipiell möglich`
    : `Kein WebGPU in diesem Browser → On-Device-VLM nicht verfügbar, Server- oder Demo-Modus nutzen`;
  return out;
}

/* Fallback-Kontakte, falls kein Server antwortet (Pages-Demo) */
export const DEFAULT_PERSONAS = [
  {
    id: 'mara', name: 'Mara', age: 24, city: 'Berlin', hue: 326,
    tagline: 'Video-Chat · Kamera aus · online',
    facts: [
      'wohnt in einer WG in Neukölln, 4. Stock ohne Aufzug',
      'arbeitet drei Tage die Woche in einer Kaffeebar, macht nebenbei ein Fernstudium (Medien)',
      'hat eine getigerte Katze namens Piko, die ständig auf der Tastatur liegt',
      'kommt ursprünglich aus Kiel und vermisst manchmal das Meer',
      'hört gerade viel UK-Garage und alten Deutschrap, früher Geige',
      'trinkt zu viel Filterkaffee und isst Müsli um 23 Uhr',
      'ihr Fenster geht auf einen Hinterhof, nachts hört sie die U-Bahn',
    ],
    greetings: [
      'Oh, hi! Ey, kurzer Hinweis vorweg: meine Kamera ist tot, die bleibt schwarz. Aber ich hör dich und ich seh dich, das reicht doch.',
      'Hallo? Cool, du hast Video an. Bei mir ist die Kamera leider im Eimer, also musst du für uns beide gucken.',
    ],
  },
  {
    id: 'juna', name: 'Juna', age: 21, city: 'Wien', hue: 196,
    tagline: 'Video-Chat · Kamera aus · online',
    facts: [
      'studiert im 5. Semester Grafikdesign und hasst Typografie-Korrekturen',
      'wohnt in einer alten Wohnung im 7. Bezirk, der Nachbar übt Trompete',
      'hat einen Mitbewohner namens Tobi, der immer das letzte Joghurt isst',
      'fährt jeden Tag mit der U6 und hasst die U6',
      'zeichnet Comics über ihre Katze Zimt, die hat 1200 Follower',
      'kann nicht kochen, macht aber den besten Espresso der Straße',
      'war ein halbes Jahr in Kopenhagen und redet zu viel davon',
    ],
    greetings: [
      'Servus! Also, peinlich: mein Video geht ned, schwarzes Bild, ich hab alles probiert. Dafür hör ich dich eh.',
      'Huhu! Du hast Kamera, ich hab keine - deal? Ich seh dich, du hörst mich.',
    ],
  },
  {
    id: 'noor', name: 'Noor', age: 27, city: 'Amsterdam', hue: 152,
    tagline: 'Video-Chat · Kamera aus · online',
    facts: [
      'arbeitet als Fahrradmechanikerin in einem kleinen Laden an der Gracht',
      'wohnt über einer Bäckerei, es riecht morgens um fünf nach Brot',
      'hat ein altes Rennrad namens de Blauwe, das nie abgeschlossen wird',
      'lernt seit einem Jahr Deutsch mit einer App',
      'spielt Bass in einer Band, die seit zwei Jahren kein Konzert hatte',
      'liest gerade zu viel über Mykologie und fotografiert sonntags Pilze',
      'telefoniert jeden Dienstag mit ihrer Schwester in Rotterdam',
    ],
    greetings: [
      'Hey. Kleine Info: Kamera ist bei mir aus, absichtlich. Ich seh dich trotzdem, reden können wir auch.',
      'Hallo. Ich hab kein Bild, nur Stimme. Ist das okay für dich?',
    ],
  },
];
