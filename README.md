# 📞 CHORUS · FastVLM Live-Vision-Sprachchat — komplett on-device

Eine Web-App wie eine Telefon-/Video-Chat-App: Kontakt antippen, „es klingelt",
dann **freihändig** reden mit einer Person, die dein Kamerabild sieht und dich
hört — während ihre eigene Kamera aus ist.

**Alles läuft on-device im Browser.** Kein Server, kein Backend, keine Cloud
für Bild, Ton oder Text. **▶ Live:** <https://lamachen.github.io/chorus/web/index.html>

---

## Was hier bewusst NICHT drin ist

| Ausgeschlossen | Warum |
|---|---|
| Geskriptete Antworten, Gruß-Tabellen, Regel-Charakter, Mock | Jede Äußerung generiert das VLM frei — auch die Begrüßung. |
| Demo-/Fallback-Modi, die eine Person vorspielen | Kann das VLM nicht laden, zeigt die App den Fehler offen und bleibt stehen. |
| Server (FastAPI/WebSocket/mlx/transformers) | ersatzlos gestrichen; Inferenz im Browser (WebGPU, sonst WASM). |
| Fake-Visualisierung | Alle Pegel aus echten Signalen (Mik-RMS, TTS-Word-Boundaries). |
| Buchstabensalat-Ausgaben | Gibberish-Gate bricht Turns ab, bevor die TTS etwas bekommt. |

---

## 1 · Die Signal-Kette (jede Stufe existent und getestet)

```
[1] Mikro ─AudioWorklet 16kHz─▶ [2] Silero-VAD ─▶ Turn-Ende / Barge-in
[1b] System-ASR ─Finals─▶ [3] Transkript: live „🎙 …", committed „✓ gehört: …" + Du-Bubble
[3] + [4] aktueller Frame (Grab-Canvas) ─▶ [5] VLM-Chat-Template
        (System-Prompt + Merkzettel + History + <image> + dein Text)
[5] ─greedy generate─▶ [6] Tokens ─▶ [7] Gibberish-Gate ─▶ [8] Satz-Splitter (max 64 Z.)
[8] ─▶ [9] System-TTS ─▶ [10] gespiegelte Echtzeit-Waveform (Mitte des Screens)
[11] Aktivierungs-Scorer (Frames + Pause) ─▶ [5] als proaktiver Turn
[12] Tastatur-Zeile = gleichwertiger Eingang, wenn System-ASR fehlt
```

* **Wie bekommt das Modell deine Nachricht?** System-ASR → Finals → Commit nach
  VAD-Pause → dein Text + aktueller Frame + History ins Chat-Template.
  FastVLM **hört nicht selbst** (Vision+Text): „hören" = System-ASR, „sehen" = Frames.
* **Systemtest in der App** (Einstellungen → „Systemtest"): misst jede Stufe real
  (Secure Context, Kamera-Varianz, Mikro-Pegel, Silero, ASR, TTS-Start,
  VLM-Mini-Inferenz in ms) und fährt zum Schluss die **Ground-Truth-Messreihe**:
  fester Frame + fester Prompt, Rohausgabe in Zeichen + ms sichtbar —
  Buchstabensalat wäre hier lesbar, bevor er jemals gesprochen wird.
* **`tests/chain_test.py`**: 12 Asserts über alle internen Übergänge (aktuell 12/12).
* **`tests/browser_smoke.py`**: ehrlicher Fehlerpfad (CDN weg → sichtbarer Fehler)
  + Layout-Invarianten (3 Zonen, Waveform-Band, keine Fake-Kamera-Box).

## 1d · Wann antwortet sie? Mit welchen Frames? Mit welchem Gedächtnis?

1. **Du sprichst** → ASR-Finals live sichtbar, VAD meldet Sprachende → ~0,6 s
   Pause → Commit. (Oder Tastatur-Zeile — gleicher Pfad.)
2. **Commit** → „✓ gehört: …" bleibt stehen, bis ihre erste Sentence kommt;
   Du-Bubble ins Protokoll.
3. **VLM-Turn** → streng nach öffentlicher Doku (ONNX-Card + Apples
   WebGPU-Demo): **single-turn** — System-Prompt (Charakter + Merkzettel +
   letzter Austausch wörtlich) + **ein** User-Turn mit `<image>` direkt vor
   dem Text + **immer ein Bild**: der live angezeigte Frame, bei Kamera aus
   das eingefrorene Restbild (Prompt-Regel: als „kein Bild" behandeln).
   Kein Multi-Turn-Template, kein text-only-Prefill — beides ist für diesen
   Export öffentlich nicht belegt und war nachweislich eine Fehlerquelle.
4. **Tokens** → Gibberish-Gate (≥24 Zeichen Prüffenster; Salat ⇒ Ausgabe wird
   verworfen, nichts an die TTS) → Satz-Splitter (hart bei 64 Zeichen) → TTS →
   Waveform. **Nie wird ein laufendes `generate` abgebrochen** (kein
   WebGPU-Abort mid-flight): Barge-in/Watchdog/Gate setzen nur ein
   Discard-Flag (Stream wird verworfen, TTS schweigt), die Inferenz läuft wie
   in Apples Demo hinter einem Lock zu Ende; neue Turns warten serialisiert.
5. **Denkt-Watchdog**: Status zeigt „denkt nach … Xs (tippen = abbrechen)";
   ohne 1. Token in 60 s (WebGPU) / 150 s (WASM) bricht der Turn ehrlich ab.
   Nach dem Laden: 1-Token-Warmup, damit der erste Turn nicht die
   Shader-Kompilierung zahlt.

**Frames:** kontinuierlich erfasst; jeder Turn nutzt den aktuell angezeigten
Canvas — kein unsichtbares anderes Bild. Proaktive Turns: derselbe Frame +
Direktion („Stille … führe das Gespräch weiter"); der Scorer entscheidet nur das Wann.

**Gedächtnis:** letzte 6 Runden wörtlich + **Merkzettel**: alle 4 Runden fasst
das VLM nebenher (alter Merkzettel + letzte Runden → ≤ 2 Sätze) zusammen; der
Merkzettel steht im System-Prompt und wird zum Bezugnehmen gefordert.

**Kein Bild-Geplapper:** Licht/Farben/Lampen als Füllstoff sind verboten; Video
nur bei sichtbarer Änderung, auf Frage oder wenn es wirklich passt. Proaktive
Turns nehmen bevorzugt den Verlauf auf.

## 2 · Layout (vertikal, drei Zonen)

* **Oben** Charakter/App: Avatar, Name, Status, ihre Untertitel, kleine Icons
  (Untertitel, Protokoll), Timer. Kein „Kamera aus"-Text, kein Fake-Kamerabild.
* **Mitte (exakt halbhohe Trennlinie):** gespiegelte Echtzeit-**Waveform** —
  oben ihre Stimme (Hüllkurve aus TTS-Word-Boundaries), unten dein Mikro
  (scrollender RMS). Canvas, 96 px, vertikal zentriert durch gleiche Flex-Anteile.
* **Unten** deine Seite: exakt die VLM-Pixel (quadratischer Cover-Crop =
  Center-Crop des Modells), HUD (Frame-#, Auflösung, fps), deine Untertitel,
  Blitz/Wechsel klein im Videofeld, darunter 4 Controls + Auflegen.
* Diagnose-Chips wohnen im Protokoll, nicht im Call-Screen.

## 3 · Kamera & Audio-Feedback

* **Rückkamera Standard**, Front auf Wunsch; Wechsel-Button; **Taschenlampe**
  nur sichtbar, wenn der Track `torch` kann; Spiegelung `auto`
  (Front gespiegelt, Rück nicht) — immer identisch auf Aufnahme und Vorschau.
* Pegel echt: Mikro-RMS aus dem Worklet; ihre Hüllkurve aus Word-Boundary-Events.

## 4 · On-Device-VLM

* `onnx-community/FastVLM-0.5B-ONNX` über **transformers.js 3.7.5** (dieselbe
  Kette wie Apples `apple/fastvlm-webgpu`-Demo), WebGPU sonst WASM.
* Quantisierungen exakt nach der öffentlichen Tabelle (Talk2FastVLM /
  onnx-community), alle Varianten wählbar, Größen pro Komponente
  (Decoder/Embed/Vision):

  | Variante | D | E | V | Summe | Einordnung |
  |---|---|---|---|---|---|
  | **q4f16 (DEFAULT)** | 282 | 272 | 253 | 807 MB | Doku-Empfehlung „for most users" |
  | fp16 | 992 | 272 | 253 | 1517 MB | „better accuracy" |
  | kleinst-Mix | 282 | 136 | 223 | 641 MB | memory-constrained (uint8 E/V + q4f16 D) |
  | int8 | 503 | 136 | 223 | 862 MB | memory-constrained |
  | uint8 | 503 | 136 | 223 | 862 MB | memory-constrained (Salat-Verdacht, Gate wacht) |
  | apple-Mix | 317 | 272 | 505 | 1095 MB | Wortlaut der Apple-WebGPU-Demo |
  | bnb4 | 287 | 544 | 505 | 1336 MB | — |
  | q4 | 317 | 544 | 505 | 1366 MB | — |
* Generate-Parameter wie Doku: `do_sample:false`, `repetition_penalty:1.2`
  (Apples Demo-Wert), `TextStreamer(skip_prompt:true)`.
* Load startet **beim Seitenstart** (Dialer-Chip: `VLM lädt x %` → `bereit` →
  `Fehler`), kein Button/Dialog; Preflight vorab, Byte-Watchdog, Kill-Switch
  gegen Retry-Stürme; Service-Worker fasst keine Cross-Origin-Fetches an.
* Latenz-KPIs: TTFT (1. Token) und TTFA (Redeende → erste Stimme) in den Einstellungen.

## 5 · Charaktere & Sicherheit

Mara/Berlin, Juna/Wien, Noor/Amsterdam: Random-Video-Chat, ihre Kamera ist aus/
kaputt, dein Video läuft. Faktenblatt + Stilregeln im System-Prompt
(`web/js/persona.js`); Antworten frei generiert. Verboten: KI-Outing,
Assistenten-Floskeln, Technik-Auskünfte, erotische Inhalte, Privatdaten;
bei Gefährdung bricht die Figur warm und ernst aus der Rolle (112 / Telefonseelsorge).

## 6 · Lokal entwickeln & publizieren

```bash
cd web && python3 -m http.server 8443        # localhost = secure context
pip install playwright && playwright install chromium
python tests/browser_smoke.py                # Fehlerpfad + Layout
python tests/chain_test.py                   # 12/12 Ketten-Stufen
npm i @huggingface/transformers@3.7.5 && node tests/ondevice_contract.mjs
tools/publish.sh "Notiz"                     # Stamp → sw.js/build_info, push, Pages baut
```

Pages-Quelle `main:/`; SW-Cache-Name = Publish-Stamp (Network-First für Code),
dadurch kein stale Code. Jede Antwort mit Code-Änderung nennt Link + Sha.

**Credentials:** nur außerhalb des Repos (`~/.git-credentials`, chmod 600);
ein im Chat geteiltes Token gilt als kompromittiert → rotieren.

## 7 · Dateien

```
index.html           Launcher (Redirect auf web/)
web/index.html       Telefon-UI (3 Zonen)
web/styles.css
web/js/app.js        Orchestrierung, Turn-Loop, Watchdogs, Merkzettel
web/js/vlm_webgpu.js FastVLM-ONNX via transformers.js, Gibberish-Gate, Splitter
web/js/persona.js    Charakterbögen + System-Prompts (Regeln, keine Antworten)
web/js/activation.js numerischer Proaktiv-Scorer (StreamBridge-Konzept)
web/js/vad.js        Silero-VAD (einzige Implementierung)
web/js/vad-worklet.js AudioWorklet (16 kHz, 512-Frame)
web/js/speech.js     System-TTS + System-ASR (+ Health-Watchdog)
web/js/vision.js     Kamera → exakte VLM-Frames, Torch, Flip
web/js/ui.js         Telefon-Chrome, Settings-Sheet, Ring-Tone, DTMF
web/diag.html        reines Messinstrument (web/js/diag.js): Adapter-Wahrheit,
                     Load-Zeiten, Roh-Outputs A/B/C/D, GPU-Device-Lost-Watcher
web/vendor/          silero_vad.onnx + onnxruntime-web (wasm)
tests/               browser_smoke.py · chain_test.py · ondevice_contract.mjs · shots.py
tools/publish.sh     Pages-Publish
```

Lizenzen: Silero VAD (MIT), onnxruntime-web (MIT), transformers.js (Apache-2.0),
FastVLM (Apple AMLR). StreamBridge-Konzept: Wang et al., NeurIPS 2025.
