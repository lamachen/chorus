# 📞 CHORUS · FastVLM Live-Vision-Sprachchat — komplett on-device

Eine Web-App wie eine Telefon-/Video-Chat-App: Du tippst einen Kontakt an,
„es klingelt", und dann redest du **freihändig** mit einer Person, die dein
Kamerabild sieht und dich hört — während ihre eigene Kamera aus ist.

**Alles läuft on-device in deinem Browser.** Kein Server, kein Backend, kein
Cloud-Roundtrip für Bild, Ton oder Text.

**▶ Live:** <https://lamachen.github.io/chorus/web/index.html>

---

## Was hier bewusst NICHT drin ist

| Ausgeschlossen | Warum |
|---|---|
| Geskriptete / vorgefertigte Antworten | Jede Äußerung der Figur generiert das VLM frei — auch die Begrüßung beim Verbinden. Es gibt keine Gruß-Texte, keine Floskel-Tabellen, keine Regel-Antworten. |
| Mock-/Demo-/Regel-Modus | Nichts spielt eine Person vor, die keine ist. Kann das VLM nicht laden, sagt die App das offen im Klingel-Screen und bleibt stehen. |
| Server (FastAPI/WebSocket/mlx/transformers) | ersatzlos gestrichen. Inferenz läuft im Browser (WebGPU, Fallback WASM). |
| Fake-Visualisierung | Alle Pegel kommen aus echten Signalen (siehe unten). |
| „Modell laden"-Button / Zustimmungs-Dialog | Mit dem Antippen des Kontakts lädt das Modell sofort — Fortschritt sichtbar im Klingel-Screen. |

---

## Die echte Pipeline

```
Kamera (Standard: RÜCK) ─▶ <video> ─▶ Cover-Crop ─▶ Grab-Canvas ─
                                                                  │  exakt dieselben Pixel
                                                                  ▼
                        FastVLM-0.5B-ONNX (transformers.js 3.7.5)
                        vision_encoder → embed_tokens → decoder (KV-Cache)
                        Gerät: WebGPU, sonst WASM · greedy · Streamer
                                                                  │ Tokens
                                                                  ▼
Mikro ─▶ AudioWorklet 16 kHz ─▶ Silero-VAD (ONNX/wasm) ─▶ Turn-Taking   Satz-Queue ─▶ speechSynthesis (System-Stimme)
        │                                                ▲
        └─ Pegel-Spike-Barge-in ───────────────────────────┘
Spracherkennung: System-ASR (webkitSpeechRecognition), halb-duplex, mit Health-Watchdog
```

* **Sehen:** `onnx-community/FastVLM-0.5B-ONNX` (vision_encoder + embed_tokens +
  decoder_model_merged), geladen über transformers.js — dieselbe Bibliothek und
  Aufrufkette wie Apples offizielles `apple/fastvlm-webgpu`-Demo. dtype-Presets:
  `compact` ≈ 676 MB (Default), `balanced`, `apple`, `fp16`. Einmaliger Download
  vom HF-CDN, danach Browser-Cache.
* **Sprechen:** System-TTS (`speechSynthesis`), automatische weibliche Stimme
  pro Charakter, Satz-Queue (Stimme startet, während noch generiert wird).
* **Hören:** System-ASR + Silero-VAD. Health-Watchdog baut die Erkennung hart
  neu, wenn sie 25 s stumm bleibt; der Chip zeigt jederzeit `läuft / wartet /
  aus / Fehler`.
* **Barge-in über zwei echte Kanäle:** VAD-Onset (während sie spricht Schwelle
  ×1,25, 5 Frames ≈ 160 ms) und Mikro-Pegelspike (>6× adaptiver Noisefloor über
  180 ms). Beide brechen Stimme **und** laufende Generierung ab.
* **Proaktives Sprechen:** numerischer Aktivierungs-Scorer
  (StreamBridge-Konzept, arXiv:2505.05467) auf den echten Frames — Neuheit,
  Bewegungsenergie, Gesprächspause. Er entscheidet nur, *wann* sie von selbst
  redet; *was* sie sagt, generiert das VLM.
* **Untere Bildschirmhälfte = exakt die VLM-Pixel:** Die Vorschau malt
  denselben Grab-Canvas, den der Processor bekommt (quadratischer Cover-Crop,
  identisch zum Center-Crop des Modells). HUD: Frame-#, Auflösung, fps → VLM.

## Kamera & ehrliches Audio-Feedback

* **Rückkamera ist Standard**, Front auf Wunsch; Wechsel-Button oben rechts im Videofeld.
* **Taschenlampe**: Button erscheint nur, wenn der Track `torch` kann (Android-Rückkamera).
* **Spiegelung `auto`**: Front gespiegelt, Rück ungespiegelt — immer identisch auf Aufnahme und Vorschau.
* **Pegel „DU"**: echter RMS-Verlauf aus dem VAD-AudioWorklet.
* **Pegel „SIE"**: Hüllkurve aus den **Word-Boundary-Events** der System-TTS + realer Satzfortschritts-Balken. Ohne Sprach-Events bleiben die Balken flach.

## Latenz („live ist live")

* Gemessen und angezeigt: **TTFA** = Redeende → erster Sprachton (Ø in den Einstellungen).
* Satz-Streaming in die TTS-Queue; greedy-Decoding; Turbo-Schalter (48 Tokens, max. 2 Sätze).
* Aktivierung/Scorer laufen nebenher und blockieren den Antwort-Pfad nie.
* Warum FastVLM: Der FastViTHD-Encoder erzeugt bewusst wenige Vision-Tokens →
  sehr kleiner Time-to-First-Token (Apple: 85× schneller als LLaVA-OneVision-0.5B) —
  genau die Eigenschaft, die einen Live-Call trägt.

## Charaktere & Sicherheit

Drei Frauen (Mara/Berlin, Juna/Wien, Noor/Amsterdam) in derselben Situation:
Random-Video-Chat, ihre Kamera ist aus/kaputt, dein Video läuft. Faktenblatt +
Stilregeln stehen im System-Prompt (`web/js/persona.js`); die Antworten sind
frei generiert. Der Prompt verbietet KI-Outing, Assistenten-Floskeln,
Technik-Auskünfte, erotische Inhalte und das Erfragen privater Daten; bei
Gefährdung bricht die Figur warm und ernst aus der Rolle (112 / Telefonseelsorge).

## Lokal entwickeln

```bash
cd web && python3 -m http.server 8443      # localhost ist secure context
# oder einfach die Pages-URL nutzen
pip install playwright && playwright install chromium
python tests/browser_smoke.py              # UI + ehrlicher Fehlerpfad (CDN blockiert)
npm i @huggingface/transformers@3.7.5 && node tests/ondevice_contract.mjs   # VLM-Vertrag
```

## Publizieren (GitHub Pages)

```bash
tools/publish.sh "Notiz"     # stamp -> sw.js + build_info.json, commit, push
```

Pages-Quelle ist `main:/`; der Service-Worker trägt den Publish-Stamp als
Cache-Namen (Network-First für Code), dadurch gibt es keinen stale Code mehr.
Jede Antwort mit Code-Änderung nennt Link + Commit-Sha. **Aktuell: siehe
letzten Publish-Kommentar / `web/build_info.json`.**

**Credentials:** liegen ausschließlich außerhalb des Repos
(`~/.secrets/gh_pat`, `~/.git-credentials`, chmod 600); das im Chat geteilte
Token gilt als kompromittiert und sollte rotiert werden.

## Dateien

```
index.html            Launcher (Redirect auf web/)
web/index.html        Telefon-UI
web/styles.css
web/js/app.js         Orchestrierung (on-device, kein Server)
web/js/vlm_webgpu.js  FastVLM-ONNX via transformers.js (WebGPU/WASM)
web/js/persona.js     Charakterbögen + System-Prompts (keine Antworten!)
web/js/activation.js  numerischer Proaktiv-Scorer
web/js/vad.js         Silero-VAD (einzige Implementierung)
web/js/vad-worklet.js AudioWorklet (16 kHz, 512-Frame)
web/js/speech.js      System-TTS + System-ASR
web/js/vision.js      Kamera → exakte VLM-Frames, Torch, Flip
web/js/ui.js          Telefon-Chrome, Settings-Sheet, Ring-Tone, DTMF
web/vendor/           silero_vad.onnx + onnxruntime-web (wasm)
tests/                browser_smoke.py, ondevice_contract.mjs
tools/publish.sh      Pages-Publish
```

Lizenzen: Silero VAD (MIT), onnxruntime-web (MIT), transformers.js (Apache-2.0),
FastVLM-Gewichte (Apple AMLR). StreamBridge-Konzept: Wang et al., NeurIPS 2025.
