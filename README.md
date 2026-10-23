# 📞 FastVLM Video-Call

Eine Web-App, die sich wie eine **Telefon-/Video-Chat-App** anfühlt und in der du
per **freihändigem Sprachchat** mit einer weiblichen KI-Persona redest, die
glaubt, mit dir in einer Random-Video-Chat-App (Omegle-artig) gelandet zu sein:
**ihre Kamera ist aus, deine ist an.** Sie *sieht* dein Video (Apples
**FastVLM**), sie *hört* dich (System-Spracherkennung) und sie *spricht* mit der
Stimmenausgabe deines Betriebssystems/Browsers – kurz, menschlich und von sich
aus.

Alles läuft **lokal**: Bild, Ton und Text verlassen dein Netz nicht.

**▶ Live-Demo (ganz ohne Server, on-device im Browser):**
<https://lamachen.github.io/chorus/web/index.html> — findet die App keinen
Server, schaltet sie automatisch in den **On-Device-Demomodus**: Charakter,
Bildanalyse, VAD, Spracherkennung und Stimme laufen dann vollständig im
Browser (z. B. direkt von GitHub Pages). Kamera-/Mikrofreigabe bleibt nötig.

| Bildschirm | Inhalt |
|---|---|
| `shots/1-dialer.png` | Telefon-App: Favoriten, Keypad, Verlauf, Kontakte |
| `shots/4-ringing.png` | Klingeln (Ringback-Ton + Vibration) |
| `shots/5-call.png` | Anruf: oben **ihre** Seite (Kamera aus, Stimme, Untertitel), unten **exakt die Frames, die FastVLM bekommt** |
| `shots/6-transcript.png` | Gesprächsprotokoll |
| `shots/3-settings.png` | Einstellungen (Stimme, VAD, Bild, Charakter, Engine) |

---

## 1 · Schnellstart

```bash
pip install -r requirements.txt     # fastapi, uvicorn, pillow (+ optional: mlx-vlm oder torch/transformers)
./run.sh                            # erzeugt ein selbstsigniertes Zertifikat und startet HTTPS auf :8443
```

* **Auf dem Rechner:**  `https://localhost:8443`
* **Vom Handy (gleiches WLAN):** `https://<IP-des-Rechners>:8443`
  Beim ersten Öffnen warnt der Browser vor dem selbstsignierten Zertifikat →
  einmalig *„Erweitert → Trotzdem fortfahren“*. Das ist zwingend, weil Kamera,
  Mikro und Spracherkennung nur in einem *secure context* freigegeben werden.

Ohne installierte Modellgewichte startet der Server automatisch im
**Mock-Modus**: die komplette Pipeline (Kamera → Frame → WebSocket → Antwort →
TTS, inklusive VAD und proaktivem Sprechen) funktioniert, nur antwortet eine
eingebaute Platzhalter-Persona, die trotzdem echte Bildmerkmale (Helligkeit,
Farben, Kanten, Änderung zum Vorframe) beschreibt. Perfekt zum Entwickeln und
für Demos ohne GPU.

### FastVLM-Backend wählen

| Backend | Installation | Modell (Default) | Wann |
|---|---|---|---|
| **mlx** | `pip install -U mlx-vlm` | `mlx-community/FastVLM-0.5B-bf16` | Apple Silicon – empfohlen, schnellster TTFT |
| **transformers** | `pip install -U torch transformers accelerate` | `apple/FastVLM-0.5B` | NVIDIA-GPU / CPU |
| **mock** | – | – | ohne Gewichte, zum Testen |

Größere Modelle über `.env` (siehe `.env.example`):
`apple/FastVLM-1.5B`, `apple/FastVLM-7B`, MLX-Quantisierungen z. B.
`EZCon/FastVLM-1.5B-4bit-mlx`, `InsightKeeper/FastVLM-0.5B-MLX-4bit`.
`FASTVLM_BACKEND=auto` wählt automatisch (macOS+mlx-vlm → mlx, sonst
transformers, sonst mock); fällt ein Backend beim Laden fehl, springt der
Server auf mock um, statt den Anruf sterben zu lassen.

### Optional: besseres Gesprächsgehirn

FastVLM-0.5B ist ein exzellentes Auge, aber ein bescheidener Smalltalker.
Mit `LLM_BACKEND=ollama` (oder `openai`, jede OpenAI-kompatible URL) wird die
Arbeit geteilt: **FastVLM beschreibt dauerhaft, was sie sieht** (Scene-Notes),
ein lokales LLM (z. B. `llama3.1:8b`) formuliert die Antworten in Character.
Einstellbar auch live über den Schalter „Externes LLM fürs Reden“.

---

## 1b · Latenz & On-Device

„Live ist live": Entscheidend ist die Zeit bis zur **gehörten** Antwort, nicht
bis zum ersten Token. Deshalb misst und optimiert die App genau das:

* **TTFT** (Server → erster Token) und **TTFA** (dein Redeende → erster
  Sprachton) werden live gemessen und in den Einstellungen als Ø angezeigt.
* **Satz-Streaming**: Die Stimme beginnt mit Satz 1, während Satz 2 noch
  generiert wird (Server *und* Demomodus).
* **Turbo-Profil** (Schalter): 512-px-Frames, greedige Dekodierung, max.
  2 Sätze, kürzere Scene-Notes – minimiert TTFT auf Kosten von etwas Tiefe.
* **Frame-Prefetch**: Frames laufen kontinuierlich und getrennt vom Dialog;
  Scene-Notes und Aktivierungs-Probes rechnen nur, wenn die Engine idle ist,
  sodass sie den Antwort-Pfad nie blockieren.
* **Warum FastVLM hier richtig ist**: FastVLMs FastViTHD-Encoder produziert
  bewusst wenige Vision-Tokens und hat dadurch einen extrem kleinen
  Time-to-First-Token (Apple: 85× schneller als LLaVA-OneVision-0.5B) – genau
  die Eigenschaft, die einen Live-Videocall trägt.

**On-Device-Stufen** (heute → morgen):
1. *Heute, überall:* On-Device-Demomodus (Regel-Charakter + System-Speech) –
   null Netzwerk-Latenz, läuft auch auf Mittelklasse-Handys.
2. *Heute, LAN:* Server-Modus mit mlx-vlm (Apple Silicon) oder CUDA; das Handy
   bleibt Thin Client mit Hardware-VAD.
3. *Implementiert:* **echtes On-Device-VLM im Browser** – `web/js/vlm_webgpu.js`
   lädt `onnx-community/FastVLM-0.5B-ONNX` über **transformers.js 3.7.5**
   (dieselbe Bibliothek wie Apples offizielles `apple/fastvlm-webgpu`-Demo) und
   inferiert auf **WebGPU** (Fallback WASM): `AutoProcessor` + `AutoModelForImageTextToText`
   mit dtype-Presets (`compact` ≈ 676 MB für Handys, `balanced`, `apple`, `fp16`),
   Chat-Template mit `<image>`, greedy + `TextStreamer` → Satz-Queue → System-TTS.
   Bildquelle ist derselbe Grab-Canvas wie im Server-Modus (exakt die angezeigten
   Frames). Einstellungen → Engine: `auto | server | ondevice | demo`; ohne Server
   fragt die App einmalig vor dem ~0,7-GB-Download (danach Browser-Cache).
   Verifiziert: Processor-/Template-/Input-Vertrag per `tests/ondevice_contract.mjs`
   (`LlavaProcessor`, `pixel_values [1,3,1024,1024]`); die Volllast-Inferenz braucht
   echtes Geräte-GPU/RAM (Sandbox mit 1 GB RAM oomt beim ORT-CPU-Laden).
4. *Nativ:* Apples offizielles Repo `apple-aiml-research/ml-fastvlm` enthält
   CoreML-Exporte/iOS-Beispiel – der Weg für eine native iOS-App mit
   Neural-Engine-Latenz.

## 2 · Architektur

```
┌──────────────────────────────  Browser (Handy, Chrome/Safari)  ──────────────────────────────┐
│                                                                                              │
│  getUserMedia ──▶ <video> ──▶ Cover-Crop (+Spiegelung) ──▶ Grab-Canvas ──▶ JPEG (toBlob)     │
│                                                            │                                 │
│                              ┌─────────────────────────────┴───────────────┐                 │
│                              ▼                                             ▼                 │
│                    WebSocket (binär, 1:1 Bytes)              createImageBitmap               │
│                              │                                             ▼                 │
│                              │                               <canvas> untere Bildschirmhälfte │
│                              │                               (= exakt das gesendete JPEG)     │
│  Mikro ──▶ AudioWorklet (16 kHz, 512-Frame) ──▶ Silero-VAD (ONNX/wasm)  ──┐                   │
│                                                     │ DSP-Fallback       │                   │
│                                                     ▼                    ▼                   │
│                                          Hysterese-State-Machine ──▶ Turn-Taking/Barge-in    │
│  webkitSpeechRecognition (System-ASR) ◀── halb-duplex gesteuert ──┘                           │
│  speechSynthesis (System-TTS) ◀── Satz-Queue aus dem Token-Stream                            │
└──────────────────────────────────────────┬───────────────────────────────────────────────────┘
                                           │ WebSocket  /ws/call
┌──────────────────────────────────────────▼───────────────────────────────────────────────────┐
│  FastAPI-Server                                                                              │
│   session.py   Turn-State-Machine, StreamBridge-Loops                                        │
│   activation.py  „Soll sie JETZT von selbst reden?“ (entkoppeltes Aktivierungsmodell)        │
│   memory.py      Round-decayed Compression: Volltext → Notiz → Zusammenfassung               │
│   engine.py      FastVLM: mlx-vlm │ transformers-Remote-Code │ mock                           │
│   persona.py     Charakter-Prompts + Ausgabe-Sanitizing (kein Markdown, keine Regieanweisung)│
│   llm.py         optionales externes Dialog-LLM (OpenAI-kompatibel, SSE)                     │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 2.1 „Exakt die Frames, die FastVLM sieht“

Die untere Hälfte des Anrufs ist **kein** roher Kamerastream:

1. Das Kamerabild wird per *Cover-Crop* auf das Zielformat gezeichnet
   (Default **quadratisch**, 768 px Kantenlänge).
2. Daraus entsteht ein JPEG; **diese Bytes** gehen binär über den WebSocket.
3. Die Vorschau malt das **zurück de-kodierte JPEG** (`createImageBitmap`) auf
   ein Canvas, dessen Backing-Store exakt die Aufnahme-Auflösung hat –
   1 Canvas-Pixel = 1 Modell-Pixel, inklusive Kompressionsartefakten.
4. FastVLMs eigener Preprocessor (CLIPImageProcessor des Modells: kürzeste
   Kante auf 1024 skalieren, quadratischer Center-Crop) ändert an quadratischen
   Frames nur noch die Skalierung, nicht den Ausschnitt. Was du siehst, ist
   also genau das, was im Modell ankommt.

Der HUD oben links im Bild zeigt Frame-Nummer, Auflösung, JPEG-Größe und die
effektive Frame-Rate Richtung Modell. Optional gibt es den Komfortmodus
„flüssiges Kamerabild“, der jeden gesendeten Frame rot markiert – Standard ist
der exakte Modus. Doppel-Tipp auf das Bild wechselt Front-/Hauptkamera,
„Pause“ friert die Zufuhr ein.

### 2.2b Kamera & ehrliches Audio-Feedback

* **Rückkamera ist Standard** (`cameraFacing: environment`), Front auf Wunsch –
  Wechsel über den Button oben rechts im Videofeld (oder Settings).
* **Taschenlampe**: eigener Button, sichtbar nur wenn der Kameratrack `torch`
  kann (Android-Rückkameras); geschaltet per `applyConstraints({advanced:[{torch}]})`.
* **Spiegelung `auto`**: Front = Selfie-Spiegel, Rück = ungespiegelt (das Modell
  kann Text lesen). Jede Spiegelung gilt identisch für Aufnahme *und* Vorschau,
  damit die untere Hälfte weiterhin exakt die Modell-Frames zeigt.
* **Kein Fake im Visualizer**: Die Pegel sind echte Signale –
  *DU*: RMS-Verlauf direkt aus dem VAD-AudioWorklet (10-Sample-Historie).
  *SIE*: Hüllkurve aus den **Word-Boundary-Events** der System-TTS
  (echte Wort-Zeitstempel) plus realem Satzfortschrittsbalken
  (`charIndex`/Dauer). Keine CSS-Zufallsanimation; ohne Sprach-Events
  bleiben die Balken ehrlich flach.

### 2.2c Damit sie dich zuverlässig hört

* **ASR-Health-Watchdog**: liefert die System-Spracherkennung 25 s lang nichts,
  obwohl sie laufen soll, wird sie hart neu aufgebaut.
* **Blind-Detektion**: hört die VAD dich sprechen, aber die ASR liefert >12 s
  kein Final, gibt es erst einen Chip-Warnhinweis, dann automatisch das
  Tipp-Fallback samt Toast – statt stiller Funkstille.
* **Barge-in über zwei Kanäle**: VAD-Onset (Schwelle ×1,25, 5 Frames ≈ 160 ms
  während sie spricht) **und** Mikro-Pegel-Spike (>6× adaptiver Noisefloor über
  180 ms). Beide brechen ihre Stimme und die laufende Generierung ab.

### 2.2 Hands-free: VAD + System-Speech

* **AudioWorklet** resampelt das Mikro auf 16 kHz und liefert 32-ms-Frames.
* **Silero VAD** (vendort: `web/vendor/silero_vad.onnx` + onnxruntime-web/wasm,
  ~14 MB, per Service-Worker gecacht) liefert die Sprachwahrscheinlichkeit;
  schlägt das Laden fehl, greift automatisch eine **DSP-VAD** (adaptiver
  Noise-Floor, SNR, Nulldurchgangsrate) – gleicher Hysterese-Automat:
  Onset-Schwelle, Hang-over ~620 ms, Mindest-/Maximaldauer.
* **Turn-Taking:** Spracherkennungs-Finals werden gepuffert; spricht der VAD
  „Stille“, wird nach ~0,6 s ein kompletter Redebeitrag an den Server geschickt.
* **Barge-in:** Sprichst du, während sie redet, wird ihre Ausgabe sofort
  abgebrochen (`speechSynthesis.cancel()` + `{"t":"interrupt"}` an den Server,
  der die Token-Generierung per `threading.Event`/StoppingCriteria stoppt).
  Damit der Lautsprecher des Handys sie nicht selbst unterbricht, werden
  Schwellen *und* Onset-Dauer während ihrer Sprache angehoben (1,55× / 224 ms)
  – zusätzlich hilft Echo-Cancellation des Browsers.
* **Halbduplex** (Standard): Die Spracherkennung pausiert, während sie spricht,
  und startet danach neu – verhindert Selbstzuhören auf lauten Geräten.
* **System-TTS:** `speechSynthesis` mit automatischer Wahl einer weiblichen
  Systemstimme passend zur Persona (de-DE/de-AT/…), Satz-Queue, damit die
  Stimme schon loslegt, während das Modell noch tokenisiert; Workarounds für
  iOS-Gesture-Unlock und Android-`onend`-Aussetzer. Fehlt die
  Spracherkennung (z. B. Desktop ohne Google-Dienst), erscheint automatisch
  ein Tippfeld.

### 2.3 StreamBridge-Ideen (arXiv:2505.05467)

Apples Paper *„StreamBridge: Turning Your Offline Video-LLM into a Proactive
Streaming Assistant“* liefert die beiden Mechanismen, die aus einem Q&A-Bot
eine Gesprächspartnerin machen. Die offiziellen Gewichte sind aus
Urheberrechtsgründen nicht veröffentlicht (und das Paper baut auf
Video-LLMs, nicht auf FastVLM) – deshalb sind hier die **Konzepte** umgesetzt:

1. **Memory-Buffer mit Round-decayed Compression** (`server/memory.py`):
   Jede Gesprächsrunde verliert mit dem Alter an Detailtiefe –
   *aktuell*: live Frame + Volltext · *jung*: Transkript + Szenenotiz ·
   *älter*: nur Notiz · *sehr alt*: läuft in eine rollende Zusammenfassung.
   So bleibt FastVLM-0.5B (8k Kontext) auch nach 20 Minuten im Rahmen.
2. **Entkoppeltes Aktivierungsmodell** (`server/activation.py`): ein leichter
   Scorer, der unabhängig vom Dialogmodell auf dem Frame-Strom läuft und
   entscheidet, *ob und wann* sie von selbst spricht. Merkmale: Bild-Neuheit
   ggü. dem letzten Anker-Frame, Bewegungsenergie (unterdrückt Dauerrauschen),
   Gesprächspause, Sprechzustände beider Seiten. Modi: `heuristic` (nur CV),
   `vlm` (FastVLM-Ein-Wort-Probe „JA/NEIN“ als gelerntes Äquivalent),
   `hybrid` (CV gated die teure Probe) – plus Schwelle, Hysterese und Cooldown
   gegen Gequassel. Feuert sie, bekommt das Dialogmodell ein
   Initiative-Direktiv statt einer Nutzeräußerung.

---

## 3 · Die Charaktere

Drei Frauen, dieselbe Situation: Random-Video-Chat, **ihre Kamera ist kaputt
oder bewusst aus**, dein Video läuft. Sie wissen nichts voneinander, duzen,
haben Meinungen, Lebensdetails (WG, Katze, Job, Schwester in Rotterdam …),
stellen Gegenfragen, schweifen ab, lachen, widersprechen – und sie **reden von
selbst**, wenn in deinem Bild etwas passiert oder eine Pause entsteht.

Der System-Prompt (`server/persona.py`) verbietet ausdrücklich
Assistenten-Floskeln, das Zugeben eine KI zu sein, Technik-Auskünfte sowie
erotische/sexualisierte Inhalte, Treffen oder das Erfragen privater Daten;
bei Selbst- oder Fremdgefährdung bricht die Figur kurz und warm aus der Rolle
und nennt Hilfsnummern. Die Ausgabe wird serverseitig saniert (kein Markdown,
keine Regieanweisungen, keine Emojis, keine Rollen-Präfixe, Zahlen als Wörter),
weil sie vorgelesen wird.

---

## 4 · WebSocket-Protokoll (`/ws/call`)

Client → Server: `hello` · `frame_meta` + *binär JPEG* · `vad{speaking}` ·
`say{text,final}` · `interrupt` · `probe` · `note` · `cfg` · `speaking` · `bye`

Server → Client: `welcome` · `ready` · `status` · `delta` · `sentence` (→ TTS) ·
`turn_end` · `cancelled` · `activation` · `probe_vote` · `scene` · `ttft_ms` ·
`error` · `keepalive`

REST: `GET /api/status`, `POST /api/warm`, `POST /api/text` (Debug-Einzelshot),
`GET /api/debug/scene` (Analyse des letzten Frames).

---

## 4b · Repo & GitHub Pages

Repo: <https://github.com/lamachen/chorus> · Pages (Quelle `main:/`):
<https://lamachen.github.io/chorus/> (Launcher) bzw. `.../web/index.html` (App).

```bash
tools/publish.sh "Kurze Notiz was sich geändert hat"
#   → schreibt web/build_info.json, committet, pusht; Pages baut automatisch
```

Jede Antwort, die etwas am Code ändert, nennt den Direktlink plus Commit-Sha,
sobald sich der Pages-Stand aktualisiert hat. **Credentials liegen niemals im
Repo**: Der Push nutzt einen Credential-Store außerhalb des Arbeitsverzeichnisses
(`~/.git-credentials`, chmod 600, von Snapshots ausgeschlossen); ein im Chat
geteiltes Token gilt als kompromittiert und sollte nach der Session rotiert
werden.

## 5 · Tests

```bash
pip install playwright && playwright install chromium
FASTVLM_BACKEND=mock python -m uvicorn server.main:app --port 8100 &
python tests/browser_smoke.py      # fährt die UI mit Fake-Kamera/-Mikro, schreibt shots/

# On-Device-Vertrag (Tokenizer/Chat-Template/Processor) ohne Gewichte prüfen:
npm i @huggingface/transformers@3.7.5 && node tests/ondevice_contract.mjs
```

Der Smoke-Test startet einen echten Anruf, injiziert einen Redebeitrag,
erzwingt proaktives Sprechen und prüft Konsolenfehler (aktuell: 0).

---

## 6 · Fehlersuche

| Symptom | Ursache / Lösung |
|---|---|
| Kamera/Mikro bleiben schwarz bzw. verweigert | Seite nicht im secure context → `./run.sh` (HTTPS) nutzen, Zertifikat einmal akzeptieren; Handy & Rechner im selben WLAN |
| „VAD dsp“ statt „VAD silero“ | onnxruntime-web konnte nicht laden (z. B. alte Browser ohne SIMD/WASM) – die DSP-VAD arbeitet automatisch weiter |
| Keine Spracherkennung | iOS-Safari ≥ 14.5 / Chrome nötig; sonst erscheint das Tippfeld |
| Sie hört sich selbst und bricht ab | Halbduplex einschalten oder Barge-in-Schwelle („VAD-Empfindlichkeit“) senken |
| Lange Ladezeit beim ersten Anruf | Modell-Download (0,5 B ≈ 1,6 GB); danach gecacht; `POST /api/warm` vorwärmen |
| Antworten wirken robotisch | `LLM_BACKEND=ollama` setzen (FastVLM bleibt das Auge) oder größeres FastVLM wählen |

## 7 · Dateien

```
run.sh · requirements.txt · .env.example
server/   main.py config.py engine.py session.py activation.py memory.py persona.py llm.py
web/      index.html styles.css manifest.webmanifest sw.js
web/js/   app.js vad.js vad-worklet.js speech.js vision.js ui.js
web/vendor/  silero_vad.onnx · ort/ (onnxruntime-web wasm, ~14 MB)
tests/    browser_smoke.py        shots/  Screenshots
```

**Lizenz-Hinweis:** Silero VAD (MIT) und onnxruntime-web (MIT) liegen vendort
bei; FastVLM-Gewichte unterliegen den Bedingungen von Apple auf Hugging Face.
StreamBridge-Idee: Wang et al., NeurIPS 2025, arXiv:2505.05467 (Code: Apple
Sample Code License – hier nur als Konzept nachgebaut).
