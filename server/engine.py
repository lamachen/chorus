"""FastVLM inference engines.

Three interchangeable backends behind one async streaming interface:

  mlx          Apple Silicon via `mlx-vlm`  (fastest, this is the intended target)
  transformers PyTorch + the model's own remote code (`apple/FastVLM-*`)
  mock         no weights at all: analyses the real pixels and answers with a
               small rule-based conversationalist, so the whole app (browser,
               VAD, ASR, WebSocket, TTS, proactive activation) is testable
               anywhere

The transformers path follows the recipe on the model card: render the chat
template to a *string*, split on `<image>`, tokenize both halves without special
tokens and splice in IMAGE_TOKEN_INDEX (-200), then preprocess with the vision
tower's own CLIPImageProcessor (shortest-edge resize + centre crop).
"""

from __future__ import annotations

import asyncio
import importlib.util
import io
import platform
import random
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, Iterable

from PIL import Image, ImageFilter, ImageStat

from .config import settings
from . import persona as P


# ==========================================================================
# scene analysis (used by the mock engine + debug endpoints)
# ==========================================================================

@dataclass
class SceneFacts:
    brightness: float = 0.5
    contrast: float = 0.2
    colorfulness: float = 0.2
    edges: float = 0.1
    dom_hues: tuple = ()
    warm: bool = False
    changed: float = 0.0

    def describe(self, lang: str = "de") -> str:
        de, en = lang.startswith("de"), None
        b = self.brightness
        light = ("sehr dunkel" if b < 0.18 else "dämmrig" if b < 0.34 else
                 "normal beleuchtet" if b < 0.62 else "hell" if b < 0.8 else "sehr hell, fast überstrahlt")
        if de:
            bits = [f"Das Bild ist {light}"]
            if self.colorfulness > 0.30:
                bits.append(" und ziemlich bunt")
            elif self.colorfulness < 0.12:
                bits.append(" und fast farblos, eher grau")
            hue = self.dom_hues[0] if self.dom_hues else None
            if hue:
                bits.append(f", dominierend {hue}")
            if self.edges > 0.22:
                bits.append(". Viele Kanten, sieht unruhig aus, vielleicht Regal oder Straße")
            elif self.edges < 0.07:
                bits.append(". Sehr glatte Flächen, wahrscheinlich Wand oder Himmel")
            if self.warm:
                bits.append(". Warmes Licht, eher abends oder eine Lampe")
            if self.changed > 0.12:
                bits.append(". Starke Veränderung zum letzten Bild")
            return "".join(bits) + "."
        bits = [f"The image is {light}"]
        if self.colorfulness > 0.30:
            bits.append(" and quite colorful")
        elif self.colorfulness < 0.12:
            bits.append(" and nearly monochrome")
        if self.edges > 0.22:
            bits.append(". Many edges, looks busy")
        elif self.edges < 0.07:
            bits.append(". Very smooth areas, probably a wall or sky")
        return "".join(bits) + "."


_HUE_NAMES = [
    (15, "Rot"), (40, "Orange"), (65, "Gelb"), (160, "Grün"), (200, "Türkis"),
    (250, "Blau"), (290, "Violett"), (330, "Pink"), (361, "Rot"),
]


def _hue_name(h: float) -> str:
    for limit, name in _HUE_NAMES:
        if h < limit:
            return name
    return "Rot"


def analyse(img: Image.Image, prev_small: Image.Image | None = None) -> tuple[SceneFacts, Image.Image]:
    """Cheap, numpy-free description of a frame. Returns facts + a small grey
    thumbnail that can be reused as `prev_small` for the next frame."""
    rgb = img.convert("RGB")
    small = rgb.resize((64, 64), Image.BILINEAR)
    grey = small.convert("L")
    st = ImageStat.Stat(grey)
    brightness = st.mean[0] / 255.0
    contrast = (st.stddev[0] / 128.0) if st.stddev else 0.0

    rs, gs, bs = ImageStat.Stat(small).stddev
    colorfulness = min(1.0, ((rs * rs + gs * gs + bs * bs) ** 0.5) / 255.0 * 1.6)

    edge = grey.filter(ImageFilter.FIND_EDGES)
    edges = min(1.0, ImageStat.Stat(edge).mean[0] / 90.0)

    pal = small.quantize(colors=5, method=Image.MEDIANCUT)
    dom: list[tuple[int, str]] = []
    counts = pal.histogram()
    palette = pal.getpalette() or []
    for i, c in sorted(enumerate(counts), key=lambda kv: -kv[1])[:3]:
        if c <= 0:
            continue
        r, g, b = palette[i * 3: i * 3 + 3]
        mx, mn = max(r, g, b), min(r, g, b)
        if mx - mn < 24:
            name = "Schwarz" if mx < 60 else ("Weiß" if mx > 200 else "Grau")
        else:
            import colorsys
            h, _, _ = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
            name = _hue_name(h * 360)
        if name not in [n for _, n in dom]:
            dom.append((c, name))
    warm = bool(palette and _warmth(small))
    changed = 0.0
    if prev_small is not None:
        from PIL import ImageChops
        d = ImageChops.difference(grey, prev_small.convert("L").resize(grey.size))
        changed = min(1.0, ImageStat.Stat(d).mean[0] / 60.0)
    facts = SceneFacts(
        brightness=brightness, contrast=contrast, colorfulness=colorfulness,
        edges=edges, dom_hues=tuple(n for _, n in dom[:2]), warm=warm, changed=changed,
    )
    return facts, grey


def _warmth(small: Image.Image) -> float:
    rs, gs, bs = ImageStat.Stat(small).mean
    return (rs - bs) / 255.0 > 0.06


# ==========================================================================
# engine interface
# ==========================================================================

class Engine(ABC):
    backend = "abstract"
    model_id = "-"

    def __init__(self) -> None:
        self.ready = False
        self.error: str | None = None
        self.load_seconds = 0.0

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def _generate(
        self, system: str, history: list[tuple[str, str]], user_text: str,
        image: Image.Image | None, max_tokens: int, temperature: float, top_p: float,
        cancel: "threading.Event | None" = None,
    ) -> Iterable[str]: ...

    def info(self) -> dict:
        return {
            "backend": self.backend, "model": self.model_id, "ready": self.ready,
            "error": self.error, "load_seconds": round(self.load_seconds, 2),
        }


# --------------------------------------------------------------------------
# PyTorch / transformers (remote code) backend
# --------------------------------------------------------------------------

class TransformersEngine(Engine):
    backend = "transformers"

    def __init__(self, model_id: str) -> None:
        super().__init__()
        self.model_id = model_id
        self.tok = None
        self.model = None
        self.image_processor = None
        self.device = None
        self.dtype = None

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        t0 = time.time()
        dev = settings.device
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        dt = settings.dtype
        if dt == "auto":
            dt = "float16" if dev == "cuda" else ("bfloat16" if dev == "mps" else "float32")
        torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dt]

        self.tok = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            device_map="auto" if dev == "cuda" else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        if dev != "cuda":
            self.model = self.model.to(dev)
        self.model.eval()
        vt = self.model.get_vision_tower()
        if not getattr(vt, "is_loaded", False):
            vt.load_model()
        self.image_processor = vt.image_processor
        self.device = next(self.model.parameters()).device
        self.dtype = next(self.model.parameters()).dtype
        self.torch = torch
        self.ready = True
        self.load_seconds = time.time() - t0

    def _build_inputs(self, system, history, user_text, image):
        torch = self.torch
        content = ("<image>\n" + user_text) if image is not None else user_text
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        for role, text in history:
            messages.append({"role": role, "content": text})
        messages.append({"role": "user", "content": content})

        rendered = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        if image is not None and "<image>" in rendered:
            pre, post = rendered.split("<image>", 1)
            pre_ids = self.tok(pre, return_tensors="pt", add_special_tokens=False).input_ids
            post_ids = self.tok(post, return_tensors="pt", add_special_tokens=False).input_ids
            img_tok = torch.tensor([[settings.image_token_index]], dtype=pre_ids.dtype)
            input_ids = torch.cat([pre_ids, img_tok, post_ids], dim=1)
        else:
            input_ids = self.tok(rendered, return_tensors="pt", add_special_tokens=False).input_ids
        input_ids = input_ids.to(self.device)
        attention_mask = torch.ones_like(input_ids)

        pixel_values = None
        if image is not None:
            px = self.image_processor(images=image, return_tensors="pt")["pixel_values"]
            pixel_values = px.to(self.device, dtype=self.dtype)
        return input_ids, attention_mask, pixel_values

    def _generate(self, system, history, user_text, image, max_tokens, temperature, top_p, cancel=None):
        from transformers import TextIteratorStreamer

        input_ids, attention_mask, pixel_values = self._build_inputs(system, history, user_text, image)
        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)
        kwargs = dict(
            inputs=input_ids, attention_mask=attention_mask, max_new_tokens=max_tokens,
            do_sample=temperature > 0, streamer=streamer,
            pad_token_id=self.tok.eos_token_id,
        )
        if temperature > 0:
            kwargs.update(temperature=float(temperature), top_p=float(top_p))
        if settings.repetition_penalty and settings.repetition_penalty != 1.0:
            kwargs["repetition_penalty"] = float(settings.repetition_penalty)
        if pixel_values is not None:
            kwargs["images"] = pixel_values
        if cancel is not None:
            from transformers import StoppingCriteria, StoppingCriteriaList

            class _Cancel(StoppingCriteria):
                def __call__(self, input_ids, scores, **kw) -> bool:
                    return cancel.is_set()

            kwargs["stopping_criteria"] = StoppingCriteriaList([_Cancel()])

        with self.torch.inference_mode():
            thread = threading.Thread(target=self.model.generate, kwargs=kwargs, daemon=True)
            thread.start()
            for piece in streamer:
                if piece:
                    yield piece
            thread.join(timeout=120)


# --------------------------------------------------------------------------
# Apple MLX backend
# --------------------------------------------------------------------------

class MlxEngine(Engine):
    backend = "mlx"

    def __init__(self, model_id: str) -> None:
        super().__init__()
        self.model_id = model_id
        self.model = None
        self.processor = None
        self.config = None

    def load(self) -> None:
        t0 = time.time()
        from mlx_vlm import load as mlx_load
        from mlx_vlm.utils import load_config

        loaded = mlx_load(self.model_id)
        self.model, self.processor = loaded[0], loaded[1]
        try:
            self.config = load_config(self.model_id)
        except Exception:
            self.config = getattr(self.model, "config", None)
        self.ready = True
        self.load_seconds = time.time() - t0

    def _prompt(self, system, history, user_text, image) -> str:
        from mlx_vlm.prompt_utils import apply_chat_template

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        for role, text in history:
            messages.append({"role": role, "content": text})
        messages.append({"role": "user", "content": user_text})
        n_img = 1 if image is not None else 0
        for attempt in ("messages", "string"):
            try:
                if attempt == "messages":
                    return apply_chat_template(self.processor, self.config, messages, num_images=n_img)
                flat = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
                return apply_chat_template(self.processor, self.config, flat, num_images=n_img)
            except TypeError:
                continue
        # last resort: render ChatML by hand (FastVLM is Qwen2-based)
        out = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        return out + "<|im_start|>assistant\n"

    def _generate(self, system, history, user_text, image, max_tokens, temperature, top_p, cancel=None):
        from mlx_vlm import stream_generate

        prompt = self._prompt(system, history, user_text, image)
        kwargs = dict(max_tokens=max_tokens, temperature=temperature, top_p=top_p)
        if image is not None:
            kwargs["image"] = [image]
        try:
            gen = stream_generate(self.model, self.processor, prompt, cancel_event=cancel, **kwargs)
        except TypeError:
            gen = stream_generate(self.model, self.processor, prompt, **kwargs)
        for chunk in gen:
            if cancel is not None and cancel.is_set():
                break
            text = getattr(chunk, "text", None)
            if text is None and isinstance(chunk, dict):
                text = chunk.get("text")
            if text:
                yield text


# --------------------------------------------------------------------------
# Mock backend (no weights) - analyses the real frame, answers by rules
# --------------------------------------------------------------------------

class MockEngine(Engine):
    backend = "mock"
    model_id = "mock-persona-engine (no weights loaded)"

    def load(self) -> None:
        t0 = time.time()
        self.ready = True
        self.load_seconds = time.time() - t0

    _DE = (" du ", "bild", "satz", "beschreib", "siehst", "was ", "gespräch", "ja oder nein", "rolle")
    _EN = (" you ", "image", "sentence", "describe", "yes or no", " the ", "role")

    def _detect_lang(self, *texts) -> str:
        t = " " + " ".join(str(x or "").lower() for x in texts) + " "
        de = sum(t.count(w) for w in self._DE)
        en = sum(t.count(w) for w in self._EN)
        return "de" if de >= en else "en"

    def _generate(self, system, history, user_text, image, max_tokens, temperature, top_p, cancel=None):
        lang = self._detect_lang(system, user_text, *[h[1] for h in history[-2:]])
        name = "Mara"
        m = __import__("re").search(r"Du bist (\w+)|You are (\w+)", system or "")
        if m:
            name = m.group(1) or m.group(2) or name
        facts, _ = analyse(image) if image is not None else (SceneFacts(), None)
        reply = self._reply(user_text, facts, name, system, lang)
        # stream it word by word with realistic pacing
        words = reply.split(" ")
        for i, w in enumerate(words):
            if cancel is not None and cancel.is_set():
                return
            time.sleep(random.uniform(0.02, 0.07))
            yield w + (" " if i < len(words) - 1 else "")

    def _reply(self, user_text, facts: SceneFacts, name, system, lang) -> str:
        t = (user_text or "").lower()
        if "ja oder nein" in t or "yes or no" in t:
            return ("JA" if (facts.changed > 0.08 or random.random() < 0.4) else "NEIN")
        if t.strip().startswith(("ein satz:", "one sentence:")) or "was ist in diesem bild" in t:
            return facts.describe(lang)
        if "zusammen" in t and "fasse" in t or t.strip().startswith(("summarise", "fasse")):
            return t[:160]
        de = lang.startswith("de")
        scene = facts.describe(lang)
        pid = "mara"
        for key in P.PERSONAS:
            if P.PERSONAS[key].name.lower() == str(name).lower():
                pid = key
        p = P.get_persona(pid)

        def ask():
            q = ([
                "Und, was machst du da gerade?", "Bist du auch so ein Nachtmensch?",
                "Wo kommst du eigentlich her?", "Was läuft bei dir so, erzähl mal.",
                "Und, guter Tag heute oder eher so naja?", "Was hast du da hinter dir, sieht interessant aus.",
            ] if de else [
                "So what are you up to?", "Are you a night person too?",
                "Where are you from?", "Tell me something about your day.",
            ])
            return random.choice(q)

        if not t.strip():
            return (f"Hm, ich seh dich nur. {scene} " + ask()) if de else (f"Hm, I can only see you. {scene} " + ask())

        table_de = [
            (("hallo", "hi ", "hey", "servus", "moin", "guten"), [
                f"Hey! Na? {scene} Sag mal was, ich seh dich die ganze Zeit.",
                "Hallo du. Bissl komisch, dass meine Kamera aus ist, oder? Aber ich seh dich.",
            ]),
            (("wie heißt du", "heißt du", "dein name", "wer bist"), [
                f"Ich bin {p.name}. Und du? Und nein, meine Kamera geht nicht, frag nicht.",
                f"{p.name}. {p.age}, aus {p.city}. Deine Kamera funktioniert zum Glück, das reicht für uns beide.",
            ]),
            (("kamera", "video aus", "bild aus", "man sieht dich nicht", "warum siehst"), [
                "Ja, ich weiß, ich weiß. " + p.camera_excuse.split(";")[0] + ". Nervt mich selbst am meisten.",
                "Meine Kamera ist halt im Eimer. Ich beschreib dir mich einfach, wenn du willst.",
            ]),
            (("was siehst du", "siehst du", "wie seh ich aus", "guck mal", "schaust du"), [
                scene + " Mehr kann ich dir ehrlich nicht sagen, so genau ist mein Blick nicht.",
                "Ich seh " + ("dich, " if True else "") + scene.lower().replace("das bild ist", "") + " Was willst du hören?",
            ]),
            (("wo bist", "wo kommst", "wo wohnst", "welche stadt"), [
                f"{p.city}. {p.facts[0]}. Und du?",
                f"Ich sitz hier in {p.city} rum. {p.facts[1]}.",
            ]),
            (("was machst du", "was geht", "wie geht", "alles gut", "wie läuft"), [
                f"Nicht viel. {p.facts[2][0].upper() + p.facts[2][1:]}. Also eigentlich genau der richtige Moment für so einen Chat.",
                f"Ganz gut. {p.facts[3][0].upper() + p.facts[3][1:]}. Bei dir? Du siehst nicht aus, als würdest du gleich schlafen.",
            ]),
            (("arbeit", "job", "studierst", "studium", "schule", "beruf"), [
                f"{p.facts[1].capitalize()}. Klingt unspektakulär, ist es auch meistens.",
                f"Also: {p.facts[1]}. Und du, was machst du so den ganzen Tag?",
            ]),
            (("musik", "hörst du", "lied", "song", "band"), [
                f"{p.facts[4].capitalize()}. Sag nicht, du hörst was Vernünftiges.",
                "Ich hab gerade die gleiche Playlist seit drei Wochen. Peinlich. Was hörst du?",
            ]),
            (("tier", "katze", "hund", "haustier", "pet"), [
                f"{p.facts[3].capitalize()}. Und ja, sie liegt gerade wieder auf meinem Arm.",
                "Tiere? Immer. Ich hab " + p.facts[3].split(",")[-1].strip() + ". Du etwa nicht?",
            ]),
            (("alt", "alter", "jahre"), [
                f"{p.age}. Und bevor du fragst: ja, ich sitze abends mit Fremden im Videochat. Läuft bei mir.",
                f"{p.age}, {p.city}. Deine Runde, sag was über dich.",
            ]),
            (("langweilig", "keine lust", "müde", "schlafen", "gute nacht"), [
                "Dann geh schlafen, ich halt dich nicht auf. …oder bleib noch fünf Minuten.",
                "Müde, aber du redest noch mit mir? Respekt. Ich auch, ehrlich.",
            ]),
            (("witz", "lach", "lustig", "spaß"), [
                "Ich bin schlecht in Witzen. Wirklich schlecht. Soll ich trotzdem?",
                "Haha, okay, du bist lustiger als ich. Notiert.",
            ]),
            (("ki ", "künstliche intelligenz", "bot", "roboter", "bist du echt", "chatgpt"), [
                "Hä? Was soll das denn jetzt. Ich sitz hier mit kaputter Kamera rum und du fragst, ob ich echt bin.",
                "Ey, ich bin zu müde für so eine Frage. Klar bin ich echt. Was bist du denn, ein Polizist?",
            ]),
            (("wetter", "regnet", "sonne", "draußen"), [
                f"Hier in {p.city} ist es grau, wie immer. {p.facts[6].capitalize()}.",
                "Keine Ahnung, ich war heute kaum draußen. Bei dir siehts auf dem Video auch nicht nach Sommer aus.",
            ]),
            (("essen", "hunger", "kochst", "kaffee", "bier"), [
                f"{p.facts[5].capitalize()}. Sag jetzt nicht, du kochst richtig.",
                "Ich hab vorhin zu spät gegessen und bin jetzt zu wach. Klassiker.",
            ]),
            (("?",), [
                "Gute Frage. Ehrlich? Keine Ahnung. Was denkst du denn?",
                "Hm. Lass mich kurz überlegen… nee, sag du erst.",
            ]),
        ]
        for keys, options in (table_de if de else []):
            if any(k in t for k in keys):
                out = random.choice(options)
                if random.random() < 0.35:
                    out += " " + ask()
                return out[:400]

        generic_de = [
            f"Hm. {scene} Sag mehr, ich hör zu.",
            "Okay, warte, das musst du mir erklären. Ich mein, ich sitz hier nur mit Ton.",
            "Echt? Das hätte ich jetzt nicht gedacht. Erzähl weiter.",
            "Haha, okay. Und was ist daraus geworden?",
            "Weißt du was, genau darüber hab ich heute auch nachgedacht. Bissl zumindest.",
            f"Krass. Also bei mir ist gerade nur {p.facts[6].split(',')[0]}. Und bei dir so?",
        ]
        generic_en = [
            f"Hm. {scene} Tell me more.",
            "Okay wait, you have to explain that. I'm just sitting here with audio.",
            "Really? I wouldn't have guessed. Go on.",
            "Haha, okay. And then what happened?",
        ]
        out = random.choice(generic_de if de else generic_en)
        if random.random() < 0.4:
            out += " " + ask()
        return out[:400]


# ==========================================================================
# manager: one model, one lock, async token stream
# ==========================================================================

def _available(spec: str) -> bool:
    try:
        return importlib.util.find_spec(spec) is not None
    except Exception:
        return False


def pick_backend(settings) -> str:
    b = (settings.backend or "auto").lower()
    if b != "auto":
        return b
    if platform.system() == "Darwin" and _available("mlx_vlm"):
        return "mlx"
    if _available("torch") and _available("transformers"):
        return "transformers"
    if _available("transformers"):
        return "transformers"
    return "mock"


def build_engine(settings) -> Engine:
    backend = pick_backend(settings)
    if backend == "mlx":
        return MlxEngine(settings.model or settings.model_mlx)
    if backend == "transformers":
        return TransformersEngine(settings.model or settings.model_pt)
    return MockEngine()


class EngineManager:
    """Loads the engine in a worker thread and serialises inference."""

    def __init__(self, settings) -> None:
        self.settings = settings
        self.engine: Engine = build_engine(settings)
        self._lock = asyncio.Lock()
        self._loading = False

    @property
    def backend(self) -> str:
        return self.engine.backend

    @property
    def ready(self) -> bool:
        return self.engine.ready

    @property
    def busy(self) -> bool:
        return self._lock.locked() or self._loading

    def info(self) -> dict:
        d = self.engine.info()
        d["loading"] = self._loading
        d["device_hint"] = platform.system()
        return d

    async def start(self) -> None:
        if self.engine.ready or self._loading:
            return
        self._loading = True
        loop = asyncio.get_running_loop()

        def work():
            try:
                self.engine.load()
            except Exception as exc:  # pragma: no cover - depends on hardware
                self.engine.error = f"{type(exc).__name__}: {exc}"
                if not isinstance(self.engine, MockEngine):
                    self.engine = MockEngine()
                    try:
                        self.engine.load()
                    except Exception as exc2:
                        self.engine.error = str(exc2)
            finally:
                self._loading = False

        await loop.run_in_executor(None, work)

    async def stream(
        self,
        *,
        system: str,
        history: list[tuple[str, str]],
        user_text: str,
        image: Image.Image | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        cancel: "threading.Event | None" = None,
    ) -> AsyncIterator[str]:
        """Yield tokens as they are produced. Blocking work stays in a thread."""
        if not self.engine.ready:
            await self.start()
        max_tokens = max_tokens or settings.max_tokens
        temperature = settings.temperature if temperature is None else temperature
        top_p = settings.top_p if top_p is None else top_p

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        SENTINEL = object()

        async with self._lock:
            def worker():
                try:
                    for piece in self.engine._generate(
                        system, history, user_text, image, max_tokens, temperature, top_p, cancel
                    ):
                        if cancel is not None and cancel.is_set():
                            break
                        loop.call_soon_threadsafe(queue.put_nowait, piece)
                except Exception as exc:
                    loop.call_soon_threadsafe(queue.put_nowait, exc)
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, SENTINEL)

            fut = loop.run_in_executor(None, worker)
            try:
                while True:
                    item = await queue.get()
                    if item is SENTINEL:
                        break
                    if isinstance(item, Exception):
                        raise item
                    yield item
            finally:
                await fut


def decode_frame(data: bytes, max_bytes: int | None = None) -> Image.Image:
    max_bytes = max_bytes or settings.max_frame_bytes
    if len(data) > max_bytes:
        raise ValueError(f"frame too large: {len(data)} > {max_bytes}")
    img = Image.open(io.BytesIO(data))
    img.load()
    return img.convert("RGB")
