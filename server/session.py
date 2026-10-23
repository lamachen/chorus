"""One live call = one CallSession.

Responsibilities
----------------
* keep the newest frame that the browser sent (these are *exactly* the frames
  shown in the lower half of the call UI)
* run the StreamBridge-style loops: activation (should she speak?) + memory
  compression (round decay)
* produce spoken turns: stream tokens -> split into sentences -> sanitise ->
  push to the client, which feeds them to the system TTS queue
* handle barge-in (client `interrupt` cancels generation mid-token)
"""

from __future__ import annotations

import asyncio
import re
import threading
import time
from typing import Any

from . import persona as P
from .activation import ActivationModel
from .config import settings
from .engine import EngineManager, analyse, decode_frame
from .llm import TextLLM
from .memory import StreamMemory


class SentenceBuffer:
    """Turns a token stream into speakable chunks as early as possible."""

    def __init__(self, min_len: int = 14, max_len: int = 78) -> None:
        self.buf = ""
        self.min_len = min_len
        self.max_len = max_len

    def push(self, text: str) -> list[str]:
        self.buf += text
        out: list[str] = []
        while True:
            m = re.search(r"[.!?…]+[\s\"')\]]*", self.buf)
            if not m:
                break
            head = self.buf[: m.end()].strip()
            if len(head) < self.min_len and m.end() < len(self.buf):
                # too short to be a real sentence end (e.g. "Hm.") -> keep going
                rest = self.buf[m.end():]
                if not rest.strip():
                    break
                nxt = re.search(r"[.!?…]+", rest)
                if not nxt or len(head) + nxt.end() < self.min_len:
                    break
            out.append(head)
            self.buf = self.buf[m.end():]
        if not out and len(self.buf) > self.max_len:
            cut = self.buf.rfind(",", 0, self.max_len)
            if cut > self.min_len:
                out.append(self.buf[:cut].strip())
                self.buf = self.buf[cut + 1:]
        return out

    def flush(self) -> list[str]:
        tail = self.buf.strip()
        self.buf = ""
        return [tail] if tail else []


class CallSession:
    def __init__(self, ws, engines: EngineManager, llm: TextLLM) -> None:
        self.ws = ws
        self.engines = engines
        self.llm = llm
        self.persona = P.get_persona(P.DEFAULT_PERSONA)
        self.lang = "de"
        self.memory = StreamMemory(
            full_rounds=settings.memory_full_rounds,
            mid_rounds=settings.memory_mid_rounds,
            summary_chars=settings.memory_summary_chars,
            budget_chars=settings.memory_budget_chars,
        )
        self.activation = ActivationModel(settings)

        # vision state
        self.frame = None                      # PIL image, newest frame from the client
        self.frame_id = -1
        self.frame_ts = 0.0
        self.frame_bytes = 0
        self.frames_seen = 0
        self._small_prev = None
        self.scene_note = ""
        self.scene_note_t = 0.0

        # dialogue state
        self.turn_task: asyncio.Task | None = None
        self.cancel_evt = threading.Event()
        self.user_speaking = False
        self.she_speaking = False
        self.last_exchange = time.monotonic()
        self.turns = 0
        self.latencies: list[float] = []
        self.connected_at = time.time()
        self.closed = False
        self._pending_meta: dict = {}
        self.pending_notes: list[str] = []
        self.turbo = False
        self._scene_lock = asyncio.Lock()
        self._last_scene_run = 0.0
        self._last_compression = 0.0
        self.greeted = False

    # ------------------------------------------------------------------
    # transport helpers
    # ------------------------------------------------------------------
    async def send(self, payload: dict) -> None:
        if self.closed:
            return
        try:
            await self.ws.send_json(payload)
        except Exception:
            self.closed = True

    async def log(self, msg: str, level: str = "info") -> None:
        await self.send({"t": "log", "level": level, "msg": msg})

    # ------------------------------------------------------------------
    # main entry point for websocket messages
    # ------------------------------------------------------------------
    async def handle(self, message: Any) -> None:
        if isinstance(message, (bytes, bytearray)):
            await self._on_frame(bytes(message))
            return
        try:
            msg = __import__("json").loads(message)
        except Exception:
            return
        t = msg.get("t")
        if t == "hello":
            await self._on_hello(msg)
        elif t == "frame_meta":
            self._pending_meta = msg
        elif t == "vad":
            self._on_vad(msg)
        elif t == "say":
            if msg.get("final"):
                await self.user_turn(msg.get("text", ""))
        elif t == "interrupt":
            await self.interrupt(msg.get("reason", "barge-in"))
        elif t == "probe":
            await self.user_turn(msg.get("text", ""), force_probe=True)
        elif t == "cfg":
            self._on_cfg(msg)
        elif t == "note":
            note = str(msg.get("text", ""))[:200]
            if note:
                self.pending_notes.append(note)
        elif t == "speaking":
            self.she_speaking = bool(msg.get("value"))
            if not self.she_speaking:
                self.last_exchange = time.monotonic()
        elif t == "ping":
            await self.send({"t": "pong", "ts": msg.get("ts")})

    # ------------------------------------------------------------------
    async def _on_hello(self, msg: dict) -> None:
        self.persona = P.get_persona(msg.get("persona"))
        self.lang = (msg.get("lang") or "de")[:5]
        if msg.get("proactive") is not None or msg.get("activation_mode"):
            self.activation.set_mode(msg.get("activation_mode"), msg.get("proactive"))
        if msg.get("activation_threshold"):
            self.activation.threshold = float(msg["activation_threshold"])
        await self.send({
            "t": "ready",
            "engine": self.engines.info(),
            "llm": self.llm.info(),
            "persona": {
                "id": self.persona.pid, "name": self.persona.name, "age": self.persona.age,
                "city": self.persona.city, "hue": self.persona.hue,
                "voice": {"lang": self.persona.voice_lang, "pitch": self.persona.voice_pitch,
                          "rate": self.persona.voice_rate, "names": self.persona.voice_names},
                "greetings": self.persona.greetings,
            },
            "personas": [
                {"id": p.pid, "name": p.name, "age": p.age, "city": p.city, "tagline": p.tagline, "hue": p.hue}
                for p in P.PERSONAS.values()
            ],
            "settings": settings.public_dict(),
            "clock": time.time(),
        })
        if not self.greeted and msg.get("greet", True):
            self.greeted = True
            asyncio.create_task(self._greet())

    async def _greet(self) -> None:
        try:
            await self._greet_inner()
        except Exception as exc:
            await self.send({"t": "error", "message": f"greeting failed: {exc}", "fatal": False})

    async def _greet_inner(self) -> None:
        await asyncio.sleep(0.4)
        line = P.greeting_for(self.persona)
        await self.send({"t": "status", "state": "speaking", "reason": "greeting"})
        r = self.memory.open_round(self.frame_id, self.scene_note)
        self.memory.set_assistant(line)
        await self._emit(line, reason="greeting")
        self.last_exchange = time.monotonic()
        _ = r

    def _on_vad(self, msg: dict) -> None:
        speaking = bool(msg.get("speaking"))
        if speaking and not self.user_speaking:
            pass
        if not speaking and self.user_speaking:
            self.last_exchange = time.monotonic()
        self.user_speaking = speaking
        if speaking and msg.get("barge_in"):
            asyncio.create_task(self.interrupt("barge-in"))

    def _on_cfg(self, msg: dict) -> None:
        if msg.get("lang"):
            self.lang = str(msg["lang"])[:5]
        if msg.get("persona"):
            self.persona = P.get_persona(msg["persona"])
        if msg.get("proactive") is not None or msg.get("activation_mode"):
            self.activation.set_mode(msg.get("activation_mode"), msg.get("proactive"))
        for key, attr in (("activation_threshold", "threshold"), ("activation_cooldown", "cooldown")):
            if msg.get(key) is not None:
                setattr(self.activation, attr, float(msg[key]))
        if msg.get("turbo") is not None:
            self.turbo = bool(msg["turbo"])

    # ------------------------------------------------------------------
    # vision
    # ------------------------------------------------------------------
    async def _on_frame(self, data: bytes) -> None:
        meta = self._pending_meta
        self._pending_meta = {}
        try:
            img = decode_frame(data, settings.max_frame_bytes)
        except Exception as exc:
            await self.send({"t": "error", "message": f"frame rejected: {exc}", "fatal": False})
            return
        self.frame = img
        self.frames_seen += 1
        self.frame_id = int(meta.get("id", self.frames_seen))
        self.frame_ts = float(meta.get("ts", time.time()))
        self.frame_bytes = len(data)
        if settings.log_frames:
            await self.send({"t": "ack_frame", "id": self.frame_id, "bytes": len(data),
                             "w": img.width, "h": img.height})
        # cheap per-frame activation features (no inference)
        if self.activation.enabled:
            self.activation.observe(
                img, now=time.monotonic(), user_speaking=self.user_speaking,
                she_speaking=self.she_speaking, last_exchange=self.last_exchange,
                allow_fire=False,
            )
        asyncio.create_task(self._background_work())

    async def _background_work(self) -> None:
        """Scene notes + memory compression, only while the engine is idle."""
        now = time.monotonic()
        if self.frame is None or self.turn_task is not None:
            return
        if self.engines.busy:
            return
        if self.llm.enabled and (now - self._last_scene_run) > 3.0:
            self._last_scene_run = now
            asyncio.create_task(self._refresh_scene_note())
            return
        if (now - self._last_compression) > 6.0:
            r = self.memory.needs_compression()
            if r is not None and not self.engines.busy:
                self._last_compression = now
                asyncio.create_task(self._compress_round(r))

    async def _refresh_scene_note(self) -> None:
        """FastVLM as always-on eyes: one short caption of the current frame."""
        if self.frame is None or self.engines.busy:
            return
        async with self._scene_lock:
            prompt = P.SCENE_NOTE_DE if self.lang.startswith("de") else P.SCENE_NOTE_EN
            try:
                chunks = []
                async for piece in self.engines.stream(
                    system="Du beschreibst in einem Satz, was in einem Standbild zu sehen ist.",
                    history=[], user_text=prompt, image=self.frame,
                    max_tokens=settings.scene_note_max_tokens, temperature=0.2,
                ):
                    chunks.append(piece)
                    if sum(len(c) for c in chunks) > 240:
                        break
                note = P.sanitize_reply("".join(chunks), self.persona.name, max_chars=200)
                if note:
                    self.scene_note = note
                    self.scene_note_t = time.time()
                    cur = self.memory.current()
                    if cur is not None:
                        self.memory.set_scene(note)
                    await self.send({"t": "scene", "text": note, "frame_id": self.frame_id})
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await self.log(f"scene note failed: {exc}", "warn")

    async def _compress_round(self, r) -> None:
        """Round-decayed compression: fold an old round into one short line."""
        text = " ".join(x for x in (r.scene, r.user, r.assistant) if x)
        if len(text) < 90:
            r.scene_brief = r.scene_brief or text[:90]
            return
        if self.engines.busy:
            r.scene_brief = text[:90]
            return
        try:
            prompt = (P.COMPRESS_DE if self.lang.startswith("de") else P.COMPRESS_EN) + text[:600]
            chunks = []
            async for piece in self.engines.stream(
                system="", history=[], user_text=prompt, image=None,
                max_tokens=48, temperature=0.2,
            ):
                chunks.append(piece)
            brief = P.sanitize_reply("".join(chunks), self.persona.name, max_chars=90)
            self.memory.apply_compression(r, brief or text[:90])
        except Exception:
            self.memory.apply_compression(r, text[:90])

    # ------------------------------------------------------------------
    # proactive loop (StreamBridge activation model)
    # ------------------------------------------------------------------
    async def proactive_loop(self) -> None:
        while not self.closed:
            try:
                await asyncio.sleep(max(0.6, settings.activation_interval))
                if not self.activation.enabled or self.frame is None:
                    continue
                if self.turn_task is not None or self.user_speaking or self.she_speaking:
                    continue
                now = time.monotonic()
                if self.activation.wants_probe(now) and not self.engines.busy:
                    await self._probe_activation()
                    self.activation.mark_probed(now)
                st = self.activation.observe(
                    self.frame, now=now, user_speaking=self.user_speaking,
                    she_speaking=self.she_speaking, last_exchange=self.last_exchange,
                )
                await self.send({"t": "activation", "score": round(st.score, 3),
                                 "speak": st.speak, "reason": st.reason, "parts": st.parts,
                                 "mode": self.activation.mode})
                if st.speak:
                    await self.proactive_turn()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await self.log(f"proactive loop: {exc}", "warn")
                await asyncio.sleep(2)

    async def _probe_activation(self) -> None:
        prompt = P.ACTIVATION_PROMPT_DE if self.lang.startswith("de") else P.ACTIVATION_PROMPT_EN
        try:
            chunks = []
            async for piece in self.engines.stream(
                system="", history=[], user_text=prompt, image=self.frame,
                max_tokens=6, temperature=0.0,
            ):
                chunks.append(piece)
                if sum(len(c) for c in chunks) > 12:
                    break
            ans = "".join(chunks).strip().lower()
            yes = ans.startswith(("ja", "yes", "j", "y")) or "ja" in ans[:8] or "yes" in ans[:8]
            self.activation.set_probe_vote(yes)
            await self.send({"t": "probe_vote", "yes": yes, "raw": ans[:16]})
        except Exception as exc:
            await self.log(f"activation probe failed: {exc}", "warn")

    # ------------------------------------------------------------------
    # turns
    # ------------------------------------------------------------------
    async def interrupt(self, reason: str = "barge-in") -> None:
        self.cancel_evt.set()
        task, self.turn_task = self.turn_task, None
        if task and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        await self.send({"t": "cancelled", "reason": reason})
        await self.send({"t": "status", "state": "idle", "reason": reason})

    async def user_turn(self, text: str, force_probe: bool = False) -> None:
        text = (text or "").strip()
        if not text and not force_probe:
            return
        if self.turn_task is not None:
            await self.interrupt("new-turn")
        self.turns += 1
        self.cancel_evt = threading.Event()
        task = asyncio.create_task(self._run_turn("user", text, force_probe=force_probe))
        self.turn_task = task
        task.add_done_callback(lambda _t: setattr(self, "turn_task", None))
        await task

    async def proactive_turn(self) -> None:
        if self.turn_task is not None:
            return
        self.turns += 1
        self.cancel_evt = threading.Event()
        task = asyncio.create_task(self._run_turn("proactive", ""))
        self.turn_task = task
        task.add_done_callback(lambda _t: setattr(self, "turn_task", None))
        await task

    async def _run_turn(self, reason: str, user_text: str, force_probe: bool = False) -> None:
        t0 = time.monotonic()
        p = self.persona
        use_llm = self.llm.enabled and not force_probe
        history = self._history()
        memory_text = self.memory.render()
        scene_text = self.scene_note if (time.time() - self.scene_note_t) < 25 else ""

        system = P.build_system_prompt(
            p, self.lang, memory_text=memory_text, scene_text=scene_text,
            proactive=self.activation.enabled, rich=use_llm or self.engines.backend == "mock",
        )

        if reason == "proactive":
            directive = (
                "(Stille. Niemand sagt etwas. Du siehst sein Video und sagst von selbst etwas Kurzes, "
                "Alltägliches - einen Gedanken, eine Beobachtung, eine Frage. Nichts wiederholen, "
                "was du schon gesagt hast.)"
                if self.lang.startswith("de") else
                "(Silence. Nobody speaks. You look at his video and say something short and casual "
                "on your own - a thought, an observation, a question. Don't repeat yourself.)"
            )
            prompt_user = directive
        elif force_probe:
            prompt_user = P.PROBE_PROMPT_DE if self.lang.startswith("de") else P.PROBE_PROMPT_EN
        else:
            prompt_user = user_text
            if use_llm and scene_text:
                # text-only LLM: hand over what FastVLM currently sees
                prompt_user = f"[du siehst gerade in seinem Video: {scene_text}]\n\nER SAGT: {user_text}"

        if self.pending_notes:
            notes = " ".join(self.pending_notes)
            self.pending_notes.clear()
            prompt_user = f"[Regieanweisung aus der App, nicht laut vorlesen: {notes}]\n\n" + prompt_user

        round_obj = self.memory.open_round(self.frame_id, self.scene_note)
        if user_text:
            self.memory.set_user(user_text)

        await self.send({"t": "status", "state": "thinking", "reason": reason,
                         "frame_id": self.frame_id, "memory": self.memory.snapshot()})

        buf = SentenceBuffer()
        pieces: list[str] = []
        spoken: list[str] = []
        sent_sentences = 0
        first_token_ms = None
        image = None if use_llm else self.frame
        stream = (
            self.llm.stream(system=system, history=history, user_text=prompt_user,
                            max_tokens=settings.llm_max_tokens, temperature=settings.temperature)
            if use_llm else
            self.engines.stream(system=system, history=history, user_text=prompt_user,
                                image=image,
                                max_tokens=70 if self.turbo else settings.max_tokens,
                                temperature=0.0 if self.turbo else settings.temperature,
                                top_p=settings.top_p,
                                cancel=self.cancel_evt)
        )

        try:
            async for piece in stream:
                if self.cancel_evt.is_set():
                    break
                if first_token_ms is None:
                    first_token_ms = (time.monotonic() - t0) * 1000
                    await self.send({"t": "ttft_ms", "ms": round(first_token_ms)})
                pieces.append(piece)
                await self.send({"t": "delta", "text": piece})
                for sent in buf.push(piece):
                    clean = P.sanitize_reply(sent, p.name, max_chars=200)
                    if not clean:
                        continue
                    spoken.append(clean)
                    sent_sentences += 1
                    await self.send({"t": "sentence", "text": clean, "i": sent_sentences})
                    cap = 2 if self.turbo else 3
                    if sent_sentences >= cap and sum(len(s) for s in spoken) > 90:
                        # enough for one spoken breath; drain the rest silently
                        self.cancel_evt.set()
                        break
            for sent in buf.flush():
                clean = P.sanitize_reply(sent, p.name, max_chars=200)
                if clean:
                    spoken.append(clean)
                    sent_sentences += 1
                    await self.send({"t": "sentence", "text": clean, "i": sent_sentences})
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self.send({"t": "error", "message": f"generation failed: {exc}", "fatal": False})

        full = P.sanitize_reply(" ".join(spoken) or "".join(pieces), p.name, max_chars=420)
        if not full or P.looks_leaked(full):
            full = P.fallback(self.lang)
        self.memory.set_assistant(full, proactive=(reason == "proactive"))
        self.activation.anchor_to_current()
        self.last_exchange = time.monotonic()
        elapsed = (time.monotonic() - t0) * 1000
        self.latencies.append(elapsed)
        self.latencies = self.latencies[-20:]
        await self.send({
            "t": "turn_end", "text": full, "reason": reason, "ms": round(elapsed),
            "ttft_ms": round(first_token_ms or elapsed), "frames_seen": self.frames_seen,
            "frame_id": self.frame_id, "memory": self.memory.snapshot(),
            "avg_ms": round(sum(self.latencies) / len(self.latencies)),
        })
        await self.send({"t": "status", "state": "idle", "reason": "turn_end"})
        _ = round_obj

    async def _emit(self, text: str, reason: str = "greeting") -> None:
        """Send a ready-made line as if it were generated (greetings, fallbacks)."""
        for i, sent in enumerate(SentenceBuffer().push(text + " ") or [text], start=1):
            clean = P.sanitize_reply(sent, self.persona.name, 200)
            if clean:
                await self.send({"t": "sentence", "text": clean, "i": i})
        await self.send({"t": "turn_end", "text": text, "reason": reason, "ms": 0, "ttft_ms": 0,
                         "frames_seen": self.frames_seen, "frame_id": self.frame_id,
                         "memory": self.memory.snapshot(), "avg_ms": 0})
        await self.send({"t": "status", "state": "idle", "reason": reason})

    def _history(self) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        rounds = self.memory.rounds[-settings.max_history_turns:]
        for r in rounds:
            if r.user:
                out.append(("user", r.user))
            if r.assistant:
                out.append(("assistant", r.assistant))
        return out[-2 * settings.max_history_turns:]

    # ------------------------------------------------------------------
    async def close(self) -> None:
        self.closed = True
        self.cancel_evt.set()
        if self.turn_task and not self.turn_task.done():
            self.turn_task.cancel()
        self.activation.reset()

    def stats(self) -> dict:
        return {
            "seconds": round(time.time() - self.connected_at, 1),
            "turns": self.turns,
            "frames": self.frames_seen,
            "avg_ms": round(sum(self.latencies) / len(self.latencies)) if self.latencies else None,
            "memory": self.memory.snapshot(),
            "engine": self.engines.info(),
            "scene_note": self.scene_note,
        }
