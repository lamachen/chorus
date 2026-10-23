"""Stream memory: round-decayed compression buffer.

Directly inspired by StreamBridge (Wang et al., NeurIPS 2025, arXiv:2505.05467):
a *memory buffer* whose entries lose fidelity the older the round they belong to,
plus a running summary that absorbs everything that falls out of the window.

FastVLM is a single-image model, so the "video memory" lives in text:

    round age 0        -> live frame goes in as an actual image, transcript verbatim
    round age 1..FULL  -> verbatim transcript + full scene note
    round age FULL+MID -> scene note only (compressed to <= N chars)
    older              -> folded into a rolling summary

`compress()` is intentionally cheap and synchronous-safe: it prefers an
extractive fallback and only calls the model when the engine is idle.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field


@dataclass
class Round:
    idx: int
    t: float
    user: str = ""
    assistant: str = ""
    scene: str = ""                 # what she saw during this round
    scene_brief: str = ""           # compressed version of `scene`
    proactive: bool = False
    frame_id: int = -1

    def render(self, level: int, max_chars: int = 220) -> str:
        """level 0 = full, 1 = mid, 2 = brief."""
        bits: list[str] = []
        if level == 0:
            if self.scene:
                bits.append(f"[ihr Blick: {self.scene}]")
            if self.user:
                bits.append(f"ER: {self.user}")
            if self.assistant:
                bits.append(f"SIE: {self.assistant}")
        elif level == 1:
            note = self.scene_brief or self.scene
            if note:
                bits.append(f"[{note}]")
            if self.user:
                bits.append(f"ER: {_clip(self.user, max_chars)}")
            if self.assistant:
                bits.append(f"SIE: {_clip(self.assistant, max_chars)}")
        else:
            note = self.scene_brief or _clip(self.scene, 60)
            frag = _clip(self.user or self.assistant, 70)
            if note:
                bits.append(note)
            if frag:
                bits.append(frag)
        text = " | ".join(b for b in bits if b).strip()
        tag = " (sie hat von selbst geredet)" if self.proactive and level == 0 else ""
        return text + tag


def _clip(s: str, n: int) -> str:
    s = re.sub(r"\s+", " ", (s or "")).strip()
    if len(s) <= n:
        return s
    cut = s[:n]
    i = max(cut.rfind(" "), cut.rfind("."), cut.rfind(","))
    return (cut[:i] if i > n * 0.5 else cut).rstrip(" ,.;") + "…"


_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+")


class StreamMemory:
    """Round-decayed buffer + rolling summary."""

    def __init__(
        self,
        full_rounds: int = 2,
        mid_rounds: int = 6,
        summary_chars: int = 420,
        budget_chars: int = 1400,
        keep_rounds: int = 40,
    ) -> None:
        self.full_rounds = max(1, full_rounds)
        self.mid_rounds = max(1, mid_rounds)
        self.summary_chars = summary_chars
        self.budget_chars = budget_chars
        self.keep_rounds = keep_rounds
        self.rounds: list[Round] = []
        self.summary: str = ""
        self._n = 0
        self._t0 = time.time()

    # -- mutation ---------------------------------------------------------
    def open_round(self, frame_id: int = -1, scene: str = "") -> Round:
        r = Round(idx=self._n, t=time.time() - self._t0, frame_id=frame_id, scene=scene)
        self._n += 1
        self.rounds.append(r)
        if len(self.rounds) > self.keep_rounds:
            dropped = self.rounds[: len(self.rounds) - self.keep_rounds]
            self.rounds = self.rounds[len(self.rounds) - self.keep_rounds :]
            self._absorb(dropped)
        return r

    def current(self) -> Round | None:
        return self.rounds[-1] if self.rounds else None

    def set_user(self, text: str) -> None:
        r = self.current()
        if r is not None and text:
            r.user = _clip(text, 400)

    def set_assistant(self, text: str, proactive: bool = False) -> None:
        r = self.current()
        if r is not None and text:
            r.assistant = _clip(text, 400)
            r.proactive = proactive

    def set_scene(self, text: str, brief: str = "") -> None:
        r = self.current()
        if r is None or not text:
            return
        r.scene = _clip(text, 260)
        r.scene_brief = _clip(brief or text, 90)

    def needs_compression(self) -> Round | None:
        """The oldest round that is still inside the mid window but has no brief."""
        lo = max(0, len(self.rounds) - (self.full_rounds + self.mid_rounds))
        for r in self.rounds[lo:-1] if len(self.rounds) > 1 else []:
            if r.scene and not r.scene_brief:
                return r
            if r.user and len(r.user) > 120:
                return r
        return None

    def apply_compression(self, r: Round, text: str) -> None:
        r.scene_brief = _clip(text, 90)
        r.user = _clip(r.user, 120)
        r.assistant = _clip(r.assistant, 120)

    def _absorb(self, dropped: list[Round]) -> None:
        bits = []
        for r in dropped:
            note = r.scene_brief or _clip(r.scene, 50)
            talk = _clip(r.assistant or r.user, 60)
            if note or talk:
                bits.append(" ".join(x for x in (note, talk) if x))
        if not bits:
            return
        merged = (self.summary + " " + " ".join(bits)).strip()
        self.summary = _clip(merged, self.summary_chars)

    # -- rendering --------------------------------------------------------
    def render(self) -> str:
        """Text block that goes into the system prompt."""
        out: list[str] = []
        n = len(self.rounds)
        if not n:
            return ""
        if self.summary:
            out.append(f"davor (zusammengefasst): {self.summary}")
        for pos, r in enumerate(reversed(self.rounds)):
            age = pos  # 0 = newest
            if age < self.full_rounds:
                level = 0
            elif age < self.full_rounds + self.mid_rounds:
                level = 1
            else:
                level = 2
            line = r.render(level)
            if line:
                out.append(line)
            if sum(len(x) for x in out) > self.budget_chars:
                break
        return "\n".join(reversed(out))

    def snapshot(self) -> dict:
        return {
            "rounds": len(self.rounds),
            "summary_chars": len(self.summary),
            "render_chars": len(self.render()),
            "full_rounds": self.full_rounds,
            "mid_rounds": self.mid_rounds,
        }
