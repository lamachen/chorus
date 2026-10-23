"""Decoupled, lightweight activation model  -  "should she speak right now?"

Second StreamBridge idea (arXiv:2505.05467): instead of only answering when the
user talks, a *separate* small module watches the stream continuously and fires
when something is worth commenting on.  StreamBridge learns that module; here it
is a cheap, interpretable scorer that runs on the frames the client already
sends, optionally backed by a one-word FastVLM probe ("hybrid" mode).

Three modes
-----------
off        never proactive (pure walkie-talkie: user speaks -> she answers)
heuristic  pure CV + conversational lull, no extra inference
vlm        only the FastVLM JA/NEIN probe
hybrid     CV score gates *when* the probe runs, probe gives the final vote
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from PIL import Image, ImageChops, ImageStat

THUMB = 48  # tiny thumbnails: the scorer must stay far cheaper than the VLM


@dataclass
class ActivationState:
    score: float = 0.0
    speak: bool = False
    reason: str = ""
    parts: dict = field(default_factory=dict)


class ActivationModel:
    def __init__(self, settings) -> None:
        self.s = settings
        self.mode = settings.activation_mode
        self.threshold = settings.activation_threshold
        self.cooldown = settings.activation_cooldown
        self.min_lull = settings.activation_min_lull
        self.novelty_gain = settings.activation_novelty_gain
        self.lull_gain = settings.activation_lull_gain

        self._prev_thumb: Image.Image | None = None
        self._anchor_thumb: Image.Image | None = None
        self._anchor_hist: list[int] | None = None
        self._last_fire = 0.0
        self._last_probe = 0.0
        self._probe_vote: float = 0.0          # decays towards 0
        self._probe_time: float = 0.0
        self._motion_ema: float = 0.0
        self.enabled = settings.proactive and self.mode != "off"

    # ------------------------------------------------------------------
    def set_mode(self, mode: str | None, proactive: bool | None = None) -> None:
        if mode:
            self.mode = mode
        if proactive is not None:
            self.enabled = bool(proactive) and self.mode != "off"

    def reset(self) -> None:
        self._prev_thumb = self._anchor_thumb = self._anchor_hist = None
        self._last_fire = self._last_probe = self._probe_vote = 0.0
        self._motion_ema = 0.0

    # ------------------------------------------------------------------
    @staticmethod
    def _thumb(img: Image.Image) -> Image.Image:
        return img.convert("L").resize((THUMB, THUMB), Image.BILINEAR)

    @staticmethod
    def _diff(a: Image.Image, b: Image.Image) -> float:
        """Mean absolute luminance difference in 0..1."""
        d = ImageChops.difference(a, b)
        return ImageStat.Stat(d).mean[0] / 255.0

    def set_probe_vote(self, yes: bool) -> None:
        self._probe_vote = 1.0 if yes else -0.6
        self._probe_time = time.time()

    def _vote(self) -> float:
        age = time.time() - self._probe_time
        return self._probe_vote * math.exp(-age / 25.0)

    def wants_probe(self, now: float) -> bool:
        """hybrid/vlm mode: is it time (and cheap enough) to ask the VLM?"""
        if self.mode not in ("vlm", "hybrid") or not self.enabled:
            return False
        return (now - self._last_probe) >= self.s.activation_probe_every

    def mark_probed(self, now: float) -> None:
        self._last_probe = now

    # ------------------------------------------------------------------
    def observe(
        self,
        img: Image.Image,
        *,
        now: float,
        user_speaking: bool = False,
        she_speaking: bool = False,
        last_exchange: float = 0.0,
        allow_fire: bool = True,
    ) -> ActivationState:
        """Feed one frame. Returns the activation decision for this moment."""
        st = ActivationState()
        thumb = self._thumb(img)

        motion = self._diff(thumb, self._prev_thumb) if self._prev_thumb else 0.0
        self._prev_thumb = thumb
        self._motion_ema = 0.85 * self._motion_ema + 0.15 * motion

        if not self.enabled:
            st.reason = "disabled"
            st.parts = {"motion": round(motion, 4)}
            return st

        if self._anchor_thumb is None:
            self._anchor_thumb = thumb
        novelty = self._diff(thumb, self._anchor_thumb)
        lull = max(0.0, now - last_exchange) if last_exchange else 0.0

        # never talk over somebody
        if user_speaking or she_speaking:
            st.score = 0.0
            st.reason = "busy"
            st.parts = {"motion": round(motion, 3), "novelty": round(novelty, 3), "lull": round(lull, 1)}
            return st

        lull_n = min(1.0, lull / 12.0)
        nov_n = min(1.0, novelty * self.novelty_gain)
        vote = self._vote() if self.mode in ("vlm", "hybrid") else 0.0

        if self.mode == "vlm":
            score = 0.5 + 0.5 * vote
        elif self.mode == "hybrid":
            score = _sig(1.35 * nov_n + self.lull_gain * lull_n + 0.55 * vote - 0.35 * min(1.0, self._motion_ema * 6))
        else:  # heuristic
            score = _sig(1.6 * nov_n + self.lull_gain * lull_n - 0.35 * min(1.0, self._motion_ema * 6))

        st.score = score
        st.parts = {
            "motion": round(motion, 3),
            "novelty": round(novelty, 3),
            "lull": round(lull, 1),
            "vote": round(vote, 2),
            "score": round(score, 3),
        }

        cooled = (now - self._last_fire) >= self.cooldown
        st.speak = bool(allow_fire and score >= self.threshold and cooled and lull >= self.min_lull)
        if self.mode == "vlm":
            st.speak = st.speak and vote > 0.2
        if st.speak:
            self.fire(now, thumb)
            st.reason = f"score={score:.2f} novelty={novelty:.2f} lull={lull:.1f}s"
        else:
            why = []
            if not cooled:
                why.append(f"cooldown {self.cooldown - (now - self._last_fire):.0f}s")
            if lull < self.min_lull:
                why.append("zu wenig Pause")
            if score < self.threshold:
                why.append("nichts Auffälliges")
            st.reason = ", ".join(why) or "wartet"
        return st

    def fire(self, now: float, thumb: Image.Image | None = None) -> None:
        """She spoke on her own initiative -> reset novelty anchor + cooldown."""
        self._last_fire = now
        self._anchor_thumb = thumb or self._prev_thumb
        self._probe_vote = 0.0

    def anchor_to_current(self) -> None:
        self._anchor_thumb = self._prev_thumb


def _sig(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))
