"""Central configuration for the FastVLM video-call server.

Everything can be overridden by environment variables (or a .env file next to
the project root).  Nothing here is mandatory: with no model installed the
server automatically falls back to the built-in mock engine so that the whole
pipeline (browser -> WebSocket -> FastVLM slot -> TTS) can be exercised.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"

try:  # optional, keeps the dependency list small
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except Exception:  # pragma: no cover
    pass


def _env(name: str, default):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    if isinstance(default, bool):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(default, int) and not isinstance(default, bool):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    return raw


@dataclass
class Settings:
    # ---- networking -------------------------------------------------------
    host: str = _env("HOST", "0.0.0.0")
    port: int = _env("PORT", 8443)
    allow_origins: str = _env("ALLOW_ORIGINS", "*")

    # ---- FastVLM ----------------------------------------------------------
    # auto -> mlx-vlm on Apple Silicon, transformers elsewhere, mock if neither
    backend: str = _env("FASTVLM_BACKEND", "auto")          # auto|mlx|transformers|mock
    model: str = _env("FASTVLM_MODEL", "")                   # empty -> sensible default
    model_mlx: str = _env("FASTVLM_MODEL_MLX", "mlx-community/FastVLM-0.5B-bf16")
    model_pt: str = _env("FASTVLM_MODEL_PT", "apple/FastVLM-0.5B")
    device: str = _env("FASTVLM_DEVICE", "auto")             # auto|cpu|cuda|mps
    dtype: str = _env("FASTVLM_DTYPE", "auto")               # auto|float16|bfloat16|float32
    image_token_index: int = _env("FASTVLM_IMAGE_TOKEN_INDEX", -200)

    # ---- generation -------------------------------------------------------
    max_tokens: int = _env("MAX_TOKENS", 140)                # spoken replies are short
    temperature: float = _env("TEMPERATURE", 0.85)
    top_p: float = _env("TOP_P", 0.92)
    repetition_penalty: float = _env("REPETITION_PENALTY", 1.05)
    max_history_turns: int = _env("MAX_HISTORY_TURNS", 8)

    # ---- vision / memory (StreamBridge-style, arXiv:2505.05467) -----------
    vision_short_side: int = _env("VISION_SHORT_SIDE", 768)   # client-side capture size
    jpeg_quality: int = _env("JPEG_QUALITY", 80)
    memory_full_rounds: int = _env("MEMORY_FULL_ROUNDS", 2)   # verbatim rounds
    memory_mid_rounds: int = _env("MEMORY_MID_ROUNDS", 6)     # compressed rounds
    memory_summary_chars: int = _env("MEMORY_SUMMARY_CHARS", 420)
    memory_budget_chars: int = _env("MEMORY_BUDGET_CHARS", 1400)
    scene_note_max_tokens: int = _env("SCENE_NOTE_MAX_TOKENS", 48)

    # ---- proactive activation model --------------------------------------
    proactive: bool = _env("PROACTIVE", True)
    activation_mode: str = _env("ACTIVATION_MODE", "hybrid")  # off|heuristic|vlm|hybrid
    activation_interval: float = _env("ACTIVATION_INTERVAL", 2.5)
    activation_threshold: float = _env("ACTIVATION_THRESHOLD", 0.62)
    activation_cooldown: float = _env("ACTIVATION_COOLDOWN", 22.0)
    activation_min_lull: float = _env("ACTIVATION_MIN_LULL", 4.0)
    activation_novelty_gain: float = _env("ACTIVATION_NOVELTY_GAIN", 2.6)
    activation_lull_gain: float = _env("ACTIVATION_LULL_GAIN", 0.9)
    activation_probe_every: float = _env("ACTIVATION_PROBE_EVERY", 9.0)

    # ---- optional external text LLM (better dialogue than the 0.5B VLM) ---
    llm_backend: str = _env("LLM_BACKEND", "none")            # none|ollama|openai
    llm_base_url: str = _env("LLM_BASE_URL", "http://127.0.0.1:11434/v1")
    llm_model: str = _env("LLM_MODEL", "llama3.1:8b")
    llm_api_key: str = _env("LLM_API_KEY", "not-needed")
    llm_max_tokens: int = _env("LLM_MAX_TOKENS", 220)

    # ---- misc -------------------------------------------------------------
    log_frames: bool = _env("LOG_FRAMES", False)
    max_frame_bytes: int = _env("MAX_FRAME_BYTES", 2_500_000)

    def resolved_model(self) -> str:
        if self.model:
            return self.model
        return self.model_mlx if self.backend == "mlx" else self.model_pt

    def public_dict(self) -> dict:
        d = asdict(self)
        for secret in ("llm_api_key",):
            d.pop(secret, None)
        return d


settings = Settings()
