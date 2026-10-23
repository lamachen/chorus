"""Optional external text LLM (Ollama or any OpenAI-compatible endpoint).

FastVLM-0.5B is a great pair of *eyes* but a modest conversationalist.  If you
run Ollama (or have an API key) the app can split the job:

    FastVLM  ->  "what do I see right now?"   (short scene note)
    LLM      ->  "what do I say?"             (in-character dialogue)

Implemented with the standard library only (urllib + SSE), so no extra deps.
"""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from typing import AsyncIterator


class TextLLM:
    def __init__(self, settings) -> None:
        self.s = settings
        self.enabled = (settings.llm_backend or "none").lower() not in ("", "none", "off")
        self.base_url = settings.llm_base_url.rstrip("/")
        self.model = settings.llm_model
        self.api_key = settings.llm_api_key
        self.error: str | None = None

    def info(self) -> dict:
        return {
            "enabled": self.enabled,
            "backend": self.s.llm_backend,
            "model": self.model if self.enabled else None,
            "base_url": self.base_url if self.enabled else None,
            "error": self.error,
        }

    async def stream(
        self,
        *,
        system: str,
        history: list[tuple[str, str]],
        user_text: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        if not self.enabled:
            return
        payload = {
            "model": self.model,
            "stream": True,
            "max_tokens": max_tokens or self.s.llm_max_tokens,
            "temperature": self.s.temperature if temperature is None else temperature,
            "messages": ([{"role": "system", "content": system}] if system else [])
            + [{"role": r, "content": c} for r, c in history]
            + [{"role": "user", "content": user_text}],
        }
        url = self.base_url + "/chat/completions"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "text/event-stream",
            },
            method="POST",
        )
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        SENTINEL = object()

        def pump() -> None:
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    for raw in resp:
                        line = raw.decode("utf-8", "replace").strip()
                        if not line or not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            obj = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        delta = (obj.get("choices") or [{}])[0].get("delta") or {}
                        piece = delta.get("content")
                        if piece:
                            loop.call_soon_threadsafe(queue.put_nowait, piece)
            except urllib.error.HTTPError as exc:  # pragma: no cover
                loop.call_soon_threadsafe(
                    queue.put_nowait, RuntimeError(f"LLM HTTP {exc.code}: {exc.read()[:200]!r}")
                )
            except Exception as exc:  # pragma: no cover
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, SENTINEL)

        fut = loop.run_in_executor(None, pump)
        try:
            while True:
                item = await queue.get()
                if item is SENTINEL:
                    break
                if isinstance(item, Exception):
                    self.error = str(item)
                    raise item
                yield item
        finally:
            await fut

    async def ping(self) -> bool:
        if not self.enabled:
            return False
        try:
            async for _ in self.stream(system="", history=[], user_text="hi", max_tokens=1):
                break
            self.error = None
            return True
        except Exception as exc:
            self.error = str(exc)
            return False
