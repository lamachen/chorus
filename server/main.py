"""FastAPI application: static web app + WebSocket call endpoint."""

from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from . import persona as P
from .config import WEB_DIR, settings
from .engine import EngineManager, analyse
from .llm import TextLLM
from .session import CallSession

mimetypes.add_type("application/wasm", ".wasm")
mimetypes.add_type("text/javascript", ".mjs")
mimetypes.add_type("application/octet-stream", ".onnx")
mimetypes.add_type("application/manifest+json", ".webmanifest")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger("fastvlm-call")

STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    engines = EngineManager(settings)
    llm = TextLLM(settings)
    STATE.update(engines=engines, llm=llm, started=time.time())
    log.info("backend=%s model=%s  (loading in background)", engines.engine.backend, engines.engine.model_id)
    loader = asyncio.create_task(engines.start())

    async def announce():
        await loader
        info = engines.info()
        log.info("engine ready: %s", json.dumps(info, default=str))
        if info.get("error"):
            log.warning("engine error: %s", info["error"])

    task = asyncio.create_task(announce())
    try:
        yield
    finally:
        task.cancel()
        loader.cancel()


app = FastAPI(title="FastVLM Video-Call", version="1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.allow_origins.split(",") if o.strip()] or ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/status")
async def status() -> JSONResponse:
    engines: EngineManager = STATE["engines"]
    return JSONResponse({
        "engine": engines.info(),
        "llm": STATE["llm"].info(),
        "personas": [
            {"id": p.pid, "name": p.name, "age": p.age, "city": p.city, "tagline": p.tagline,
             "hue": p.hue, "facts": p.facts, "greetings": p.greetings}
            for p in P.PERSONAS.values()
        ],
        "settings": settings.public_dict(),
        "uptime_s": round(time.time() - STATE["started"], 1),
    })


@app.post("/api/warm")
async def warm() -> JSONResponse:
    await STATE["engines"].start()
    return JSONResponse(STATE["engines"].info())


@app.post("/api/text")
async def one_shot(payload: dict) -> JSONResponse:
    """Debug helper: one reply without a websocket.
    {"text": "...", "persona": "mara", "lang": "de", "image_b64": "..."}"""
    import base64
    import io

    from PIL import Image

    engines: EngineManager = STATE["engines"]
    p = P.get_persona(payload.get("persona"))
    lang = payload.get("lang", "de")
    img = None
    if payload.get("image_b64"):
        img = Image.open(io.BytesIO(base64.b64decode(payload["image_b64"]))).convert("RGB")
    system = P.build_system_prompt(p, lang, memory_text="", scene_text="", proactive=True, rich=True)
    chunks = []
    async for piece in engines.stream(
        system=system, history=payload.get("history", []), user_text=payload.get("text", "hi"),
        image=img, max_tokens=payload.get("max_tokens", settings.max_tokens),
    ):
        chunks.append(piece)
    raw = "".join(chunks)
    return JSONResponse({"raw": raw, "clean": P.sanitize_reply(raw, p.name), "engine": engines.info()})


@app.get("/api/debug/scene")
async def debug_scene() -> JSONResponse:
    """Analyse the last frame a session received (mock engine uses the same code)."""
    for sess in STATE.get("sessions", {}).values():
        if sess.frame is not None:
            facts, _ = analyse(sess.frame)
            return JSONResponse({"frame_id": sess.frame_id, "facts": vars(facts),
                                 "describe_de": facts.describe("de"), "stats": sess.stats()})
    return JSONResponse({"error": "no active session with a frame"}, status_code=404)


@app.websocket("/ws/call")
async def ws_call(ws: WebSocket) -> None:
    await ws.accept()
    sessions = STATE.setdefault("sessions", {})
    sid = f"s{len(sessions)+1}-{int(time.time())}"
    session = CallSession(ws, STATE["engines"], STATE["llm"])
    sessions[sid] = session
    log.info("call connected: %s", sid)

    async def keepalive():
        while not session.closed:
            await asyncio.sleep(20)
            await session.send({"t": "keepalive", "clock": time.time()})

    loop_task = asyncio.create_task(session.proactive_loop())
    ka_task = asyncio.create_task(keepalive())
    try:
        await session.send({"t": "welcome", "sid": sid, "engine": session.engines.info(),
                            "llm": session.llm.info()})
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("bytes") is not None:
                await session.handle(msg["bytes"])
            elif msg.get("text") is not None:
                await session.handle(msg["text"])
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # pragma: no cover
        log.exception("session error: %s", exc)
        try:
            await session.send({"t": "error", "message": str(exc), "fatal": True})
        except Exception:
            pass
    finally:
        loop_task.cancel()
        ka_task.cancel()
        await session.close()
        stats = session.stats()
        sessions.pop(sid, None)
        log.info("call ended: %s %s", sid, json.dumps(stats, default=str)[:300])


app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")


def main() -> None:  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info", ws_max_size=16 * 1024 * 1024)


if __name__ == "__main__":  # pragma: no cover
    main()
