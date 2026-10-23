"""Browser smoke test for the video-call app.

Runs the whole UI in headless Chromium with a fake camera/microphone,
starts a call, simulates a spoken turn, forces a proactive utterance and
writes screenshots to shots/.

    pip install playwright && playwright install chromium
    FASTVLM_BACKEND=mock python -m uvicorn server.main:app --port 8100 &
    python tests/browser_smoke.py
"""

import asyncio
import json
import os
import sys

from playwright.async_api import async_playwright

PORT = os.environ.get("PORT", "8100")
URL = f"http://127.0.0.1:{PORT}/"
OUT = os.path.join(os.path.dirname(__file__), "..", "shots")


async def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(args=[
            "--use-fake-device-for-media-stream",
            "--use-fake-ui-for-media-stream",
            "--autoplay-policy=no-user-gesture-required",
            "--no-sandbox",
        ])
        ctx = await browser.new_context(
            viewport={"width": 390, "height": 844}, device_scale_factor=2,
            is_mobile=True, has_touch=True,
            user_agent=("Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/126 Mobile Safari/537.36"),
            permissions=["camera", "microphone"],
        )
        page = await ctx.new_page()
        errors, logs = [], []

        def on_console(m):
            logs.append(f"{m.type}: {m.text}")
            if m.type == "error":
                errors.append(m.text)

        page.on("console", on_console)
        page.on("pageerror", lambda e: errors.append("PAGEERROR: " + str(e)))

        await page.goto(URL, wait_until="networkidle")
        await page.wait_for_timeout(1500)

        st = await page.evaluate(
            "window.__app ? {ok:true, personas: window.__app.personas.length,"
            " engine: window.__app.server && window.__app.server.engine.backend} : {ok:false}")
        print("app state:", json.dumps(st, default=str))
        await page.screenshot(path=f"{OUT}/1-dialer.png")

        await page.click('.tab[data-tab="key"]')
        await page.wait_for_timeout(300)
        for k in ("1", "2", "3"):
            await page.click(f'.key:has(b:text-is("{k}"))')
        await page.wait_for_timeout(200)
        await page.screenshot(path=f"{OUT}/2-keypad.png")
        await page.click('.tab[data-tab="fav"]')
        await page.wait_for_timeout(300)

        await page.click("#btn-settings")
        await page.wait_for_timeout(700)
        await page.screenshot(path=f"{OUT}/3-settings.png")
        await page.click("#settings-close")
        await page.wait_for_timeout(400)

        await page.click("#fav-list li:first-child .c-main")
        await page.wait_for_timeout(1200)
        if await page.locator("#od-no").count():
            await page.click("#od-no")   # On-Device-Download-Dialog ablehnen
        await page.screenshot(path=f"{OUT}/4-ringing.png")
        await page.wait_for_timeout(3500)

        call = await page.evaluate("""() => { const a = window.__app; return {
            state:a.state, framesSent:a.vision.frameId, vad:a.vad.info,
            voices:a.tts.voices.length, asrSupported:a.asr.supported,
            canvas:a.vision.view.width+'x'+a.vision.view.height,
            ws:a.ws && a.ws.readyState,
            chips:[...document.querySelectorAll('.chip')].map(c=>c.textContent),
            subHer:document.getElementById('sub-her').textContent,
            timer:document.getElementById('call-timer').textContent }; }""")
        print("in-call:", json.dumps(call, default=str, ensure_ascii=False))
        await page.screenshot(path=f"{OUT}/5-call.png")

        # headless Chromium has no SpeechRecognition -> inject the turn directly
        await page.evaluate("window.__app.pushUser('Hey, wie heißt du und was siehst du gerade?')")
        await page.wait_for_timeout(7000)
        after = await page.evaluate("""() => ({ frames:window.__app.vision.frameId,
            transcript:[...document.querySelectorAll('#transcript .msg')]
                .map(m=>m.className+': '+m.textContent.slice(0,120)),
            subHer:document.getElementById('sub-her').textContent.slice(0,150) })""")
        print("after turn:", json.dumps(after, default=str, ensure_ascii=False, indent=1))

        await page.click("#btn-panel")
        await page.wait_for_timeout(400)
        await page.screenshot(path=f"{OUT}/6-transcript.png")
        await page.click("#btn-panel")
        await page.wait_for_timeout(200)

        box = await page.locator("#mine").bounding_box()
        await page.screenshot(path=f"{OUT}/7-modelframes.png", clip=box)

        await page.evaluate("""() => window.__app.ws.send(JSON.stringify(
            {t:'cfg', proactive:true, activation_mode:'heuristic',
             activation_threshold:0.05, activation_cooldown:1}))""")
        await page.wait_for_timeout(10000)
        pro = await page.evaluate("""() => ({ activation: window.__app.lastActivation,
            sys:[...document.querySelectorAll('#transcript .msg.sys')].map(m=>m.textContent),
            transcript:[...document.querySelectorAll('#transcript .msg')].length })""")
        print("proactive:", json.dumps(pro, default=str, ensure_ascii=False))

        await page.click("#btn-hangup")
        await page.wait_for_timeout(900)
        await page.click('.tab[data-tab="recents"]')
        await page.wait_for_timeout(400)
        await page.screenshot(path=f"{OUT}/8-recents.png")

        print("\n=== console errors ===")
        for e in errors[:25]:
            print(" !", e[:300])
        print(f"({len(errors)} errors / {len(logs)} console messages)")
        await browser.close()
        return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
