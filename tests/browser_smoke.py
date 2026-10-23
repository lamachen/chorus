"""UI-Smoke-Test ohne Server: statischer File-Server + Playwright.

Szenario 1 (CDN blockiert): Das On-Device-VLM kann nicht laden - die App muss
das OFFEN anzeigen (Ring-Screen-Fehler, roter Chip) und darf nicht abstürzen
und nichts vorspielen.
Szenario 2 (CDN erlaubt, nur optional): echter Modell-Load - zu schwer für CI,
dafür gibt es tests/ondevice_contract.mjs.

    pip install playwright && playwright install chromium
    python tests/browser_smoke.py
"""

import asyncio
import functools
import http.server
import os
import socketserver
import sys
import threading

from playwright.async_api import async_playwright

ROOT = os.path.join(os.path.dirname(__file__), "..", "web")
PORT = 8123


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


def serve():
    os.chdir(ROOT)
    socketserver.TCPServer.allow_reuse_address = True
    httpd = socketserver.TCPServer(("127.0.0.1", PORT), Quiet)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


async def main() -> int:
    httpd = serve()
    errors = []
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(args=[
            "--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream",
            "--autoplay-policy=no-user-gesture-required", "--no-sandbox",
        ])
        ctx = await browser.new_context(
            viewport={"width": 390, "height": 844}, device_scale_factor=1,
            is_mobile=True, has_touch=True,
            user_agent="Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Mobile Safari/537.36",
            permissions=["camera", "microphone"],
        )
        page = await ctx.new_page()
        page.on("pageerror", lambda e: errors.append("PAGEERROR: " + str(e)))
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" and "ERR_FAILED" not in m.text and "404" not in m.text else None)

        print("goto …")
        await page.goto(f"http://127.0.0.1:{PORT}/index.html", wait_until="networkidle")
        await page.wait_for_timeout(1500)
        print("boot ok")
        # ab hier offline -> Modell-CDN unerreichbar -> ehrlicher Fehlerpfad
        await ctx.set_offline(True)
        n_personas = await page.evaluate("document.querySelectorAll('#fav-list li').length")
        print("Kontakte:", n_personas)
        await page.screenshot(path="shots/1-dialer.png")

        print("call …")
        await page.click("#fav-list li:first-child .c-main")
        try:
            await page.wait_for_function(
                "() => document.getElementById('ring-state').textContent.includes('nicht geladen')",
                timeout=20000)
        except Exception:
            pass
        await page.wait_for_timeout(500)
        st = await page.evaluate("""() => ({
            state: window.__app.state,
            ringState: document.getElementById('ring-state').textContent,
            chip: document.getElementById('chip-vlm').textContent,
            vlmReady: window.__app.vlm && window.__app.vlm.ready,
            vlmError: (window.__app.vlm && window.__app.vlm.error || window.__app._vlmError || '').slice(0, 80),
            frames: window.__app.vision.frameId,
            vad: window.__app.vad.info.detector,
            youBars: [...document.querySelectorAll('#wave-you i')].map(i => +i.style.height.replace('px','')),
            torchHidden: document.getElementById('btn-torch').hidden,
            facing: window.__app.vision.facing })""")
        print("Fehlerpfad:", st)
        assert st["vlmReady"] is not True, "VLM dürfte mit blockiertem CDN nicht ready sein"
        assert "nicht geladen" in st["ringState"], "Ring-Screen muss den Ladefehler offen zeigen"
        assert st["facing"] == "environment", "Rückkamera muss Standard sein"
        await page.screenshot(path="shots/4-ring-error.png")

        await page.click("#ring-cancel")
        await page.wait_for_timeout(800)
        assert await page.evaluate("window.__app.state") == "idle"
        print("OK: ehrlicher Fehlerpfad, kein Fake, kein Absturz")
        print("errors:", errors[:5] or "keine")
        await browser.close()
    httpd.shutdown()
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
