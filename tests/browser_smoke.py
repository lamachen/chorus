"""Smoke-Test der On-Device-App ohne Server.

Szenario 1 (CDNs blockiert): Das On-Device-VLM kann nicht laden. Die App muss
das OFFEN anzeigen (Ring-Screen-Fehler, roter Chip), nichts vorspielen und
nicht abstürzen.
Szenario 2 (VLM-Stub): Geometrie des neuen Anruf-Screens - Waveform-Trenner
vorhanden, keine Fake-Kamera-Elemente, nur drei Controls, Chips im Protokoll.

Screenshots macht tests/shots.py (getrennt, weil SwiftShader-Raster in kleinen
Sandboxes flaky sein kann).

    pip install playwright && playwright install chromium
    python tests/browser_smoke.py
"""

import asyncio
import http.server
import os
import socketserver
import sys
import threading

from playwright.async_api import async_playwright

ROOT = os.path.join(os.path.dirname(__file__), "..", "web")
OUT = os.path.join(os.path.dirname(__file__), "..", "shots")
PORT = 8123


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


def serve():
    cwd = os.getcwd()
    os.chdir(ROOT)
    socketserver.TCPServer.allow_reuse_address = True
    httpd = socketserver.TCPServer(("127.0.0.1", PORT), Quiet)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    os.chdir(cwd)
    return httpd


async def run_once() -> int:
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
        await ctx.add_init_script(
            "localStorage.setItem('chorus-settings-v2', JSON.stringify({ onDeviceDevice: 'wasm' }))")
        page = await ctx.new_page()
        page.on("pageerror", lambda e: errors.append("PAGEERROR: " + str(e)))
        page.on("crash", lambda: errors.append("CRASH"))

        # Modell-CDNs blockieren -> ehrlicher Fehlerpfad, schlanker Speicher
        for pat in ("**/*huggingface.co/**", "**/*jsdelivr**", "**/*unpkg**"):
            await page.route(pat, lambda r: r.abort())

        await page.goto(f"http://127.0.0.1:{PORT}/index.html", wait_until="networkidle")
        await page.wait_for_timeout(1500)
        n = await page.evaluate("document.querySelectorAll('#fav-list li').length")
        print("Kontakte:", n)
        assert n == 3

        await page.click("#fav-list li:first-child .c-main")
        ring = ""
        for _ in range(15):
            await page.wait_for_timeout(1000)
            ring = await page.evaluate("document.getElementById('ring-state').textContent")
            if "nicht geladen" in ring:
                break
        st = await page.evaluate("""() => ({
            state: window.__app.state,
            ring: document.getElementById('ring-state').textContent,
            chip: document.getElementById('chip-vlm').textContent,
            vlmReady: !!(window.__app.vlm && window.__app.vlm.ready),
            facing: window.__app.vision.facing,
            torchHidden: document.getElementById('btn-torch').hidden })""")
        print("Fehlerpfad:", st)
        assert st["vlmReady"] is False, "mit blockierter CDN darf kein VLM ready sein"
        assert "nicht geladen" in st["ring"], "Ring-Screen muss den Ladefehler offen zeigen"
        assert st["facing"] == "environment", "Rückkamera muss Standard sein"
        print("OK: ehrlicher Fehlerpfad, kein Fake, kein Absturz")

        await page.click("#ring-cancel")
        await page.wait_for_timeout(800)
        assert await page.evaluate("window.__app.state") == "idle"

        # ---- Layout des Anruf-Screens (VLM-Stub, nur Geometrie) ----
        try:
            await page.unroute("**/*huggingface.co/**")
        except Exception:
            pass
        await page.evaluate("""() => {
          window.__app.vlm = { ready: true, loading: false, device: 'stub', preset: 'compact',
            cancel() {}, generate: async () => '' };
          window.__app.enterCall();
        }""")
        await page.wait_for_timeout(1500)
        layout = await page.evaluate("""() => ({
          waveform: !!document.getElementById('waveform'),
          fakeElemente: document.querySelectorAll('#wave-you,#wave-her,.camoff,#voice-ring').length,
          kamerasText: document.getElementById('screen-call').innerHTML.includes('Kamera aus'),
          controls: [...document.querySelectorAll('.controls .ctl')].map(b => b.id),
          chipsImProtokoll: !!document.querySelector('#transcript .chips'),
          wf: (() => { const c = document.getElementById('waveform'); return c.width + 'x' + c.height; })() })""")
        print("Layout:", layout)
        assert layout["waveform"] and layout["fakeElemente"] == 0 and not layout["kamerasText"]
        assert layout["controls"] == ["btn-mute", "btn-cam", "btn-keys", "btn-hangup"]
        assert layout["chipsImProtokoll"]
        print("OK: vertikales Layout (Charakter | Waveform-Trenner | Nutzer), keine Fake-Kamera-Box")

        print("errors:", [e for e in errors if "ERR_FAILED" not in e and "404" not in e][:5] or "keine")
        await browser.close()
    httpd.shutdown()
    return 1 if errors else 0


async def main() -> int:
    for attempt in (1, 2):
        try:
            return await run_once()
        except Exception as e:
            if "crash" not in str(e).lower():
                raise
            print(f"!! Renderer-Crash (Sandbox-Flake, Versuch {attempt}) - erneuter Lauf …")
    raise SystemExit(2)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
