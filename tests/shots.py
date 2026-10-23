"""Visuelle Screenshots (getrennt vom Smoke-Test: SwiftShader-Raster kann in
kleinen Sandboxes flaky crashen - deshalb hier mit Retry pro Shot)."""
import asyncio, os, threading, http.server, socketserver
from playwright.async_api import async_playwright
ROOT = os.path.join(os.path.dirname(__file__), "..", "web")
OUT = os.path.join(os.path.dirname(__file__), "..", "shots")
PORT = 8131

class Q(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a): pass

async def shoot(pw, port):
    os.chdir(ROOT)
    socketserver.TCPServer.allow_reuse_address = True
    httpd = socketserver.TCPServer(("127.0.0.1", port), Q)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    browser = await pw.chromium.launch(args=["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream",
                                              "--autoplay-policy=no-user-gesture-required", "--no-sandbox"])
    ctx = await browser.new_context(viewport={"width": 390, "height": 844}, device_scale_factor=2,
                                    is_mobile=True, has_touch=True, permissions=["camera", "microphone"])
    p = await ctx.new_page()
    await p.goto(f"http://127.0.0.1:{port}/index.html", wait_until="networkidle")
    await p.wait_for_timeout(1500)
    await p.screenshot(path=f"{OUT}/1-dialer.png")
    await p.evaluate("""() => { window.__app.vlm = { ready:true, loading:false, device:'stub', preset:'compact',
        cancel(){}, warmup: async () => {}, generate: async () => '' }; window.__app.enterCall(); }""")
    await p.wait_for_timeout(1500)
    await p.screenshot(path=f"{OUT}/5-call.png")
    await p.click("#btn-panel"); await p.wait_for_timeout(400)
    await p.screenshot(path=f"{OUT}/6-transcript.png")
    await browser.close()
    httpd.shutdown()

async def main():
    os.makedirs(OUT, exist_ok=True)
    for attempt in range(3):
        try:
            async with async_playwright() as pw:
                await shoot(pw, PORT + attempt)
            print("shots ok")
            return
        except Exception as e:
            print("retry", attempt, str(e)[:80])
    print("shots fehlgeschlagen (Sandbox-Flake)")

asyncio.run(main())
