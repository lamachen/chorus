"""Integrations-Test der SIGNAL-KETTE (Grenzen gestubt, Logik real).

Kette:  Gehörtes/Getipptes -> vlmTurn -> VLM.generate(Frame+Text) -> Tokens
        -> Satz-Splitter -> TTS-Queue -> Transcript/Bubbles -> Barge-in-Cancel
        sowie: Proaktiv-Scorer -> vlmTurn.

Jede Stufe wird einzeln assertiert und als PASS/FAIL ausgegeben - kein
stummes "irgendwo klemmt es" mehr.

    pip install playwright && playwright install chromium
    python tests/chain_test.py
"""

import asyncio
import http.server
import os
import socketserver
import sys
import threading

from playwright.async_api import async_playwright

ROOT = os.path.join(os.path.dirname(__file__), "..", "web")
PORT = 8135

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""), flush=True)


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


async def main() -> int:
    cwd = os.getcwd()
    os.chdir(ROOT)
    socketserver.TCPServer.allow_reuse_address = True
    httpd = socketserver.TCPServer(("127.0.0.1", PORT), Quiet)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    os.chdir(cwd)

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(args=[
            "--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream",
            "--autoplay-policy=no-user-gesture-required", "--no-sandbox",
        ])
        ctx = await browser.new_context(viewport={"width": 390, "height": 844},
                                        permissions=["camera", "microphone"])
        page = await ctx.new_page()
        page.on("pageerror", lambda e: print("PAGEERROR:", str(e)[:200], flush=True))
        for pat in ("**/*huggingface.co/**", "**/*jsdelivr**", "**/*unpkg**"):
            await page.route(pat, lambda r: r.abort())
        await page.goto(f"http://127.0.0.1:{PORT}/index.html", wait_until="networkidle")
        await page.wait_for_timeout(1200)

        # VLM-Stub mit Aufzeichnung + TTS-Spy + Call starten
        await page.evaluate("""() => {
          const a = window.__app;
          window.__rec = { calls: [], cancels: 0 };
          a.vlm = {
            ready: true, loading: false, device: 'stub', preset: 'compact',
            cancel() { window.__rec.cancels++; },
            generate: async (o) => {
              window.__rec.calls.push({ user: o.userText, hasCanvas: !!o.canvas, sys: (o.system||'').slice(0,40), hist: (o.history||[]).length });
              const reply = 'Ich sehe dich gerade deutlich. Magst du erzählen, was du machst?';
              const words = reply.split(' ');
              for (let i = 0; i < words.length; i++) {
                o.onDelta && o.onDelta(words[i] + (i < words.length - 1 ? ' ' : ''));
                await new Promise(r => setTimeout(r, 8));
              }
              return reply;
            },
          };
          window.__ttsQ = [];
          const orig = a.tts.enqueue.bind(a.tts);
          a.tts.enqueue = (t) => { window.__ttsQ.push(t); return orig(t); };
          a.enterCall();
        }""")
        await page.wait_for_timeout(1200)

        # Stufe 0: Begrüssung kommt aus dem VLM (Direktiv, kein Skripttext)
        g0 = await page.evaluate("() => window.__rec.calls.length ? window.__rec.calls[0] : null")
        check("Kette 0: Connect -> VLM-Turn (Begrüssungs-Direktiv)", g0 and 'Stille' not in g0['user'] and len(g0['user']) > 20, str(g0)[:90])
        check("Kette 0b: Frame (Canvas) liegt am VLM an", g0 and g0['hasCanvas'])

        # Stufe 1: Gehörtes (ASR-Final) -> VLM-Input
        await page.evaluate("() => window.__app.onAsrFinal('Hallo Mara, was siehst du gerade bei mir?')")
        await page.wait_for_timeout(1600)
        c1 = await page.evaluate("() => window.__rec.calls[window.__rec.calls.length-1]")
        check("Kette 1: ASR-Final -> VLM-User-Text", 'was siehst du gerade' in c1['user'], c1['user'][:60])
        check("Kette 1b: System-Prompt (Charakter) liegt an", 'Mara' in c1['sys'] or 'ROLLE' in c1['sys'], c1['sys'])

        # Stufe 2: Tokens -> Satz-Splitter -> TTS-Queue
        q = await page.evaluate("() => window.__ttsQ")
        check("Kette 2: Tokens -> Sätze -> TTS-Queue", len(q) >= 2, f"{len(q)} Sätze: {q[:2]}")

        # Stufe 3: Transcript zeigt GEHÖRTES + Antwort
        tr = await page.evaluate("[...document.querySelectorAll('#transcript .msg')].map(m => m.className + '::' + m.textContent)")
        check("Kette 3: Protokoll mit Du-Bubble", any('msg you' in t and 'was siehst du' in t for t in tr))
        check("Kette 3b: Protokoll mit Ihrer-Bubble", any('msg her' in t for t in tr))

        # Stufe 4: Tastatur-Eingang (gleichwertig)
        await page.evaluate("""() => { document.getElementById('keyrow').hidden = false;
          document.getElementById('key-input').value = 'Getippter Testsatz Nummer eins';
          document.getElementById('key-send').click(); }""")
        await page.wait_for_timeout(1400)
        c4 = await page.evaluate("() => window.__rec.calls[window.__rec.calls.length-1]")
        check("Kette 4: Tastatur-Eingang -> VLM-User-Text", 'Getippter Testsatz' in c4['user'], c4['user'][:50])

        # Stufe 5: Barge-in bricht VLM + TTS ab
        b = await page.evaluate("""async () => {
          const a = window.__app;
          a.tts.queue.push('Restsatz');            // sprechend simulieren
          const before = window.__rec.cancels;
          a.onSpeechStart();                        // VAD sagt: Nutzer spricht
          await new Promise(r => setTimeout(r, 200));
          return { cancels: window.__rec.cancels - before, queue: a.tts.queue.length, speaking: a.tts.speaking };
        }""")
        check("Kette 5: Barge-in -> VLM-Cancel + TTS-Stop", b['cancels'] >= 1 and b['queue'] == 0, str(b))

        # Stufe 6: Proaktiv-Scorer -> VLM-Turn
        await page.evaluate("""() => { const a = window.__app; a.act.threshold = 0; a.act.cooldown = 0; a.act.minLull = 0; a.lastExchange = performance.now() - 99999; }""")
        await page.wait_for_timeout(3200)
        pro = await page.evaluate("""() => {
          const calls = window.__rec.calls;
          return calls.some(c => c.user.includes('Stille'));
        }""")
        check("Kette 6: Aktivierungs-Scorer -> proaktiver VLM-Turn", pro)

        # Stufe 7: History wächst korrekt (Kontext für Folgerunden)
        h = await page.evaluate("() => window.__app.vlmHistory.length")
        check("Kette 7: Gesprächs-History für Kontext", h >= 4, f"{h} Einträge")

        await browser.close()
    httpd.shutdown()
    fails = [r for r in RESULTS if not r[1]]
    print(f"\n{len(RESULTS) - len(fails)}/{len(RESULTS)} Stufen PASS" + (f" — FAILS: {[f[0] for f in fails]}" if fails else ""))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
