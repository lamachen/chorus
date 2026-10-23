/* Service Worker: versionierter Cache + Network-First für Code.
 *
 * Lesson learned: cache-first ohne Versionswechsel liefert nach einem Deploy
 * stale Code aus. Deshalb:
 *   - CACHE-Name carries the publish stamp (web/build_info.json -> ts);
 *     jeder Publish erzeugt einen neuen Cache-Namen, alte Caches werden im
 *     activate-Step gelöscht.
 *   - HTML/JS/CSS/build_info: NETWORK-FIRST (mit Cache-Fallback offline)
 *   - vendor/ + assets (unveränderlich, groß): cache-first
 * Das SW-Skript selbst holt der Browser immer frisch (Spec), daher greift
 * der neue Stamp zuverlässig.
 */

const STAMP = '20261023031416';
const CACHE = 'chorus-' + STAMP;
const IMMUTABLE = ['/vendor/', '/assets/', '.onnx', '.wasm', '.mjs.map'];

self.addEventListener('install', (e) => {
  e.waitUntil(
    caches.open(CACHE).then((c) =>
      c.addAll(['./', './index.html', './web/index.html', './web/build_info.json']).catch(() => {})
    ).then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE && k.startsWith('chorus-') || k.startsWith('fastvlm-call-')).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
      .then(() => self.clients.matchAll({ type: 'window' }).then((clients) => {
        // offene Tabs sofort auf den neuen Stand nehmen
        clients.forEach((cl) => { if ('navigate' in cl) cl.navigate(cl.url); });
      }))
  );
});

self.addEventListener('fetch', (e) => {
  const url = new URL(e.request.url);
  if (e.request.method !== 'GET') return;
  if (url.pathname.startsWith('/api') || url.pathname.startsWith('/ws')) return;

  const immutable = IMMUTABLE.some((t) => url.pathname.includes(t));

  if (immutable) {
    e.respondWith(
      caches.match(e.request).then((hit) => hit || fetch(e.request).then((res) => {
        if (res.ok) { const cp = res.clone(); caches.open(CACHE).then((c) => c.put(e.request, cp)); }
        return res;
      }))
    );
    return;
  }

  e.respondWith(
    fetch(e.request, { cache: 'no-store' })
      .then((res) => {
        if (res.ok) { const cp = res.clone(); caches.open(CACHE).then((c) => c.put(e.request, cp)); }
        return res;
      })
      .catch(() => caches.match(e.request).then((hit) => hit || caches.match('./web/index.html')))
  );
});
