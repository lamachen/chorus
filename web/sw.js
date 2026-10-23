/* Service Worker: versionierter Cache + Network-First für Code.
 *
 * WICHTIG: Es werden NUR Same-Origin-Requests behandelt. Cross-Origin
 * (Hugging Face, CDNs) läuft bewusst am SW vorbei - ein SW, der fremde
 * Fetches anfässt und im Fehlerfall undefinierte Responses liefert,
 * schiebt Bibliotheken wie transformers.js in Retry-Schleifen.
 *
 *   - CACHE-Name = Publish-Stamp (tools/publish.sh); alte Caches werden beim
 *     activate gelöscht, offene Tabs einmalig neu navigiert.
 *   - HTML/JS/CSS/build_info: network-first (Cache-Fallback nur offline)
 *   - vendor/ + assets (unveränderlich, groß): cache-first
 */

const STAMP = '20261023221529';
const CACHE = 'chorus-' + STAMP;
const IMMUTABLE = ['/vendor/', '/assets/', '.onnx', '.wasm'];

self.addEventListener('install', (e) => {
  e.waitUntil(
    caches.open(CACHE).then((c) =>
      c.addAll(['./', './index.html', './build_info.json', './styles.css', './js/app.js']).catch(() => {})
    ).then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(
        keys.filter((k) => k !== CACHE && (k.startsWith('chorus-') || k.startsWith('fastvlm-call-')))
            .map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
      .then(() => self.clients.matchAll({ type: 'window' }).then((clients) => {
        clients.forEach((cl) => { if ('navigate' in cl) cl.navigate(cl.url); });
      }))
  );
});

self.addEventListener('fetch', (e) => {
  const url = new URL(e.request.url);
  if (url.origin !== self.location.origin) return;          // cross-origin: Finger weg
  if (e.request.method !== 'GET') return;
  if (url.pathname.startsWith('/api') || url.pathname.startsWith('/ws')) return;

  const immutable = IMMUTABLE.some((t) => url.pathname.includes(t));

  if (immutable) {
    e.respondWith(
      caches.match(e.request).then((hit) => hit || fetch(e.request).then((res) => {
        if (res && res.ok) { const cp = res.clone(); caches.open(CACHE).then((c) => c.put(e.request, cp)); }
        return res;
      }))
    );
    return;
  }

  e.respondWith(
    fetch(e.request, { cache: 'no-store' })
      .then((res) => {
        if (res && res.ok) { const cp = res.clone(); caches.open(CACHE).then((c) => c.put(e.request, cp)); }
        return res;
      })
      .catch(() => caches.match(e.request).then((hit) =>
        hit || (e.request.mode === 'navigate'
          ? caches.match('./index.html')
          : new Response('offline', { status: 503, headers: { 'Content-Type': 'text/plain' } })))
  );
});
