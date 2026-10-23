/* Minimal service worker: caches the heavy local assets (11 MB onnxruntime wasm,
   Silero ONNX, JS/CSS) so repeat calls on the phone start instantly.
   API and WebSocket traffic is never touched. */

const CACHE = 'fastvlm-call-v1';
const PRECACHE = [
  './', './index.html', './styles.css', './manifest.webmanifest',
  './js/app.js', './js/vad.js', './js/vad-worklet.js', './js/speech.js',
  './js/vision.js', './js/ui.js',
  './vendor/silero_vad.onnx', './vendor/ort/ort.wasm.min.mjs', './vendor/ort/ort-wasm-simd-threaded.mjs',
  './vendor/ort/ort-wasm-simd-threaded.wasm',
];

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(PRECACHE.map(u => new Request(u, { cache: 'reload' }))).catch(() => {})));
  self.skipWaiting();
});

self.addEventListener('activate', (e) => {
  e.waitUntil(caches.keys().then((keys) => Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))));
  self.clients.claim();
});

self.addEventListener('fetch', (e) => {
  const url = new URL(e.request.url);
  if (url.pathname.startsWith('/api') || url.pathname.startsWith('/ws')) return;
  if (e.request.method !== 'GET') return;
  e.respondWith(
    caches.match(e.request).then((hit) => hit || fetch(e.request).then((res) => {
      if (res.ok && (url.pathname.includes('/vendor/') || url.pathname.includes('/js/'))) {
        const copy = res.clone();
        caches.open(CACHE).then((c) => c.put(e.request, copy));
      }
      return res;
    }).catch(() => caches.match('./index.html')))
  );
});
