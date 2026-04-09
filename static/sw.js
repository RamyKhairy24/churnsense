const CACHE = 'churnsense-v3';

// On install: skip waiting immediately — don't pre-cache anything
self.addEventListener('install', event => {
  self.skipWaiting();
});

// On activate: clean up old caches, claim clients right away
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

// Fetch strategy:
//   - HTML pages → always network, never cache
//   - /predict API → always network
//   - Static assets (js, css, images) → network-first, cache fallback
self.addEventListener('fetch', event => {
  const url = new URL(event.request.url);

  // Never intercept API calls or HTML navigations
  if (url.pathname === '/predict') return;
  if (event.request.mode === 'navigate') return;

  // Static assets: network-first with cache fallback
  event.respondWith(
    fetch(event.request)
      .then(res => {
        if (res.ok) {
          const clone = res.clone();
          caches.open(CACHE).then(c => c.put(event.request, clone));
        }
        return res;
      })
      .catch(() => caches.match(event.request))
  );
});
