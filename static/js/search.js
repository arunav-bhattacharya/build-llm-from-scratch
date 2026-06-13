/* search.js — MiniSearch modal. Rehydrates the build-time index (inlined as a
   global, no fetch → works on file://). Cmd/Ctrl-K to open; keyboard navigable. */
(function () {
  'use strict';
  var modal = document.getElementById('search');
  if (!modal) return;
  var input = modal.querySelector('.search__input');
  var results = modal.querySelector('[data-search-results]');
  var mini = null, ready = false, items = [], activeIdx = -1, debounce;
  var HINT = '<p class="search__hint">Start typing to search across every chapter.</p>';

  function ensureIndex() {
    if (ready) return true;
    try {
      if (typeof MiniSearch === 'undefined' || !window.__SEARCH_INDEX__) return false;
      mini = MiniSearch.loadJSON(window.__SEARCH_INDEX__, window.__SEARCH_OPTS__);
      ready = true;
    } catch (e) { ready = false; }
    return ready;
  }
  function esc(s) {
    return (s || '').replace(/[&<>"]/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c];
    });
  }
  function highlight(text, q) {
    var safe = esc(text);
    var terms = (q || '').split(/\s+/).filter(Boolean).map(function (t) { return t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); });
    if (!terms.length) return safe;
    try { return safe.replace(new RegExp('(' + terms.join('|') + ')', 'ig'), '<mark>$1</mark>'); }
    catch (e) { return safe; }
  }

  function open() {
    modal.hidden = false;
    modal.setAttribute('aria-hidden', 'false');
    document.body.style.overflow = 'hidden';
    ensureIndex();
    setTimeout(function () { input.focus(); }, 20);
  }
  function close() {
    modal.hidden = true;
    modal.setAttribute('aria-hidden', 'true');
    document.body.style.overflow = '';
    input.value = ''; results.innerHTML = HINT; items = []; activeIdx = -1;
  }
  function render(q) {
    if (!ensureIndex()) { results.innerHTML = '<p class="search__empty">Search index unavailable.</p>'; return; }
    if (!q.trim()) { results.innerHTML = HINT; items = []; activeIdx = -1; return; }
    var found = mini.search(q, { prefix: true, fuzzy: 0.2, boost: { title: 3 }, combineWith: 'AND' });
    if (!found.length) found = mini.search(q, { prefix: true, fuzzy: 0.3, boost: { title: 3 }, combineWith: 'OR' });
    if (!found.length) { results.innerHTML = '<p class="search__empty">No results for &ldquo;' + esc(q) + '&rdquo;.</p>'; items = []; return; }
    found = found.slice(0, 24);
    results.innerHTML = found.map(function (r) {
      var url = (window.__BASE__ || '') + r.url;
      return '<a class="search__result" href="' + url + '">' +
        '<div class="search__result-meta">' + esc(r.chapterTitle) + '</div>' +
        '<div class="search__result-title">' + highlight(r.title, q) + '</div>' +
        (r.snippet ? '<div class="search__result-snip">' + highlight(r.snippet, q) + '</div>' : '') +
        '</a>';
    }).join('');
    items = Array.prototype.slice.call(results.querySelectorAll('.search__result'));
    activeIdx = -1;
  }
  function setActive(i) {
    if (!items.length) return;
    activeIdx = (i + items.length) % items.length;
    items.forEach(function (el, k) { el.classList.toggle('is-active', k === activeIdx); });
    items[activeIdx].scrollIntoView({ block: 'nearest' });
  }

  document.addEventListener('click', function (e) {
    if (e.target.closest('[data-search-open]')) { open(); return; }
    if (e.target.closest('[data-search-close]')) { close(); }
  });
  document.addEventListener('keydown', function (e) {
    if ((e.metaKey || e.ctrlKey) && (e.key === 'k' || e.key === 'K')) {
      e.preventDefault(); return modal.hidden ? open() : close();
    }
    if (e.key === '/' && modal.hidden && !/INPUT|TEXTAREA|SELECT/.test((e.target.tagName || ''))) {
      e.preventDefault(); return open();
    }
    if (modal.hidden) return;
    if (e.key === 'Escape') close();
    else if (e.key === 'ArrowDown') { e.preventDefault(); setActive(activeIdx + 1); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); setActive(activeIdx - 1); }
    else if (e.key === 'Enter' && activeIdx >= 0 && items[activeIdx]) { window.location.href = items[activeIdx].getAttribute('href'); }
  });
  input.addEventListener('input', function () {
    clearTimeout(debounce);
    var q = input.value;
    debounce = setTimeout(function () { render(q); }, 110);
  });
})();
