/* theme.js — light/dark toggle with persistence. The initial theme is set
   pre-paint by an inline <head> snippet; this only handles toggle clicks. */
(function () {
  'use strict';
  var root = document.documentElement;

  function setMeta(theme) {
    var m = document.querySelector('meta[name="theme-color"]');
    if (m) m.setAttribute('content', theme === 'dark' ? '#000000' : '#f6f7f9');
  }
  function store(v) {
    try { localStorage.setItem('llmbook:theme', v); } catch (e) {}
  }

  setMeta(root.getAttribute('data-theme'));

  document.addEventListener('click', function (e) {
    var btn = e.target.closest('[data-theme-toggle]');
    if (!btn) return;
    var next = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    root.setAttribute('data-theme', next);
    store(next);
    setMeta(next);
    btn.setAttribute('aria-pressed', String(next === 'dark'));
  });
})();
