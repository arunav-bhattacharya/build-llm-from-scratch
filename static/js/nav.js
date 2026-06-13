/* nav.js — one toggle that collapses the sidebar on desktop and opens the
   drawer on mobile; scroll-spy for the on-this-page TOC; heading anchor links. */
(function () {
  'use strict';
  var root = document.documentElement;
  var scrim = document.querySelector('[data-nav-close]');
  var isMobile = function () { return window.matchMedia('(max-width: 920px)').matches; };

  function store(v) { try { localStorage.setItem('llmbook:sidebar', v); } catch (e) {} }

  function setDrawer(open) {
    if (open) { root.setAttribute('data-drawer', 'open'); if (scrim) scrim.hidden = false; }
    else { root.removeAttribute('data-drawer'); if (scrim) scrim.hidden = true; }
    var t = document.querySelector('.nav-toggle');
    if (t) t.setAttribute('aria-expanded', String(!!open));
  }
  function toggleCollapse() {
    var collapsed = root.getAttribute('data-sidebar') === 'collapsed';
    if (collapsed) { root.removeAttribute('data-sidebar'); store('expanded'); }
    else { root.setAttribute('data-sidebar', 'collapsed'); store('collapsed'); }
  }

  document.addEventListener('click', function (e) {
    if (e.target.closest('[data-nav-toggle]')) {
      if (isMobile()) setDrawer(root.getAttribute('data-drawer') !== 'open');
      else toggleCollapse();
      return;
    }
    if (e.target.closest('[data-nav-close]')) { setDrawer(false); return; }
    if (isMobile() && e.target.closest('.sidebar a')) setDrawer(false);
  });

  document.addEventListener('keydown', function (e) {
    var tag = (e.target && e.target.tagName) || '';
    if (e.key === 'Escape') setDrawer(false);
    if (e.key === '[' && !/INPUT|TEXTAREA|SELECT/.test(tag) && !isMobile()) toggleCollapse();
  });
  window.addEventListener('resize', function () { if (!isMobile()) setDrawer(false); });

  // Hover anchor links on headings
  var prose = document.querySelector('.prose');
  if (prose) {
    prose.querySelectorAll('h2[id], h3[id]').forEach(function (h) {
      var a = document.createElement('a');
      a.className = 'heading-anchor';
      a.href = '#' + h.id;
      a.setAttribute('aria-label', 'Direct link to “' + (h.textContent || '').trim() + '”');
      a.textContent = '#';
      h.appendChild(a);
    });
  }

  // Scroll-spy
  var tocLinks = Array.prototype.slice.call(document.querySelectorAll('.toc__link'));
  if (tocLinks.length && 'IntersectionObserver' in window) {
    var map = {};
    tocLinks.forEach(function (l) { map[l.getAttribute('href').slice(1)] = l; });
    var heads = Object.keys(map).map(function (id) { return document.getElementById(id); }).filter(Boolean);
    var current = null;
    function setActive(id) {
      if (current === id) return;
      current = id;
      tocLinks.forEach(function (l) { l.classList.remove('is-active'); });
      if (map[id]) map[id].classList.add('is-active');
    }
    var obs = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) { if (en.isIntersecting) setActive(en.target.id); });
    }, { rootMargin: '-78px 0px -68% 0px', threshold: 0 });
    heads.forEach(function (h) { obs.observe(h); });
  }
})();
