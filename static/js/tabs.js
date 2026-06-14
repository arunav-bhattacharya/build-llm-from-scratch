/* tabs.js — turns the "Test your knowledge" section into tabs
   (Flashcards / Quiz / Assignments), built from the rendered widgets. */
(function () {
  'use strict';
  var h = document.getElementById('test-your-knowledge');
  if (!h) return;

  // Collect the widgets that follow the heading (until the next heading).
  var nodes = [], el = h.nextElementSibling;
  while (el && !/^H[1-3]$/.test(el.tagName)) { nodes.push(el); el = el.nextElementSibling; }

  var flash = nodes.filter(function (n) { return n.classList && n.classList.contains('flashdeck'); });
  var quiz = nodes.filter(function (n) { return n.classList && n.classList.contains('quiz'); });
  var assign = nodes.filter(function (n) { return n.classList && n.classList.contains('assignment'); });

  var groups = [];
  if (flash.length) groups.push({ label: 'Flashcards', items: flash });
  if (quiz.length) groups.push({ label: quiz.length > 1 ? 'Quizzes' : 'Quiz', items: quiz });
  if (assign.length) groups.push({ label: assign.length > 1 ? 'Assignments' : 'Assignment', items: assign });
  if (groups.length < 2) return; // nothing worth tabbing

  var wrap = document.createElement('div'); wrap.className = 'tk-tabs';
  var bar = document.createElement('div'); bar.className = 'tk-tabs__bar'; bar.setAttribute('role', 'tablist');
  var panels = document.createElement('div'); panels.className = 'tk-tabs__panels';

  groups.forEach(function (g, i) {
    var btn = document.createElement('button');
    btn.className = 'tk-tabs__tab' + (i === 0 ? ' is-active' : '');
    btn.type = 'button';
    btn.setAttribute('role', 'tab');
    btn.setAttribute('aria-selected', i === 0 ? 'true' : 'false');
    btn.dataset.tab = String(i);
    btn.textContent = g.label;
    bar.appendChild(btn);

    var panel = document.createElement('div');
    panel.className = 'tk-tabs__panel' + (i === 0 ? ' is-active' : '');
    panel.setAttribute('role', 'tabpanel');
    panel.dataset.panel = String(i);
    g.items.forEach(function (it) { panel.appendChild(it); }); // moves the node out of the flow
    panels.appendChild(panel);
  });

  wrap.appendChild(bar);
  wrap.appendChild(panels);

  // Insert after any intro paragraph that's still in place (widgets were moved out).
  var anchor = h;
  nodes.forEach(function (n) { if (n.isConnected) anchor = n; });
  anchor.parentNode.insertBefore(wrap, anchor.nextSibling);

  bar.addEventListener('click', function (e) {
    var t = e.target.closest('.tk-tabs__tab');
    if (!t) return;
    var idx = t.dataset.tab;
    bar.querySelectorAll('.tk-tabs__tab').forEach(function (b) {
      var on = b.dataset.tab === idx;
      b.classList.toggle('is-active', on);
      b.setAttribute('aria-selected', on ? 'true' : 'false');
    });
    panels.querySelectorAll('.tk-tabs__panel').forEach(function (p) {
      p.classList.toggle('is-active', p.dataset.panel === idx);
    });
  });
})();
