/* code.js — copy-to-clipboard (from the raw <template> source, preserving exact
   indentation) and expand/collapse for long listings. */
(function () {
  'use strict';

  function legacyCopy(text) {
    try {
      var ta = document.createElement('textarea');
      ta.value = text; ta.style.position = 'fixed'; ta.style.top = '-9999px'; ta.style.opacity = '0';
      document.body.appendChild(ta); ta.focus(); ta.select();
      document.execCommand('copy'); document.body.removeChild(ta);
    } catch (e) {}
  }

  function flash(btn) {
    var t = btn.querySelector('.codeblock__btn-txt');
    var old = t ? t.textContent : '';
    if (t) t.textContent = 'Copied';
    btn.classList.add('is-copied');
    setTimeout(function () { if (t) t.textContent = old; btn.classList.remove('is-copied'); }, 1600);
  }

  document.addEventListener('click', function (e) {
    var copy = e.target.closest('[data-copy]');
    if (copy) {
      var block = copy.closest('.codeblock');
      var tpl = block && block.querySelector('.codeblock__source');
      var text = tpl ? (tpl.content ? tpl.content.textContent : tpl.textContent) : '';
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(function () { flash(copy); }, function () { legacyCopy(text); flash(copy); });
      } else { legacyCopy(text); flash(copy); }
      return;
    }
    var tog = e.target.closest('[data-code-toggle]');
    if (tog) {
      var block2 = tog.closest('.codeblock');
      var scroll = block2.querySelector('.codeblock__scroll');
      var open = block2.classList.toggle('is-open');
      if (scroll) scroll.classList.toggle('is-collapsed', !open);
      tog.setAttribute('aria-expanded', String(open));
      var t2 = tog.querySelector('.codeblock__btn-txt');
      if (t2) t2.textContent = open ? 'Hide code' : 'Show code';
    }
  });
})();
