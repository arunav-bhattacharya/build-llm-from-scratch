/* flashcards.js — flip on click/Enter, shuffle, reset. The 3D flip itself is
   pure CSS driven by aria-pressed. */
(function () {
  'use strict';

  function shuffle(deck) {
    var grid = deck.querySelector('.flashdeck__grid');
    if (!grid) return;
    var cards = Array.prototype.slice.call(grid.children);
    for (var i = cards.length - 1; i > 0; i--) {
      var j = Math.floor(Math.random() * (i + 1));
      var tmp = cards[i]; cards[i] = cards[j]; cards[j] = tmp;
    }
    cards.forEach(function (c) { c.setAttribute('aria-pressed', 'false'); grid.appendChild(c); });
  }

  document.addEventListener('click', function (e) {
    var card = e.target.closest('[data-flashcard]');
    if (card) {
      card.setAttribute('aria-pressed', String(card.getAttribute('aria-pressed') !== 'true'));
      return;
    }
    var sh = e.target.closest('[data-deck-shuffle]');
    if (sh) { shuffle(sh.closest('[data-flashdeck]')); return; }
    var rs = e.target.closest('[data-deck-reset]');
    if (rs) {
      rs.closest('[data-flashdeck]')
        .querySelectorAll('[data-flashcard]')
        .forEach(function (c) { c.setAttribute('aria-pressed', 'false'); });
    }
  });
})();
