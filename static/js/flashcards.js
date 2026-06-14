/* flashcards.js — one card shown at a time; Prev/Next step through the deck;
   tap a card to flip (CSS 3D, driven by aria-pressed); Shuffle reorders. */
(function () {
  'use strict';

  function cards(deck) {
    return Array.prototype.slice.call(deck.querySelectorAll('[data-flashcard]'));
  }

  function show(deck, idx) {
    var cs = cards(deck);
    if (!cs.length) return;
    idx = (idx + cs.length) % cs.length;
    cs.forEach(function (c, i) {
      c.classList.toggle('is-current', i === idx);
      if (i !== idx) c.setAttribute('aria-pressed', 'false'); // hidden cards face question-side
    });
    var pos = deck.querySelector('[data-deck-pos]');
    if (pos) pos.textContent = String(idx + 1);
    deck.__idx = idx;
  }

  function shuffle(deck) {
    var stage = deck.querySelector('.flashdeck__stage');
    if (!stage) return;
    var cs = cards(deck);
    for (var i = cs.length - 1; i > 0; i--) {
      var j = Math.floor(Math.random() * (i + 1));
      var t = cs[i]; cs[i] = cs[j]; cs[j] = t;
    }
    cs.forEach(function (c) { c.setAttribute('aria-pressed', 'false'); stage.appendChild(c); });
    show(deck, 0);
  }

  document.addEventListener('click', function (e) {
    var card = e.target.closest('[data-flashcard]');
    if (card) {
      card.setAttribute('aria-pressed', String(card.getAttribute('aria-pressed') !== 'true'));
      return;
    }
    var deck = e.target.closest('[data-flashdeck]');
    if (!deck) return;
    if (e.target.closest('[data-deck-next]')) show(deck, (deck.__idx || 0) + 1);
    else if (e.target.closest('[data-deck-prev]')) show(deck, (deck.__idx || 0) - 1);
    else if (e.target.closest('[data-deck-shuffle]')) shuffle(deck);
  });

  document.querySelectorAll('[data-flashdeck]').forEach(function (d) { show(d, 0); });
})();
