'use strict';

// Parses a ```flashcards fenced block into Q/A pairs and renders a flip-card deck.
// Authoring:
//   ```flashcards
//   Q: What is a token?
//   A: The atomic chunk of text a model reads — roughly a word or word-piece.
//   ---
//   Q: Why embeddings?
//   A: Neural nets need **numbers**, not raw characters.
//   ```
// Front shows the question; clicking flips to reveal the answer (3D rotateY).

function parseFlashcards(raw) {
  const cards = [];
  let cur = null;
  let mode = null;
  for (const line of raw.split('\n')) {
    const q = line.match(/^\s*Q:\s?(.*)$/);
    const a = line.match(/^\s*A:\s?(.*)$/);
    if (q) {
      if (cur) cards.push(cur);
      cur = { q: q[1], a: '' };
      mode = 'q';
    } else if (a && cur) {
      cur.a = a[1];
      mode = 'a';
    } else if (/^\s*---\s*$/.test(line)) {
      if (cur) { cards.push(cur); cur = null; mode = null; }
    } else if (cur) {
      if (mode === 'q') cur.q += '\n' + line;
      else if (mode === 'a') cur.a += '\n' + line;
    }
  }
  if (cur) cards.push(cur);
  return cards.map((c) => ({ q: c.q.trim(), a: c.a.trim() })).filter((c) => c.q || c.a);
}

function renderFlashcards(cards, env, md) {
  env.counter = env.counter || {};
  env.counter.deck = (env.counter.deck || 0) + 1;
  const deckId = `deck-${env.counter.deck}`;

  const cardHtml = cards
    .map(
      (c, i) => `
      <button class="flashcard${i === 0 ? ' is-current' : ''}" type="button" data-flashcard aria-pressed="false" aria-label="Flashcard ${i + 1} of ${cards.length}, tap to flip">
        <span class="flashcard__inner">
          <span class="flashcard__face flashcard__face--front">
            <span class="flashcard__tag">Question</span>
            <span class="flashcard__text">${md.renderInline(c.q)}</span>
            <span class="flashcard__hint">Tap to reveal</span>
          </span>
          <span class="flashcard__face flashcard__face--back">
            <span class="flashcard__tag">Answer</span>
            <span class="flashcard__text">${md.renderInline(c.a)}</span>
          </span>
        </span>
      </button>`
    )
    .join('');

  // One card shown at a time; Prev/Next step through the deck (see flashcards.js + components.css).
  return `
  <div class="flashdeck" id="${deckId}" data-flashdeck>
    <div class="flashdeck__stage">${cardHtml}</div>
    <div class="flashdeck__controls">
      <button class="flashdeck__nav-btn" type="button" data-deck-prev aria-label="Previous flashcard">← Prev</button>
      <span class="flashdeck__count"><strong data-deck-pos>1</strong> / ${cards.length}</span>
      <button class="flashdeck__nav-btn" type="button" data-deck-next aria-label="Next flashcard">Next →</button>
    </div>
    <div class="flashdeck__sub"><button class="chip-btn" type="button" data-deck-shuffle>Shuffle</button></div>
  </div>`;
}

module.exports = { parseFlashcards, renderFlashcards };
