'use strict';

// Parses a ```quiz fenced block into multiple-choice questions and renders an
// interactive quiz (instant per-option feedback, scoring, explanations, retry).
// Correctness lives in DOM data attributes (data-correct) — read by quiz.js.
// Authoring:
//   ```quiz
//   1. Softmax converts a vector of scores into…
//      - ( ) larger raw scores
//      - (x) a probability distribution that sums to 1
//      - ( ) token IDs
//      > It exponentiates each score then normalizes, so outputs are positive and sum to 1.
//
//   2. …
//   ```

function parseQuiz(raw) {
  const questions = [];
  let cur = null;
  for (const line of raw.split('\n')) {
    const qm = line.match(/^\s*\d+\.\s+(.*)$/);
    const om = line.match(/^\s*[-*]\s*\(([ xX])\)\s*(.*)$/);
    const em = line.match(/^\s*>\s?(.*)$/);
    if (qm) {
      if (cur) questions.push(cur);
      cur = { prompt: qm[1].trim(), options: [], explain: '' };
    } else if (om && cur) {
      cur.options.push({ text: om[2].trim(), correct: om[1].toLowerCase() === 'x' });
    } else if (em && cur) {
      cur.explain += (cur.explain ? '\n' : '') + em[1];
    } else if (cur && cur.explain && line.trim()) {
      cur.explain += '\n' + line.trim();
    }
  }
  if (cur) questions.push(cur);
  return questions.filter((q) => q.options.length);
}

function renderQuiz(questions, env, md) {
  env.counter = env.counter || {};
  env.counter.quiz = (env.counter.quiz || 0) + 1;
  const quizId = `quiz-${env.counter.quiz}`;

  const qHtml = questions
    .map((q, qi) => {
      const name = `${quizId}-q${qi}`;
      const opts = q.options
        .map(
          (o, oi) => `
          <label class="quiz__opt" data-correct="${o.correct ? '1' : '0'}">
            <input type="radio" name="${name}" value="${oi}">
            <span class="quiz__marker" aria-hidden="true"></span>
            <span class="quiz__opt-text">${md.renderInline(o.text)}</span>
          </label>`
        )
        .join('');
      const explain = q.explain
        ? `<div class="quiz__explain" hidden><span class="quiz__explain-tag">Why</span>${md.renderInline(q.explain)}</div>`
        : '';
      return `
      <li class="quiz__q" data-question>
        <p class="quiz__prompt"><span class="quiz__num">${qi + 1}</span>${md.renderInline(q.prompt)}</p>
        <div class="quiz__opts">${opts}</div>
        ${explain}
      </li>`;
    })
    .join('');

  return `
  <form class="quiz" id="${quizId}" data-quiz data-total="${questions.length}">
    <ol class="quiz__list">${qHtml}</ol>
    <div class="quiz__foot">
      <button class="btn btn--primary" type="button" data-quiz-check>Check answers</button>
      <button class="btn btn--ghost" type="button" data-quiz-reset hidden>Try again</button>
      <output class="quiz__score" data-quiz-score aria-live="polite"></output>
    </div>
  </form>`;
}

module.exports = { parseQuiz, renderQuiz };
