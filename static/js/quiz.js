/* quiz.js — grade multiple-choice quizzes, reveal explanations, score, retry.
   Correctness is read from each option's data-correct attribute. */
(function () {
  'use strict';

  function grade(quiz) {
    var qs = quiz.querySelectorAll('[data-question]');
    var total = qs.length, correct = 0;
    qs.forEach(function (q) {
      var opts = q.querySelectorAll('.quiz__opt');
      var picked = null;
      opts.forEach(function (o) {
        var inp = o.querySelector('input');
        var isC = o.getAttribute('data-correct') === '1';
        if (inp && inp.checked) picked = o;
        if (isC) o.classList.add('is-correct');
        if (inp && inp.checked && !isC) o.classList.add('is-wrong');
        if (inp) inp.disabled = true;
      });
      if (picked && picked.getAttribute('data-correct') === '1') correct++;
      var ex = q.querySelector('.quiz__explain');
      if (ex) ex.hidden = false;
    });
    quiz.classList.add('is-checked');
    var score = quiz.querySelector('[data-quiz-score]');
    if (score) {
      score.textContent = 'You scored ' + correct + ' / ' + total;
      score.classList.toggle('is-good', correct === total);
    }
    var check = quiz.querySelector('[data-quiz-check]'); if (check) check.hidden = true;
    var reset = quiz.querySelector('[data-quiz-reset]'); if (reset) reset.hidden = false;
    try {
      var key = 'llmbook:quiz:' + (quiz.id || '');
      var prev = parseInt(localStorage.getItem(key) || '-1', 10);
      if (correct > prev) localStorage.setItem(key, String(correct));
    } catch (e) {}
  }

  function reset(quiz) {
    quiz.classList.remove('is-checked');
    quiz.querySelectorAll('.quiz__opt').forEach(function (o) {
      o.classList.remove('is-correct', 'is-wrong');
      var inp = o.querySelector('input');
      if (inp) { inp.checked = false; inp.disabled = false; }
    });
    quiz.querySelectorAll('.quiz__explain').forEach(function (x) { x.hidden = true; });
    var score = quiz.querySelector('[data-quiz-score]'); if (score) { score.textContent = ''; score.classList.remove('is-good'); }
    var check = quiz.querySelector('[data-quiz-check]'); if (check) check.hidden = false;
    var rst = quiz.querySelector('[data-quiz-reset]'); if (rst) rst.hidden = true;
  }

  document.addEventListener('click', function (e) {
    var c = e.target.closest('[data-quiz-check]');
    if (c) { grade(c.closest('[data-quiz]')); return; }
    var r = e.target.closest('[data-quiz-reset]');
    if (r) { reset(r.closest('[data-quiz]')); }
  });
})();
