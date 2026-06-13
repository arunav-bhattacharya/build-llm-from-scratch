'use strict';

const { renderCode } = require('./code');
const { parseFlashcards, renderFlashcards } = require('./flashcards');
const { parseQuiz, renderQuiz } = require('./quiz');
const { renderAssignment } = require('./assignment');

// Overrides markdown-it's fence renderer to dispatch by the fence's "language":
//   ```flashcards / ```quiz / ```assignment  -> interactive widgets
//   ```python / ```bash / ```text / …        -> Shiki-highlighted code block
module.exports = function fencePlugin(md, opts = {}) {
  const highlighter = opts.highlighter;

  md.renderer.rules.fence = function (tokens, idx, _options, env) {
    const tok = tokens[idx];
    const info = (tok.info || '').trim();
    const first = (info.split(/\s+/)[0] || '').toLowerCase();

    if (first === 'flashcards') return renderFlashcards(parseFlashcards(tok.content), env, md);
    if (first === 'quiz') return renderQuiz(parseQuiz(tok.content), env, md);
    if (first === 'assignment') return renderAssignment(tok.content, info, md);
    return renderCode(highlighter, tok.content, info, md);
  };
};
