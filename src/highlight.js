'use strict';

// Shiki highlighter singleton. Created once (async), then `codeToHtml` is synchronous,
// which lets us highlight inside markdown-it's synchronous render pipeline.
// Dual-theme output (github-light / github-dark) recolors instantly via CSS variables
// — see static/css/shiki.css — so the pre-highlighted HTML needs zero runtime JS.

let highlighterPromise = null;

const LANGS = ['python', 'bash', 'shell', 'json', 'text', 'diff', 'yaml'];

async function getHighlighter() {
  if (!highlighterPromise) {
    const { createHighlighter } = await import('shiki');
    highlighterPromise = createHighlighter({
      themes: ['github-light', 'github-dark'],
      langs: LANGS,
    });
  }
  return highlighterPromise;
}

module.exports = { getHighlighter };
