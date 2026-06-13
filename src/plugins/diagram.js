'use strict';

const container = require('markdown-it-container');

// Authoring:  ::: diagram ch01-ai-ml-dl-llm "AI ⊃ ML ⊃ Deep Learning ⊃ LLMs"
//             :::
// Inlines assets/diagrams/<name>.svg into the page (so it is DOM-themeable and
// needs no extra request) and wraps it in a <figure> with an optional caption.
module.exports = function diagramPlugin(md, opts = {}) {
  const getSvg = opts.getSvg || (() => null);
  const warn = opts.warn || (() => {});

  md.use(container, 'diagram', {
    validate(params) {
      return /^diagram(\s|$)/.test(params.trim());
    },
    render(tokens, idx) {
      const tok = tokens[idx];
      if (tok.nesting !== 1) return '';
      const info = tok.info.trim().replace(/^diagram\s*/, '');
      const m = info.match(/^(\S+)\s*(?:"([^"]*)")?/);
      const name = m && m[1];
      const caption = (m && m[2]) || '';
      const svg = name ? getSvg(name) : null;
      if (!svg) {
        warn(`Missing diagram SVG: "${name}"`);
        return (
          `<figure class="diagram diagram--missing">` +
          `<div class="diagram__frame">⚠ missing diagram: ${md.utils.escapeHtml(name || '(unnamed)')}</div>` +
          `</figure>`
        );
      }
      const fig =
        `<figure class="diagram">` +
        `<div class="diagram__frame">${svg}</div>` +
        (caption ? `<figcaption>${md.renderInline(caption)}</figcaption>` : '') +
        `</figure>`;
      return fig;
    },
  });
};
