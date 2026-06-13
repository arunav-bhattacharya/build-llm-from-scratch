'use strict';

const container = require('markdown-it-container');

// Semantic content groupings rendered as styled cards. Pair them with a real
// Markdown H2 heading so they still appear in the on-this-page TOC + search:
//
//   ## Key takeaways
//   ::: takeaways
//   - …
//   :::
//
//   ::: objectives "What you'll learn"   (gets its own title — used at page top)
//   - …
//   :::
const ICONS = {
  takeaways:
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>',
  refs:
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20V3H6.5A2.5 2.5 0 0 0 4 5.5v14Z"/><path d="M8 7h8M8 11h6"/></svg>',
  objectives:
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1.4" fill="currentColor"/></svg>',
};
const NAMES = Object.keys(ICONS);

module.exports = function blocksPlugin(md) {
  NAMES.forEach((name) => {
    md.use(container, name, {
      validate(params) {
        return new RegExp('^' + name + '(\\s|$)').test(params.trim());
      },
      render(tokens, idx) {
        const tok = tokens[idx];
        if (tok.nesting === 1) {
          const info = tok.info.trim().replace(new RegExp('^' + name + '\\s*'), '');
          const tm = info.match(/"([^"]*)"/);
          const title = tm ? tm[1] : '';
          const head = title
            ? `<div class="block__head"><span class="block__ico" aria-hidden="true">${ICONS[name]}</span><h4 class="block__title">${md.utils.escapeHtml(title)}</h4></div>`
            : '';
          return `<section class="block block--${name}">${head}<div class="block__body">`;
        }
        return `</div></section>\n`;
      },
    });
  });
};
