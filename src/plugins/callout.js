'use strict';

const container = require('markdown-it-container');

// Compact, stroke-based icons (inherit currentColor). One per callout type.
const ICONS = {
  tip:     '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M9 18h6M10 21h4M12 3a6 6 0 0 0-4 10.5c.7.7 1 1.2 1 2.5h6c0-1.3.3-1.8 1-2.5A6 6 0 0 0 12 3Z"/></svg>',
  note:    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M12 8h.01M11 12h1v4h1"/></svg>',
  warning: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M10.3 3.8 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.8a2 2 0 0 0-3.4 0Z"/><path d="M12 9v4M12 17h.01"/></svg>',
  analogy: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3a9 9 0 0 0-5 16.5V21h10v-1.5A9 9 0 0 0 12 3Z"/><path d="M9 21h6M2 12H1M23 12h-1M5 5 4 4M20 5l1-1"/></svg>',
  example: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M4 5h16M4 12h10M4 19h7"/></svg>',
  key:     '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><circle cx="7.5" cy="15.5" r="3.5"/><path d="m10 13 8-8M15 5l3 3M13 7l2 2"/></svg>',
  math:    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M5 4h14M9 4l-2 16M15 4l2 16M4 12h16"/></svg>',
};

const DEFAULT_TITLE = {
  tip: 'Tip',
  note: 'Note',
  warning: 'Watch out',
  analogy: 'Real-world analogy',
  example: 'Example',
  key: 'Key idea',
  math: 'Math, gently',
};

// Authoring:  ::: callout analogy "A library with infinite shelves"
//             ...markdown body...
//             :::
module.exports = function calloutPlugin(md) {
  md.use(container, 'callout', {
    validate(params) {
      return /^callout(\s|$)/.test(params.trim());
    },
    render(tokens, idx) {
      const tok = tokens[idx];
      if (tok.nesting === 1) {
        const info = tok.info.trim().replace(/^callout\s*/, '');
        const m = info.match(/^([a-z]+)?\s*(?:"([^"]*)")?/i);
        const type = (m && m[1] && ICONS[m[1]] ? m[1] : 'note');
        const title = (m && m[2]) || DEFAULT_TITLE[type] || 'Note';
        const icon = ICONS[type] || ICONS.note;
        return (
          `<div class="callout callout--${type}" role="note">` +
          `<div class="callout__head"><span class="callout__icon" aria-hidden="true">${icon}</span>` +
          `<span class="callout__title">${md.utils.escapeHtml(title)}</span></div>` +
          `<div class="callout__body">`
        );
      }
      return `</div></div>\n`;
    },
  });
};
