'use strict';

const MarkdownIt = require('markdown-it');
const texmath = require('markdown-it-texmath');
const katex = require('katex');

const calloutPlugin = require('./plugins/callout');
const diagramPlugin = require('./plugins/diagram');
const fencePlugin = require('./plugins/fence');
const blocksPlugin = require('./plugins/blocks');

function slugify(s) {
  return (
    s
      .toLowerCase()
      .trim()
      .replace(/`/g, '')
      .replace(/[^\w\s-]/g, '')
      .replace(/\s+/g, '-')
      .replace(/-+/g, '-')
      .replace(/^-|-$/g, '') || 'section'
  );
}

// Core rule: give every heading a stable, unique id and collect an on-this-page TOC.
function headingAnchors(md) {
  md.core.ruler.push('heading_anchors', function (state) {
    const seen = {};
    const toc = (state.env.toc = state.env.toc || []);
    const tokens = state.tokens;
    for (let i = 0; i < tokens.length; i++) {
      if (tokens[i].type !== 'heading_open') continue;
      const level = Number(tokens[i].tag.slice(1));
      const inline = tokens[i + 1];
      const text = inline && inline.type === 'inline' ? inline.content : '';
      let base = slugify(text);
      let slug = base;
      let n = 1;
      while (seen[slug]) slug = `${base}-${++n}`;
      seen[slug] = true;
      tokens[i].attrSet('id', slug);
      if (level === 2 || level === 3) toc.push({ level, text, slug });
    }
  });
}

// Open external links in a new tab safely.
function externalLinks(md) {
  const base = md.renderer.rules.link_open || ((t, i, o, e, s) => s.renderToken(t, i, o));
  md.renderer.rules.link_open = function (tokens, idx, options, env, self) {
    const href = tokens[idx].attrGet('href') || '';
    if (/^https?:\/\//i.test(href)) {
      tokens[idx].attrSet('target', '_blank');
      tokens[idx].attrSet('rel', 'noopener noreferrer');
      tokens[idx].attrJoin('class', 'ext-link');
    }
    return base(tokens, idx, options, env, self);
  };
}

// Wrap tables so they can scroll horizontally on small screens.
function responsiveTables(md) {
  const open = md.renderer.rules.table_open || ((t, i, o, e, s) => s.renderToken(t, i, o));
  const close = md.renderer.rules.table_close || ((t, i, o, e, s) => s.renderToken(t, i, o));
  md.renderer.rules.table_open = (t, i, o, e, s) => `<div class="table-wrap">` + open(t, i, o, e, s);
  md.renderer.rules.table_close = (t, i, o, e, s) => close(t, i, o, e, s) + `</div>`;
}

function buildMarkdown({ highlighter, getSvg, warn }) {
  const md = new MarkdownIt({
    html: true, // trusted, authored content (inline SVG / raw HTML allowed)
    linkify: true,
    typographer: true,
    breaks: false,
  });

  md.use(texmath, {
    engine: katex,
    delimiters: 'dollars',
    katexOptions: { throwOnError: false, strict: false },
  });

  headingAnchors(md);
  externalLinks(md);
  responsiveTables(md);

  calloutPlugin(md);
  blocksPlugin(md);
  diagramPlugin(md, { getSvg, warn });
  fencePlugin(md, { highlighter });

  return md;
}

module.exports = { buildMarkdown, slugify };
