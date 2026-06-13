'use strict';

// Renders a fenced code block: Shiki dual-theme highlighting (zero runtime JS),
// a title/lang bar, a copy button (copies the RAW pre-highlight source from a
// <template>, preserving exact indentation), and optional collapsibility.

function escapeHtml(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

const LANG_ALIAS = { py: 'python', sh: 'bash', shell: 'bash', console: 'bash', txt: 'text', plaintext: 'text', '': 'text' };
const LANG_LABEL = { python: 'Python', bash: 'Shell', text: 'Text', json: 'JSON', yaml: 'YAML', diff: 'Diff' };

function parseInfo(info) {
  const first = info.split(/\s+/)[0] || '';
  const lang = (LANG_ALIAS[first.toLowerCase()] || first.toLowerCase() || 'text');
  const titleM = info.match(/title="([^"]*)"/);
  const collapsible = /\bcollapsible\b/.test(info);
  return { lang, title: titleM ? titleM[1] : '', collapsible };
}

function renderCode(highlighter, rawCode, info, md) {
  const { lang, title, collapsible } = parseInfo(info);
  const code = rawCode.replace(/\n$/, '');
  let shikiLang = lang;
  try {
    if (!highlighter.getLoadedLanguages().includes(shikiLang)) shikiLang = 'text';
  } catch (_) {
    shikiLang = 'text';
  }

  let highlighted;
  try {
    highlighted = highlighter.codeToHtml(code, {
      lang: shikiLang,
      themes: { light: 'github-light', dark: 'github-dark' },
      defaultColor: false, // emit both colors as CSS vars; bridge in shiki.css
    });
  } catch (e) {
    highlighted = `<pre class="shiki"><code>${escapeHtml(code)}</code></pre>`;
  }

  const label = title || LANG_LABEL[shikiLang] || shikiLang;
  const source = `<template class="codeblock__source">${escapeHtml(code)}</template>`;

  const copyBtn =
    `<button class="codeblock__btn" type="button" data-copy aria-label="Copy code to clipboard">` +
    `<span class="codeblock__btn-ico" aria-hidden="true">` +
    `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h10"/></svg></span>` +
    `<span class="codeblock__btn-txt">Copy</span></button>`;

  const toggleBtn = collapsible
    ? `<button class="codeblock__btn codeblock__toggle" type="button" data-code-toggle aria-expanded="false">` +
      `<span class="codeblock__btn-txt">Show code</span></button>`
    : '';

  const bar =
    `<div class="codeblock__bar">` +
    `<span class="codeblock__lang">${md.utils.escapeHtml(label)}</span>` +
    `<span class="codeblock__actions">${toggleBtn}${copyBtn}</span>` +
    `</div>`;

  const scrollCls = 'codeblock__scroll' + (collapsible ? ' is-collapsed' : '');
  const body = `<div class="${scrollCls}">${highlighted}</div>`;

  const cls = 'codeblock' + (collapsible ? ' codeblock--collapsible' : '');
  return `<div class="${cls}"${collapsible ? ' data-collapsible' : ''}>${source}${bar}${body}</div>`;
}

module.exports = { renderCode, escapeHtml };
