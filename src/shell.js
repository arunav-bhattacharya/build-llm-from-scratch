'use strict';

const esc = (s) => String(s == null ? '' : s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');

// ---- small building blocks -------------------------------------------------

const LOGO = `<svg class="brand__mark" viewBox="0 0 32 32" aria-hidden="true"><defs><linearGradient id="lg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="var(--accent-bright)"/><stop offset="1" stop-color="var(--or)"/></linearGradient></defs><rect x="2" y="2" width="28" height="28" rx="8" fill="url(#lg)"/><text x="16" y="16.5" text-anchor="middle" dominant-baseline="central" font-family="'Google Sans Flex', system-ui, -apple-system, 'Segoe UI', Roboto, Arial, sans-serif" font-weight="700" font-size="11" letter-spacing="-0.4" fill="#ffffff">LLM</text></svg>`;

const CLOCK = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg>`;
const ICON_CARDS = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="3" y="8" width="13" height="13" rx="2"/><path d="M8 8V5a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2h-3"/></svg>`;
const ICON_QUIZ = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M9.2 9.3a2.8 2.8 0 0 1 5.2 1.4c0 1.7-2.4 2-2.4 3.6"/><path d="M12 17.5h.01"/></svg>`;
const ICON_ASSIGN = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 20h9"/><path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4Z"/></svg>`;

function kicker(site, section) {
  const cat = (site.categories && site.categories[section.category]) || null;
  const label = section.group === 'chapter' ? `Chapter ${section.num}` : section.group === 'appendix' ? `Appendix ${section.letter}` : '';
  const stage = cat ? cat.label : '';
  return (
    `<span class="kicker" data-category="${esc(section.category)}">` +
    (label ? `<span class="kicker__num">${esc(label)}</span>` : '') +
    (stage ? `<span class="kicker__stage">${esc(stage)}</span>` : '') +
    `</span>`
  );
}

// A pink meta line at the top of a chapter/appendix: estimated read time plus
// counts of the practice material on the page.
function pageStats(section, stats) {
  const items = [];
  if (section.time) items.push(`<span class="page-stats__item">${CLOCK}~${esc(section.time)} read</span>`);
  if (stats && stats.flashcards) items.push(`<span class="page-stats__item">${ICON_CARDS}${stats.flashcards} flashcards</span>`);
  if (stats && stats.quizQuestions) items.push(`<span class="page-stats__item">${ICON_QUIZ}${stats.quizQuestions} quiz questions</span>`);
  if (stats && stats.assignments) items.push(`<span class="page-stats__item">${ICON_ASSIGN}${stats.assignments} assignment${stats.assignments > 1 ? 's' : ''}</span>`);
  return items.length ? `<div class="page-stats">${items.join('')}</div>` : '';
}

function sidebarNav(sections, current, prefix, categories) {
  const item = (s) => {
    const dir = s.group === 'appendix' ? 'appendices' : 'chapters';
    const url = `${prefix}${dir}/${s.slug}.html`;
    const badge = s.group === 'appendix' ? s.letter : s.num;
    const active = s.slug === current.slug;
    return (
      `<li><a class="nav__link${active ? ' is-active' : ''}" href="${url}"${active ? ' aria-current="page"' : ''} data-category="${esc(s.category)}">` +
      `<span class="nav__badge" data-category="${esc(s.category)}">${esc(badge)}</span>` +
      `<span class="nav__text">${esc(s.title)}</span></a></li>`
    );
  };
  // Group by build stage (same colour-coded grouping as the overview page).
  const order = ['intro', 'build', 'pretrain', 'finetune', 'appendix'];
  let groups = '';
  for (const cat of order) {
    const items = sections.filter((s) => s.category === cat);
    if (!items.length) continue;
    const label = cat === 'appendix' ? 'Appendices' : (categories && categories[cat] && categories[cat].label) || cat;
    groups +=
      `<p class="nav__group" data-category="${esc(cat)}">${esc(label)}</p>` +
      `<ul class="nav__list">${items.map(item).join('')}</ul>`;
  }
  const home = `${prefix}index.html`;
  return (
    `<nav class="nav" aria-label="Chapters and appendices">` +
    `<a class="nav__home${current.group === 'home' ? ' is-active' : ''}" href="${home}">` +
    `<span class="nav__home-ico" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="m3 11 9-8 9 8M5 10v10h5v-6h4v6h5V10"/></svg></span>Overview</a>` +
    groups +
    `</nav>`
  );
}

function tocHtml(toc, section) {
  if (!toc || toc.length < 2) return '';
  const items = toc
    .map((t) => `<li class="toc__item toc__item--h${t.level}"><a href="#${t.slug}" class="toc__link">${esc(t.text)}</a></li>`)
    .join('');
  const num = section.group === 'appendix' ? section.letter : section.num;
  const chap = num ? `<div class="toc__chap"><span class="toc__chap-num">${esc(num)}</span></div>` : '';
  return (
    `<aside class="toc" aria-label="On this page"><div class="toc__inner">` +
    chap +
    `<p class="toc__title">On this page</p><ul class="toc__list">${items}</ul></div></aside>`
  );
}

function pager(prevNext, prefix) {
  if (!prevNext) return '';
  const link = (s, rel) => {
    if (!s) return `<span class="pager__slot"></span>`;
    const url = `@/${s.group === 'appendix' ? 'appendices' : 'chapters'}/${s.slug}.html`.replace('@/', prefix);
    const lbl = s.group === 'appendix' ? `Appendix ${s.letter}` : `Chapter ${s.num}`;
    return (
      `<a class="pager__slot pager__slot--${rel}" href="${url}">` +
      `<span class="pager__dir">${rel === 'prev' ? '← Previous' : 'Next →'}</span>` +
      `<span class="pager__name">${esc(lbl)} · ${esc(s.title)}</span></a>`
    );
  };
  return `<nav class="pager" aria-label="Chapter navigation">${link(prevNext.prev, 'prev')}${link(prevNext.next, 'next')}</nav>`;
}

function searchModal() {
  return (
    `<div class="search" id="search" hidden aria-hidden="true">` +
    `<div class="search__scrim" data-search-close></div>` +
    `<div class="search__panel" role="dialog" aria-modal="true" aria-label="Search the companion">` +
    `<div class="search__bar"><span class="search__ico" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.2-3.2"/></svg></span>` +
    `<input type="search" class="search__input" placeholder="Search concepts, code, chapters…" aria-label="Search" autocomplete="off" spellcheck="false">` +
    `<button class="search__esc" type="button" data-search-close aria-label="Close search">Esc</button></div>` +
    `<div class="search__results" data-search-results><p class="search__hint">Start typing to search across every chapter.</p></div>` +
    `</div></div>`
  );
}

const themeToggle =
  `<button class="iconbtn theme-toggle" type="button" data-theme-toggle aria-label="Toggle dark mode">` +
  `<span class="theme-toggle__sun" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="4.5"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg></span>` +
  `<span class="theme-toggle__moon" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M20 14.5A8 8 0 1 1 9.5 4 6.5 6.5 0 0 0 20 14.5Z"/></svg></span></button>`;

const navToggle =
  `<button class="iconbtn nav-toggle" type="button" data-nav-toggle aria-label="Toggle navigation" aria-expanded="false">` +
  `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round"><path d="M3 6h18M3 12h18M3 18h18"/></svg></button>`;

const searchTrigger =
  `<button class="search-trigger" type="button" data-search-open aria-label="Search">` +
  `<span class="search-trigger__ico" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.2-3.2"/></svg></span>` +
  `<span class="search-trigger__txt">Search</span><kbd class="search-trigger__kbd">⌘K</kbd></button>`;

// Shared SVG paint defs (injected once per page). Diagram shapes reference these
// gradients via CSS `fill: url(#g-…)`; the stops use theme tokens, so diagrams
// recolour automatically on light/dark switch.
const SVG_DEFS =
  `<svg class="svg-defs" aria-hidden="true" focusable="false" width="0" height="0"><defs>` +
  `<linearGradient id="g-node" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="var(--svg-node)"/><stop offset="1" stop-color="var(--svg-node-lo)"/></linearGradient>` +
  `<linearGradient id="g-node2" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="var(--svg-node-2)"/><stop offset="1" stop-color="var(--svg-node-2-lo)"/></linearGradient>` +
  `<linearGradient id="g-soft" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="var(--svg-soft-hi)"/><stop offset="1" stop-color="var(--svg-soft-lo)"/></linearGradient>` +
  `<linearGradient id="g-accent" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="var(--svg-accent-hi)"/><stop offset="1" stop-color="var(--svg-accent)"/></linearGradient>` +
  `<linearGradient id="g-chip" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="var(--svg-chip)"/><stop offset="1" stop-color="var(--svg-chip-lo)"/></linearGradient>` +
  `</defs></svg>`;

// ---- the page document -----------------------------------------------------

function renderPage(opts) {
  const { site, section, sections, bodyHtml, toc = [], prefix, prevNext, scripts = [], stats } = opts;
  const isHome = section.group === 'home';
  const titleText = isHome ? `${site.site.title}` : `${section.title} · ${site.site.shortTitle}`;
  const desc = section.desc || site.site.tagline;

  const css = ['tokens', 'base', 'shell', 'components', 'shiki']
    .map((n) => `<link rel="stylesheet" href="${prefix}css/${n}.css">`)
    .join('') + `<link rel="stylesheet" href="${prefix}css/katex.min.css">`;

  const head =
    `<!doctype html><html lang="en"><head>` +
    `<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">` +
    `<title>${esc(titleText)}</title>` +
    `<meta name="description" content="${esc(desc)}">` +
    `<meta name="color-scheme" content="light dark">` +
    `<meta name="theme-color" content="#f6f7f9">` +
    // no-FOUC: set theme + sidebar state before first paint
    `<script>(function(){try{var d=document.documentElement,ls=localStorage,t=ls.getItem('llmbook:theme');if(!t)t=matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light';d.setAttribute('data-theme',t);if(ls.getItem('llmbook:sidebar')==='collapsed')d.setAttribute('data-sidebar','collapsed');}catch(e){document.documentElement.setAttribute('data-theme','light');}})();</script>` +
    `<script>window.__BASE__=${JSON.stringify(prefix)};</script>` +
    `<link rel="preload" as="font" type="font/woff2" crossorigin href="${prefix}fonts/google-sans-flex/GoogleSansFlex-latin.woff2">` +
    `<link rel="icon" href="${prefix}favicon.svg">` +
    css +
    `</head>`;

  const brandHome = `${prefix}index.html`;
  const topbar =
    `<header class="topbar">` +
    `<div class="topbar__left">${isHome ? '' : navToggle}` +
    `<a class="brand" href="${brandHome}" aria-label="${esc(site.site.title)} — home">${LOGO}<span class="brand__name">${esc(site.site.shortTitle)}</span></a></div>` +
    `<div class="topbar__center">${searchTrigger}</div>` +
    `<div class="topbar__right">${themeToggle}</div>` +
    `</header>`;

  const sidebar =
    `<aside class="sidebar" id="sidebar">` +
    `<div class="sidebar__inner">` +
    `<a class="sidebar__book" href="${brandHome}"><span class="sidebar__book-title">${esc(site.site.book.title)}</span>` +
    `<span class="sidebar__book-by">by ${esc(site.site.book.author)}</span></a>` +
    sidebarNav(sections, section, prefix, site.categories) +
    `</div></aside>` +
    `<button class="sidebar-reopen" type="button" data-nav-toggle aria-label="Open navigation"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m9 18 6-6-6-6"/></svg></button>`;

  const pageHead = isHome
    ? ''
    : `<header class="page-head">${kicker(site, section)}<h1 class="page-title">${esc(section.title)}</h1>` +
      (section.desc ? `<p class="page-lede">${esc(section.desc)}</p>` : '') + pageStats(section, stats) + `</header>`;

  const article = `<article class="prose">${pageHead}${bodyHtml}${isHome ? '' : pager(prevNext, prefix)}</article>`;
  const rightToc = isHome ? '' : tocHtml(toc, section);
  const layoutClass = isHome ? 'layout--home' : 'layout' + (rightToc ? '' : ' layout--notoc');

  const footer =
    `<footer class="footer"><div class="footer__inner">` +
    `<p class="footer__main">An unofficial, free study companion for <a href="${esc(site.site.book.repo)}" target="_blank" rel="noopener noreferrer"><em>${esc(site.site.book.title)}</em></a> by ${esc(site.site.book.author)} (${esc(site.site.book.publisher)}, ${esc(site.site.book.year)}).</p>` +
    `<p class="footer__sub">Summaries are original explanations; code examples are adapted from the author's open-source <a href="${esc(site.site.book.repo)}" target="_blank" rel="noopener noreferrer">LLMs-from-scratch</a> repository. Please support the author by buying the book.</p>` +
    `</div></footer>`;

  const scriptTags = ['theme', 'nav', 'code', 'flashcards', 'quiz', 'tabs', 'minisearch', 'search', ...scripts]
    .map((n) => `<script defer src="${prefix}js/${n}.js"></script>`)
    .join('');
  const searchIndex = `<script defer src="${prefix}search-index.js"></script>`;

  const body =
    `<body data-category="${esc(section.category || 'intro')}">` +
    SVG_DEFS +
    `<a class="skip-link" href="#main">Skip to content</a>` +
    topbar +
    `<div class="${layoutClass}">` +
    (isHome ? '' : sidebar) +
    `<main class="main" id="main">${article}${footer}</main>` +
    rightToc +
    `</div>` +
    `<div class="drawer-scrim" data-nav-close hidden></div>` +
    searchModal() +
    searchIndex +
    scriptTags +
    `</body></html>`;

  return head + body;
}

module.exports = { renderPage };
