'use strict';

const fs = require('fs');
const path = require('path');
const matter = require('gray-matter');

const { getHighlighter } = require('./highlight');
const { buildMarkdown } = require('./markdown');
const { renderPage } = require('./shell');
const { relPrefix, resolveLinks } = require('./paths');
const { buildSearchScript } = require('./search-index');

const ROOT = path.resolve(__dirname, '..');
const CONTENT = path.join(ROOT, 'content');
const DIAGRAMS = path.join(ROOT, 'assets', 'diagrams');
const STATIC = path.join(ROOT, 'static');
const DIST = path.join(ROOT, 'dist');

const warnings = [];
const warn = (m) => warnings.push(m);

function outPathFor(section) {
  if (section.group === 'home') return 'index.html';
  if (section.group === 'appendix') return `appendices/${section.slug}.html`;
  return `chapters/${section.slug}.html`;
}

function htmlToText(html) {
  return html
    .replace(/<template[\s\S]*?<\/template>/gi, ' ')
    .replace(/<pre[\s\S]*?<\/pre>/gi, ' ')
    .replace(/<svg[\s\S]*?<\/svg>/gi, ' ')
    .replace(/<script[\s\S]*?<\/script>/gi, ' ')
    .replace(/<style[\s\S]*?<\/style>/gi, ' ')
    .replace(/<[^>]+>/g, ' ')
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&nbsp;/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

// Split a rendered page into per-section search documents (deep-linkable by heading id).
function sectionDocs(pageHtml, section, pageUrl) {
  const docs = [];
  const re = /<h2 id="([^"]+)"[^>]*>([\s\S]*?)<\/h2>/g;
  const marks = [];
  let m;
  while ((m = re.exec(pageHtml))) marks.push({ id: m[1], title: htmlToText(m[2]), start: m.index });

  const introEnd = marks.length ? marks[0].start : pageHtml.length;
  const introText = htmlToText(pageHtml.slice(0, introEnd));
  if (introText) {
    docs.push({
      id: `${section.slug}#_top`, title: section.title, chapterTitle: section.title,
      url: pageUrl, category: section.category, body: introText, snippet: introText.slice(0, 170),
    });
  }
  for (let i = 0; i < marks.length; i++) {
    const start = marks[i].start;
    const end = i + 1 < marks.length ? marks[i + 1].start : pageHtml.length;
    const body = htmlToText(pageHtml.slice(start, end));
    docs.push({
      id: `${section.slug}#${marks[i].id}`, title: marks[i].title || section.title, chapterTitle: section.title,
      url: `${pageUrl}#${marks[i].id}`, category: section.category, body, snippet: body.slice(0, 170),
    });
  }
  return docs;
}

function esc(s) {
  return String(s == null ? '' : s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

const CLOCK_SVG = '<svg class="ico-clock" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg>';

// Auto-generated grid of section cards for the landing page (injected at <!--SECTION_GRID-->).
// Chapters and appendices are rendered as two separate, labeled grids.
function sectionGridHtml(sections, categories) {
  const card = (s) => {
    const dir = s.group === 'appendix' ? 'appendices' : 'chapters';
    const url = `${dir}/${s.slug}.html`;
    const badge = s.group === 'appendix' ? s.letter : s.num;
    const label = s.group === 'appendix' ? `Appendix ${s.letter}` : `Chapter ${s.num}`;
    const stage = (categories[s.category] && categories[s.category].label) || '';
    const time = s.time ? `<span class="sec-card__time">${CLOCK_SVG}~${esc(s.time)}</span>` : '';
    return (
      `<a class="sec-card" href="${url}" data-category="${esc(s.category)}">` +
      `<div class="sec-card__top"><span class="sec-card__badge" data-category="${esc(s.category)}">${esc(badge)}</span>` +
      `<span class="sec-card__kicker">${esc(label)} · ${esc(stage)}</span></div>` +
      `<div class="sec-card__title">${esc(s.title)}</div>` +
      `<div class="sec-card__desc">${esc(s.desc || '')}</div>` +
      `<div class="sec-card__foot">${time}</div></a>`
    );
  };
  const chapters = sections.filter((s) => s.group === 'chapter').map(card).join('');
  const apps = sections.filter((s) => s.group === 'appendix').map(card).join('');
  return (
    `<h3 class="home-h3">Chapters</h3><div class="sec-grid">${chapters}</div>` +
    `<h3 class="home-h3">Appendices</h3><div class="sec-grid">${apps}</div>`
  );
}

function loadDiagrams() {
  const map = {};
  if (!fs.existsSync(DIAGRAMS)) return map;
  for (const f of fs.readdirSync(DIAGRAMS)) {
    if (f.endsWith('.svg')) map[f.replace(/\.svg$/, '')] = fs.readFileSync(path.join(DIAGRAMS, f), 'utf8').trim();
  }
  return map;
}

async function build() {
  const t0 = Date.now();
  console.log('· cleaning dist/');
  fs.rmSync(DIST, { recursive: true, force: true });
  fs.mkdirSync(DIST, { recursive: true });

  console.log('· copying static assets');
  fs.cpSync(STATIC, DIST, { recursive: true });
  const nojekyll = path.join(ROOT, '.nojekyll');
  if (fs.existsSync(nojekyll)) fs.copyFileSync(nojekyll, path.join(DIST, '.nojekyll'));

  console.log('· initializing Shiki');
  const highlighter = await getHighlighter();

  const diagrams = loadDiagrams();
  const getSvg = (name) => diagrams[name] || null;
  const md = buildMarkdown({ highlighter, getSvg, warn });

  const meta = JSON.parse(fs.readFileSync(path.join(CONTENT, '_meta.json'), 'utf8'));
  const sections = meta.sections;
  const navSections = sections.filter((s) => s.group !== 'home');

  // ---- pass 1: render each body, collect TOC + search docs ----
  console.log('· rendering content');
  const rendered = [];
  let searchDocs = [];
  for (const section of sections) {
    const file = path.join(CONTENT, `${section.slug}.md`);
    if (!fs.existsSync(file)) {
      if (section.slug !== 'index') warn(`Missing content file: content/${section.slug}.md`);
      continue;
    }
    const env = { toc: [], counter: {} };
    const parsed = matter(fs.readFileSync(file, 'utf8'));
    let bodyHtml = md.render(parsed.content, env);
    if (section.group === 'home') bodyHtml = bodyHtml.replace('<!--SECTION_GRID-->', sectionGridHtml(sections, meta.categories));
    const out = outPathFor(section);
    const pageUrl = out; // root-relative; search.js prepends window.__BASE__
    const stats = {
      flashcards: (bodyHtml.match(/class="flashcard"/g) || []).length,
      quizQuestions: (bodyHtml.match(/class="quiz__q"/g) || []).length,
      assignments: (bodyHtml.match(/class="assignment"/g) || []).length,
    };
    rendered.push({ section, bodyHtml, toc: env.toc, out, stats });
    if (section.group !== 'home') searchDocs = searchDocs.concat(sectionDocs(bodyHtml, section, pageUrl));
  }

  // ---- search index ----
  console.log(`· indexing ${searchDocs.length} sections for search`);
  const searchScript = buildSearchScript(searchDocs);
  fs.writeFileSync(path.join(DIST, 'search-index.js'), searchScript);

  // ---- pass 2: wrap in shell + write ----
  console.log('· writing pages');
  for (let i = 0; i < rendered.length; i++) {
    const { section, bodyHtml, toc, out, stats } = rendered[i];
    const prefix = relPrefix(out);
    let prevNext = null;
    if (section.group !== 'home') {
      const idx = navSections.findIndex((s) => s.slug === section.slug);
      prevNext = { prev: navSections[idx - 1] || null, next: navSections[idx + 1] || null };
    }
    let html = renderPage({ site: meta, section, sections: navSections, bodyHtml, toc, prefix, prevNext, stats });
    html = resolveLinks(html, prefix); // resolve any @/ links in authored content
    const dest = path.join(DIST, out);
    fs.mkdirSync(path.dirname(dest), { recursive: true });
    fs.writeFileSync(dest, html);
  }

  const indexKB = (Buffer.byteLength(searchScript) / 1024).toFixed(1);
  const dt = ((Date.now() - t0) / 1000).toFixed(2);
  console.log(`\n✓ built ${rendered.length} pages in ${dt}s  ·  search index ${indexKB} KB  ·  ${Object.keys(diagrams).length} diagrams`);
  if (warnings.length) {
    console.log(`\n⚠ ${warnings.length} warning(s):`);
    warnings.forEach((w) => console.log('   - ' + w));
  } else {
    console.log('  no warnings');
  }
}

build().catch((e) => {
  console.error('\n✗ build failed:\n', e);
  process.exit(1);
});
