# Build a Large Language Model (From Scratch) — Interactive Study Companion

A free, **unofficial** study companion for Sebastian Raschka's book
[*Build a Large Language Model (From Scratch)*](https://www.manning.com/books/build-a-large-language-model-from-scratch).
It re-explains every chapter and appendix for beginners and visual learners — with
real-world analogies, hand-drawn diagrams, full explained code, key takeaways,
researched references, and interactive flashcards / quizzes / assignments.

The site is a **custom static site** (no framework). Content is authored in Markdown
and compiled to plain HTML that runs **both** opened directly from disk (`file://`)
**and** hosted on GitHub Pages or any static host.

## Quick start

```bash
npm install        # install build dependencies
npm run build      # compile content/ + static/ -> dist/
npm run serve      # serve dist/ at http://localhost:8080
# or: npm run dev  (build, then serve)
```

Then open <http://localhost:8080>, or just open `dist/index.html` in a browser.

> Requires Node.js 18+. The build uses `markdown-it`, `shiki` (syntax highlighting),
> `katex` (math), `gray-matter`, and `minisearch` (search). Fonts (Google Sans Flex/Code,
> OFL-1.1) and KaTeX assets are vendored under `static/fonts/` and `static/css/`.

## Project layout

```
content/              # ← author here (Markdown only)
  _meta.json          #   source of truth: the ordered list of sections
  index.md            #   landing page
  ch0X-*.md            #   chapters 1–7
  appX-*.md            #   appendices A–E
assets/diagrams/*.svg # hand-authored, theme-aware SVG diagrams
static/               # copied verbatim into dist/ (css, js, fonts, favicon)
src/                  # the Node build pipeline (build.js, markdown.js, shell.js, plugins/…)
dist/                 # build output — open or deploy this (git-ignored)
```

## Authoring content

Each page is Markdown with a few custom block directives:

| Syntax | Renders as |
|---|---|
| `::: objectives "What you'll learn"` … `:::` | a "what you'll learn" card |
| `::: callout tip\|note\|warning\|analogy\|example\|key\|math "Title"` … `:::` | a styled callout |
| `::: diagram <svg-name> "Caption"` … `:::` | inlines `assets/diagrams/<svg-name>.svg` |
| `::: takeaways` / `::: refs` … `:::` | key-takeaways / reference cards |
| ` ```python title="…" collapsible ` | syntax-highlighted, copyable code |
| ` ```flashcards ` (`Q:` / `A:` lines, cards split by `---`) | flip-card deck |
| ` ```quiz ` (`- ( )` / `- (x)` options, `> explanation`) | interactive quiz |
| ` ```assignment "Title" level=… ` | a challenge card |
| `$…$` / `$$…$$` | KaTeX math |

Diagrams are inline SVGs that use a small set of theme-aware classes (`d-node`, `d-soft`,
`d-chip`, `d-line`, …) so they recolor with the light/dark theme. See any
`assets/diagrams/ch03-*.svg` for the conventions.

## Deploying to GitHub Pages

The build uses **relative paths only** (no `<base href>`, no absolute `/` paths) and
inlines the search index, so the same `dist/` works at a repository sub-path.

- **Option A (CI):** add a GitHub Action that runs `npm ci && npm run build` and deploys
  `dist/` to Pages.
- **Option B (no CI):** run `npm run build` locally and publish the `dist/` folder
  (e.g. push it to a `gh-pages` branch). A `.nojekyll` file is emitted automatically.

## Credits & license

Explanations on this site are original. Code examples are adapted from the author's
open-source [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) repository.
Please [buy the book](https://www.manning.com/books/build-a-large-language-model-from-scratch)
to support the author. Site code is MIT-licensed; bundled fonts are OFL-1.1.
