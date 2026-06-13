'use strict';

// Compute the relative path prefix for a page given its output path (relative to dist/).
//   'index.html'                 -> ''      (root)
//   'chapters/ch01-….html'       -> '../'   (one level deep)
//   'appendices/appA-….html'     -> '../'
// Used so a single build works on file://, user.github.io, and user.github.io/repo/.
function relPrefix(outPath) {
  const depth = outPath.split('/').length - 1;
  return '../'.repeat(depth);
}

// Resolve site-root-relative links authored as "@/path" into a page-correct relative URL.
// Anchor-only links ("#id") and external links are left untouched.
function resolveLinks(html, prefix) {
  return html.replace(/(href|src)="@\/([^"]*)"/g, (_, attr, path) => `${attr}="${prefix}${path}"`);
}

module.exports = { relPrefix, resolveLinks };
