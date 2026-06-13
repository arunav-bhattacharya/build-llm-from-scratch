'use strict';

const MiniSearch = require('minisearch');

const OPTS = {
  fields: ['title', 'body'],
  storeFields: ['title', 'chapterTitle', 'url', 'category', 'snippet'],
};

// Build a MiniSearch index at compile time and serialize it. The browser rehydrates
// with MiniSearch.loadJSON (no re-indexing). Emitted as a classic-script global so it
// loads over file:// without any fetch().
function buildSearchScript(docs) {
  const mini = new MiniSearch(OPTS);
  mini.addAll(docs);
  const json = JSON.stringify(mini);
  return (
    `window.__SEARCH_INDEX__=${JSON.stringify(json)};\n` +
    `window.__SEARCH_OPTS__=${JSON.stringify(OPTS)};\n`
  );
}

module.exports = { buildSearchScript };
