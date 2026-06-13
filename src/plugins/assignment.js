'use strict';

// Renders a ```assignment fenced block as a "challenge" card.
// Authoring:
//   ```assignment "Re-implement causal masking" level=intermediate
//   Without peeking at section 3.5, write a function that masks the upper triangle…
//
//   Hint: think about `torch.triu`.
//   ```
// Lines beginning "Hint:" become collapsible hints; everything else is markdown body.

const LEVELS = { intro: 'Intro', beginner: 'Beginner', intermediate: 'Intermediate', advanced: 'Advanced' };

function parseInfo(info) {
  const titleM = info.match(/"([^"]*)"/);
  const levelM = info.match(/level=([a-z]+)/i);
  return {
    title: titleM ? titleM[1] : 'Try it yourself',
    level: levelM && LEVELS[levelM[1].toLowerCase()] ? LEVELS[levelM[1].toLowerCase()] : '',
  };
}

function renderAssignment(raw, info, md) {
  const { title, level } = parseInfo(info);
  const bodyLines = [];
  const hints = [];
  for (const line of raw.split('\n')) {
    const h = line.match(/^\s*Hint:\s?(.*)$/i);
    if (h) hints.push(h[1]);
    else bodyLines.push(line);
  }
  const body = md.render(bodyLines.join('\n').trim());
  const hintsHtml = hints.length
    ? `<details class="assignment__hints"><summary>Show ${hints.length} hint${hints.length === 1 ? '' : 's'}</summary>` +
      `<ul>${hints.map((h) => `<li>${md.renderInline(h)}</li>`).join('')}</ul></details>`
    : '';
  const badge =
    `<span class="assignment__badge" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="m12 2 2.4 6.9L21 9.3l-5 4.7L17.8 21 12 17.3 6.2 21 8 14l-5-4.7 6.6-.4Z"/></svg></span>`;
  const levelHtml = level ? `<span class="assignment__level">${level}</span>` : '';
  return `
  <div class="assignment">
    <div class="assignment__head">${badge}<span class="assignment__kicker">Challenge</span><h4 class="assignment__title">${md.utils.escapeHtml(title)}</h4>${levelHtml}</div>
    <div class="assignment__body">${body}</div>
    ${hintsHtml}
  </div>`;
}

module.exports = { renderAssignment };
