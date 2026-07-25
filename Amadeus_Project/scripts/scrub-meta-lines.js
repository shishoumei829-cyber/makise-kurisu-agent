'use strict';

/**
 * 从 amadeus_data 清掉「刚才那句不算」等产品元话语污染。
 * usage: node scripts/scrub-meta-lines.js --dir <amadeus_data>
 */

const fs = require('fs');
const path = require('path');

const index = process.argv.indexOf('--dir');
const dataDir = path.resolve(index >= 0 ? process.argv[index + 1] : '');
if (!dataDir || !fs.existsSync(dataDir)) {
  throw new Error('usage: node scripts/scrub-meta-lines.js --dir <amadeus_data>');
}

const META = /刚才那句不算|我重新说|说重点。|听着呢。|有事？/;

function scrubText(value) {
  return String(value || '')
    .replace(/[…\.．]*\s*刚才那句不算[，,]?\s*(?:我)?重新说[。.!！]?/g, '')
    .replace(/刚才那句不算[，,]?/g, '')
    .replace(/(?:那句)?不算[，,]?\s*我重新说[。.!！]?/g, '')
    .trim();
}

function isBadLine(text) {
  const t = String(text || '').trim();
  if (!t) return true;
  if (/刚才那句不算/.test(t)) return true;
  if (/^……?刚才那句不算/.test(t)) return true;
  if (/^(?:说重点|怎么了|有事|听着呢|讲)[。.!！?？]?$/.test(t.replace(/\s+/g, ''))) return true;
  return false;
}

const results = [];

// unified_dialogue_log.json
{
  const file = path.join(dataDir, 'unified_dialogue_log.json');
  if (fs.existsSync(file)) {
    const before = fs.readFileSync(file, 'utf8');
    const hits = (before.match(/刚才那句不算/g) || []).length;
    if (hits) {
      const backup = `${file}.backup-meta-${Date.now()}`;
      fs.copyFileSync(file, backup);
      const data = JSON.parse(before);
      const entries = Array.isArray(data.entries) ? data.entries : [];
      data.entries = entries
        .map((e) => ({ ...e, text: scrubText(e.text) }))
        .filter((e) => !isBadLine(e.text));
      fs.writeFileSync(file, `${JSON.stringify(data, null, 2)}\n`, 'utf8');
      results.push({ file, backup, removedOrScrubbed: hits, remaining: (fs.readFileSync(file, 'utf8').match(/刚才那句不算/g) || []).length });
    }
  }
}

// butler/events.jsonl
{
  const file = path.join(dataDir, 'butler', 'events.jsonl');
  if (fs.existsSync(file)) {
    const before = fs.readFileSync(file, 'utf8');
    const hits = (before.match(/刚才那句不算/g) || []).length;
    if (hits) {
      const backup = `${file}.backup-meta-${Date.now()}`;
      fs.copyFileSync(file, backup);
      const lines = before.split(/\r?\n/).filter(Boolean);
      const next = [];
      for (const line of lines) {
        try {
          const ev = JSON.parse(line);
          const text = ev?.payload?.text;
          if (typeof text === 'string' && isBadLine(text)) continue;
          if (typeof text === 'string') {
            ev.payload.text = scrubText(text);
            if (!ev.payload.text) continue;
          }
          next.push(JSON.stringify(ev));
        } catch {
          next.push(line);
        }
      }
      fs.writeFileSync(file, `${next.join('\n')}\n`, 'utf8');
      results.push({ file, backup, removedOrScrubbed: hits, remaining: (fs.readFileSync(file, 'utf8').match(/刚才那句不算/g) || []).length });
    }
  }
}

// memory_admission quarantine meta fragments longer
{
  const file = path.join(dataDir, 'memory_admission.json');
  if (fs.existsSync(file)) {
    const data = JSON.parse(fs.readFileSync(file, 'utf8'));
    data.quarantined = data.quarantined || {};
    const until = Date.now() + 30 * 86400000;
    for (const frag of ['刚才那句不算', '我重新说', '那句不算', '不算重新说', '说重点', '听着呢']) {
      data.quarantined[frag] = until;
    }
    fs.writeFileSync(file, `${JSON.stringify(data, null, 2)}\n`, 'utf8');
    results.push({ file, quarantined: true });
  }
}

console.log(JSON.stringify({ dataDir, meta: META.source, results }, null, 2));
