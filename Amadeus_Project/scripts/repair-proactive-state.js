'use strict';

const fs = require('fs');
const path = require('path');

const index = process.argv.indexOf('--dir');
const dataDir = path.resolve(index >= 0 ? process.argv[index + 1] : '');
if (!dataDir || !fs.existsSync(dataDir)) {
  throw new Error('usage: node scripts/repair-proactive-state.js --dir <amadeus_data>');
}

const polluted = /刷抖音|抖音这种事|毫无意义的东西|压缩为≤20字标签|NO_CONFLICT|已有：.*\n新：/i;

function clean(value) {
  if (Array.isArray(value)) {
    return value
      .filter((item) => !polluted.test(JSON.stringify(item)))
      .map(clean);
  }
  if (!value || typeof value !== 'object') return value;
  const out = {};
  for (const [key, child] of Object.entries(value)) {
    if (polluted.test(key)) continue;
    if (typeof child === 'string' && polluted.test(child)) continue;
    out[key] = clean(child);
  }
  return out;
}

const files = [
  'autonomy_subsystem.json',
  'user_model.json',
  'understanding_subsystem.json',
  'unified_dialogue_log.json',
];
const results = [];
for (const name of files) {
  const file = path.join(dataDir, name);
  if (!fs.existsSync(file)) continue;
  const beforeText = fs.readFileSync(file, 'utf8');
  const beforeHits = (beforeText.match(new RegExp(polluted.source, 'gi')) || []).length;
  if (!beforeHits) continue;
  const backup = `${file}.backup-${Date.now()}`;
  fs.copyFileSync(file, backup);
  const next = clean(JSON.parse(beforeText));
  const afterText = `${JSON.stringify(next, null, 2)}\n`;
  fs.writeFileSync(file, afterText, 'utf8');
  const afterHits = (afterText.match(new RegExp(polluted.source, 'gi')) || []).length;
  results.push({ file, backup, beforeHits, afterHits });
}

console.log(JSON.stringify({ dataDir, results }));
