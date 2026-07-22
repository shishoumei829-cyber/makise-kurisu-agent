'use strict';

const fs = require('fs');
const path = require('path');

function arg(name) {
  const index = process.argv.indexOf(`--${name}`);
  return index >= 0 ? process.argv[index + 1] : '';
}

const file = path.resolve(arg('file') || '');
const fromTs = Number(arg('from'));
const toTs = Number(arg('to') || Number.MAX_SAFE_INTEGER);
if (!file || !Number.isFinite(fromTs) || !Number.isFinite(toTs)) {
  throw new Error('usage: node scripts/repair-dialogue-log.js --file <json> --from <timestamp> [--to <timestamp>]');
}

const raw = JSON.parse(fs.readFileSync(file, 'utf8'));
const entries = Array.isArray(raw) ? raw : raw.entries;
if (!Array.isArray(entries)) throw new Error('dialogue log has no entries array');

const kept = entries.filter((entry) => {
  const ts = Number(entry?.ts);
  return !Number.isFinite(ts) || ts < fromTs || ts > toTs;
});
const removed = entries.length - kept.length;
const backup = `${file}.backup-${Date.now()}`;
fs.copyFileSync(file, backup);
const next = Array.isArray(raw) ? kept : { ...raw, entries: kept };
fs.writeFileSync(file, `${JSON.stringify(next, null, 2)}\n`, 'utf8');
console.log(JSON.stringify({ file, backup, removed, remaining: kept.length }));
