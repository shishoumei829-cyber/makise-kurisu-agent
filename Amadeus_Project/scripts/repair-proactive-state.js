'use strict';

/**
 * 清理 amadeus_data 里的毒句 / 产品腔 / 旁白混排。
 * 用法: node scripts/repair-proactive-state.js --dir <amadeus_data>
 *
 * 主判：lib/generationGate（结构阀门），辅以内部泄漏与旧 polluted 模式。
 */

const fs = require('fs');
const path = require('path');

const index = process.argv.indexOf('--dir');
const dataDir = path.resolve(index >= 0 ? process.argv[index + 1] : '');
if (!dataDir || !fs.existsSync(dataDir)) {
  throw new Error('usage: node scripts/repair-proactive-state.js --dir <amadeus_data>');
}

const { gateAssistantReply } = require('../lib/generationGate');
const { isInternalControlLeak } = require('../lib/unifiedDialogueLog');

const legacyPolluted = /刷抖音|抖音这种事|毫无意义的东西|压缩为≤20字标签|NO_CONFLICT|已有：.*\n新：/i;

function shouldDropText(text, ctx = {}) {
  const t = String(text || '').trim();
  if (!t) return true;
  if (isInternalControlLeak(t)) return true;
  if (legacyPolluted.test(t)) return true;
  const gated = gateAssistantReply(t, ctx);
  return gated.action === 'drop';
}

function sanitizeText(text, ctx = {}) {
  const t = String(text || '').trim();
  if (!t) return '';
  if (isInternalControlLeak(t) || legacyPolluted.test(t)) return '';
  const gated = gateAssistantReply(t, ctx);
  if (gated.action === 'drop') return '';
  return gated.text || t;
}

function cleanDialogueEntries(entries) {
  if (!Array.isArray(entries)) return entries;
  const out = [];
  for (const e of entries) {
    if (!e || typeof e !== 'object') continue;
    const role = e.role === 'user' ? 'user' : 'assistant';
    const text = String(e.text || e.content || '');
    if (role === 'user') {
      if (isInternalControlLeak(text) || legacyPolluted.test(text)) continue;
      out.push({ ...e, text: text.trim() });
      continue;
    }
    const autonomy = e.autonomy === true || e.proactive === true || e.lite === true
      || /autonomy|lite|proactive|copresence/.test(String(e.source || ''));
    const next = sanitizeText(text, { autonomy, proactive: autonomy });
    if (!next) continue;
    out.push({ ...e, text: next });
  }
  return out;
}

function cleanDeep(value, ctx = {}) {
  if (Array.isArray(value)) {
    // 对话条目数组
    if (value.length && value[0] && typeof value[0] === 'object' && ('role' in value[0]) && ('text' in value[0] || 'content' in value[0])) {
      return cleanDialogueEntries(value);
    }
    return value
      .map((item) => cleanDeep(item, ctx))
      .filter((item) => {
        if (typeof item === 'string') return !shouldDropText(item, ctx);
        if (item == null) return false;
        return true;
      });
  }
  if (!value || typeof value !== 'object') {
    if (typeof value === 'string') {
      const next = sanitizeText(value, ctx);
      return next;
    }
    return value;
  }
  const out = {};
  for (const [key, child] of Object.entries(value)) {
    if (legacyPolluted.test(key)) continue;
    if (key === 'entries' && Array.isArray(child)) {
      out[key] = cleanDialogueEntries(child);
      continue;
    }
    if (typeof child === 'string') {
      const next = sanitizeText(child, ctx);
      if (!next && shouldDropText(child, ctx)) continue;
      if (next) out[key] = next;
      continue;
    }
    out[key] = cleanDeep(child, ctx);
  }
  return out;
}

const files = [
  'autonomy_subsystem.json',
  'user_model.json',
  'understanding_subsystem.json',
  'unified_dialogue_log.json',
  'memory_palace.json',
  'conversation_initiative.json',
];

const results = [];
for (const name of files) {
  const file = path.join(dataDir, name);
  if (!fs.existsSync(file)) continue;
  const beforeText = fs.readFileSync(file, 'utf8');
  let parsed;
  try {
    parsed = JSON.parse(beforeText);
  } catch (e) {
    results.push({ file, error: e.message });
    continue;
  }

  const beforeEntries = Array.isArray(parsed?.entries) ? parsed.entries.length : null;
  const next = cleanDeep(parsed);
  const afterText = `${JSON.stringify(next, null, 2)}\n`;
  if (afterText === beforeText) continue;

  const backup = `${file}.backup-${Date.now()}`;
  fs.copyFileSync(file, backup);
  fs.writeFileSync(file, afterText, 'utf8');
  const afterEntries = Array.isArray(next?.entries) ? next.entries.length : null;
  results.push({
    file,
    backup,
    beforeEntries,
    afterEntries,
    removedEntries: beforeEntries != null && afterEntries != null ? beforeEntries - afterEntries : null,
    bytesBefore: beforeText.length,
    bytesAfter: afterText.length,
  });
}

// 顺带清备份目录里对话实录（只读扫描 + 可选 --scrub-backups）
const scrubBackups = process.argv.includes('--scrub-backups');
if (scrubBackups) {
  const backupDirs = fs.readdirSync(dataDir, { withFileTypes: true })
    .filter((d) => d.isDirectory() && /^_memory_backup_/.test(d.name))
    .map((d) => path.join(dataDir, d.name));
  for (const dir of backupDirs) {
    const file = path.join(dir, 'unified_dialogue_log.json');
    if (!fs.existsSync(file)) continue;
    const beforeText = fs.readFileSync(file, 'utf8');
    let parsed;
    try {
      parsed = JSON.parse(beforeText);
    } catch {
      continue;
    }
    const next = cleanDeep(parsed);
    const afterText = `${JSON.stringify(next, null, 2)}\n`;
    if (afterText === beforeText) continue;
    const backup = `${file}.backup-${Date.now()}`;
    fs.copyFileSync(file, backup);
    fs.writeFileSync(file, afterText, 'utf8');
    results.push({
      file,
      backup,
      beforeEntries: Array.isArray(parsed?.entries) ? parsed.entries.length : null,
      afterEntries: Array.isArray(next?.entries) ? next.entries.length : null,
      scrubBackup: true,
    });
  }
}

console.log(JSON.stringify({ dataDir, results }, null, 2));
