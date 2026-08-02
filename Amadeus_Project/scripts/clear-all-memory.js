'use strict';

/**
 * 清空运行时记忆（对话/宫殿/admission/事件等）。
 * 用法: node scripts/clear-all-memory.js [--dir <amadeus_data>]
 */

const fs = require('fs');
const path = require('path');
const os = require('os');

function arg(name) {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 ? process.argv[i + 1] : '';
}

const dataDir = path.resolve(arg('dir') || process.env.AMADEUS_DATA_DIR || path.join(os.homedir(), 'amadeus_data'));

const FILES = [
  'unified_dialogue_log.json',
  'conversation_log.json',
  'memory_palace.json',
  'memory_admission.json',
  'event_log.json',
  'timeline.json',
  'observations.json',
  'patterns.json',
  'user_model.json',
  'user_profile.json',
  'learning_state.json',
  'conversation_initiative.json',
  'behavior_context.json',
  'digital_life_state.json',
  'understanding_subsystem.json',
  'autonomy_subsystem.json',
  'evolution_subsystem.json',
  'metacognition_subsystem.json',
  'self_reflection.json',
  'value_consistency.json',
  'personality_evolution.json',
];

const KEEP_RESET = {
  whoami: {
    name: '冈部伦太郎',
    traits: ['中二', '不按常理', '其实很在意同伴'],
    preferences: [],
    basics: { role: '未来道具研究所 · 凤凰院凶真' },
    relationship_note: '他就是冈部伦太郎——未来道具研究所那个很烦、很中二、但你很熟的人。不是陌生人，禁止装不认识。',
    partner_id: 'okabe',
    last_updated: Date.now(),
  },
};

if (!fs.existsSync(dataDir)) {
  console.log(JSON.stringify({ ok: true, dataDir, cleared: 0, note: 'dir missing' }));
  process.exit(0);
}

const stamp = Date.now();
const backupDir = path.join(dataDir, `_memory_backup_${stamp}`);
fs.mkdirSync(backupDir, { recursive: true });

const cleared = [];
for (const name of FILES) {
  const p = path.join(dataDir, name);
  if (fs.existsSync(p)) {
    fs.copyFileSync(p, path.join(backupDir, name));
  }
  // 始终写回空结构（含原先不存在的宫殿文件）
  if (name === 'unified_dialogue_log.json') {
    fs.writeFileSync(p, JSON.stringify({ version: 1, savedAt: Date.now(), entries: [] }, null, 2));
  } else if (name === 'memory_palace.json') {
    fs.writeFileSync(p, JSON.stringify({
      version: 1,
      rooms: { hall: [], lab: [], cafe: [], forbidden: [] },
      proactiveBuffer: [],
      audits: [],
      updatedAt: Date.now(),
    }, null, 2));
  } else if (name === 'memory_admission.json') {
    fs.writeFileSync(p, JSON.stringify({ version: 1, evidence: {}, quarantined: {}, audits: [] }, null, 2));
  } else if (name === 'timeline.json' || name === 'event_log.json' || name === 'observations.json' || name === 'patterns.json' || name === 'conversation_log.json') {
    fs.writeFileSync(p, '[]\n');
  } else if (name.endsWith('.json')) {
    fs.writeFileSync(p, '{}\n');
  } else {
    fs.writeFileSync(p, '');
  }
  cleared.push(name);
}

const whoamiPath = path.join(dataDir, 'whoami.json');
if (fs.existsSync(whoamiPath)) {
  fs.copyFileSync(whoamiPath, path.join(backupDir, 'whoami.json'));
}
fs.writeFileSync(whoamiPath, JSON.stringify(KEEP_RESET.whoami, null, 2));

// butler journal 可选截断（保留文件，清空内容）
const journal = path.join(dataDir, 'butler', 'events.jsonl');
if (fs.existsSync(journal)) {
  fs.copyFileSync(journal, path.join(backupDir, 'butler_events.jsonl'));
  fs.writeFileSync(journal, '');
  cleared.push('butler/events.jsonl');
}

console.log(JSON.stringify({
  ok: true,
  dataDir,
  backupDir,
  cleared,
  whoami: 'reset_okabe_seed',
}, null, 2));
