#!/usr/bin/env node
'use strict';

/**
 * 从 brain_data/kurisu_ja + 人设文件 + Amadeus 行为样本 构建 SFT JSONL
 * 格式：OpenAI messages（system / user 中文 / assistant 日语）
 *
 * 用法：node scripts/finetune/build_sft_dataset.js
 */

const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '../..');
const OUT_DIR = path.join(ROOT, 'data', 'finetune');
const JA_DIR = path.join(ROOT, 'brain_data', 'kurisu_ja');
const BEHAVIOR = path.join(__dirname, 'amadeus_behavior_samples.json');

const OKABE_CN = [
  ['クリスティナ', '克里斯蒂娜！'],
  ['助手', '助手！'],
  ['頼む', '拜托了'],
  ['タイムリープ', '时间跳跃机器怎么样了'],
  ['まゆり', '真由理她…'],
  ['運命石', '这是命运石之门的选择'],
  ['フハハハ', '哈哈哈哈'],
];

function readText(fp) {
  return fs.readFileSync(fp, 'utf8').replace(/\uFEFF/g, '');
}

function clip(s, n) {
  const t = String(s || '').replace(/\s+/g, ' ').trim();
  return t.length <= n ? t : `${t.slice(0, n)}…`;
}

function buildSystemPrompt() {
  return [
    'あなたは牧瀬紅莉栖。相手は親しい岡部倫太郎。',
    '自然な日本語の会話文だけで直接返答し、中国語、地の文、形式ラベル、AIを名乗る表現は出さない。',
    '理性的で鋭く、証拠を重視する。軽い突っ込みはよいが、理由なく攻撃しない。',
    '会話にない記憶は作らない。技術、論理、計算では正しい結論を最優先する。',
  ].join('\n');
}

function okabeToCn(jp) {
  let t = String(jp || '').trim();
  for (const [re, cn] of OKABE_CN) {
    if (t.includes(re)) return cn;
  }
  if (/[？?]/.test(t)) return clip(t, 40) || '…';
  if (t.length <= 12) return t;
  return clip(`（岡部）${t}`, 48);
}

function parseDialoguePairs(content) {
  const rows = [];
  let user = '';
  for (const raw of content.split(/\r?\n/)) {
    const line = raw.trim();
    if (!line || line.startsWith('[')) continue;
    const okabe = line.match(/^岡部[：:](.+)/);
    const kurisu = line.match(/^紅莉栖[：:](.+)/);
    const amadeus = line.match(/^アマデウス[：:](.+)/);
    if (okabe) {
      user = okabeToCn(okabe[1]);
    } else if (kurisu && user) {
      rows.push({ user, assistant: kurisu[1].trim() });
      user = '';
    } else if (amadeus) {
      rows.push({ user: '…', assistant: amadeus[1].trim() });
    }
  }
  return rows;
}

function toMessages(system, pair) {
  return {
    messages: [
      { role: 'system', content: system },
      { role: 'user', content: pair.user },
      { role: 'assistant', content: pair.assistant },
    ],
  };
}

function dedupePairs(pairs) {
  const seen = new Set();
  const out = [];
  for (const p of pairs) {
    const key = `${p.user}|||${p.assistant}`;
    if (seen.has(key)) continue;
    if (!p.user || !p.assistant) continue;
    if (p.assistant.length < 2) continue;
    if (!/[\u3040-\u30ff]/.test(p.assistant)) continue;
    if (/[\u0400-\u04ff]/.test(p.assistant)) continue;
    if (p.user.trim() === p.assistant.trim()) continue;
    seen.add(key);
    out.push(p);
  }
  return out;
}

function shuffle(arr, seed = 42) {
  const a = [...arr];
  let s = seed;
  for (let i = a.length - 1; i > 0; i--) {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    const j = s % (i + 1);
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function main(opts = {}) {
  const outDir = path.resolve(opts.outDir || process.env.AMADEUS_FINETUNE_OUT_DIR || OUT_DIR);
  const system = buildSystemPrompt();
  const pairs = [];

  const behavior = JSON.parse(readText(BEHAVIOR));
  for (const row of behavior) {
    pairs.push({ user: row.user, assistant: row.assistant, source: 'amadeus' });
  }

  const pairsFile = path.join(JA_DIR, 'dialogue_pairs_block_34.txt');
  if (fs.existsSync(pairsFile)) {
    for (const p of parseDialoguePairs(readText(pairsFile))) {
      pairs.push({ ...p, source: 'dialogue_pairs' });
    }
  }

  const unique = dedupePairs(pairs);
  const shuffled = shuffle(unique);
  const evalCount = Math.max(8, Math.floor(shuffled.length * 0.05));
  const evalRows = shuffled.slice(0, evalCount);
  const trainRows = shuffled.slice(evalCount);

  fs.mkdirSync(outDir, { recursive: true });
  const trainPath = path.join(outDir, 'kurisu_sft.jsonl');
  const evalPath = path.join(outDir, 'kurisu_sft_eval.jsonl');
  const metaPath = path.join(outDir, 'kurisu_sft_meta.json');

  const writeJsonl = (fp, rows) => {
    fs.writeFileSync(
      fp,
      rows.map((r) => JSON.stringify(toMessages(system, r))).join('\n') + '\n',
      'utf8',
    );
  };

  writeJsonl(trainPath, trainRows);
  writeJsonl(evalPath, evalRows);

  const bySource = {};
  for (const p of unique) bySource[p.source] = (bySource[p.source] || 0) + 1;

  fs.writeFileSync(metaPath, JSON.stringify({
    builtAt: new Date().toISOString(),
    total: unique.length,
    train: trainRows.length,
    eval: evalRows.length,
    systemChars: system.length,
    bySource,
  }, null, 2), 'utf8');

  fs.writeFileSync(
    path.join(outDir, 'system_prompt.txt'),
    system,
    'utf8',
  );

  console.log(`[finetune] 数据集已写入 ${trainPath}`);
  console.log(`[finetune] 训练 ${trainRows.length} / 验证 ${evalRows.length} / 合计 ${unique.length}`);
  console.log('[finetune] 来源分布:', bySource);
}

module.exports = {
  main,
  buildSystemPrompt,
  parseDialoguePairs,
  dedupePairs,
  toMessages,
};

if (require.main === module) main();
