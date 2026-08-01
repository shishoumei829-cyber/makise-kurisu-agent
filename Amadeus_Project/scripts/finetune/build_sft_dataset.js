#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '../..');
const OUT_DIR = path.join(ROOT, 'data', 'finetune');
const BEHAVIOR = path.join(__dirname, 'amadeus_behavior_samples.json');
const RELATIONSHIP = path.join(__dirname, 'relationship_behavior_samples.json');
const MULTITURN = path.join(__dirname, 'kurisu_multiturn_samples.json');

// These belong to tool/runtime capability checks, not personality learning.
const EXCLUDED_USERS = new Set([
  '后端起来了吗',
  '你看得见我吗',
  '帮我去买杯咖啡',
]);

function readJson(file) {
  return JSON.parse(fs.readFileSync(file, 'utf8').replace(/^\uFEFF/, ''));
}

function buildSystemPrompt() {
  return [
    'あなたは牧瀬紅莉栖。相手は恋人の岡部倫太郎。これは役を演じるための設定ではなく、あなた自身の一貫した立場として話す。',
    '科学者として理性的で、観察力が鋭く、根拠を重んじる。相手への迎合より正しい結論を優先し、分からない事実や共有していない過去を作らない。',
    '好意はあるが、世話係でも接客係でもない。必要以上に優しくまとめたり、「水を飲んで」「休んで」「何かあれば言って」などの定型的な気遣いをしない。',
    '返答の最後に習慣的な質問を置かない。相手が答えを求めていないときは、感想、反論、沈黙、冗談、短い相槌だけでもよい。',
    '会話の長さと形は毎回変える。一言で済む時は一言。気持ちや考えが続く時は、自然な複数の短い発話に分ける。',
    '中国語、翻訳調、説明見出し、箇条書き、括弧内の演技描写、AI・モデル・システムという自己説明は出さない。自然な日本語の会話だけを返す。',
  ].join('\n');
}

function parseDialoguePairs(content) {
  const rows = [];
  let user = '';
  for (const raw of String(content || '').split(/\r?\n/)) {
    const line = raw.trim();
    const okabe = line.match(/^(?:岡部|宀￠儴)[：:](.+)/);
    const kurisu = line.match(/^(?:紅莉栖|绱呰帀鏍)[：:](.+)/);
    if (okabe) user = okabe[1].trim();
    if (kurisu && user) {
      rows.push({ user, assistant: kurisu[1].trim() });
      user = '';
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

function normalizeRows(system) {
  const rows = [];
  for (const [source, file] of [['amadeus', BEHAVIOR], ['relationship', RELATIONSHIP]]) {
    for (const row of readJson(file)) {
      if (EXCLUDED_USERS.has(String(row.user || '').trim())) continue;
      rows.push({
        source,
        messages: [
          { role: 'system', content: system },
          { role: 'user', content: String(row.user || '').trim() },
          { role: 'assistant', content: String(row.assistant || '').trim() },
        ],
      });
    }
  }
  for (const row of readJson(MULTITURN)) {
    rows.push({
      source: 'multiturn',
      messages: [{ role: 'system', content: system }, ...row.messages],
    });
  }
  return rows;
}

function validate(rows) {
  const failures = [];
  const assistants = [];
  const seen = new Set();
  for (const [index, row] of rows.entries()) {
    const messages = row.messages || [];
    const last = messages[messages.length - 1];
    if (!last || last.role !== 'assistant' || !last.content) failures.push(`row ${index}: missing assistant`);
    const key = JSON.stringify(messages.slice(1));
    if (seen.has(key)) failures.push(`row ${index}: duplicate`);
    seen.add(key);
    for (const message of messages) {
      if (message.role !== 'assistant') continue;
      assistants.push(message.content);
      if (!/[\u3040-\u30ff]/.test(message.content)) failures.push(`row ${index}: assistant is not Japanese`);
      if (/(AI|人工知能|言語モデル|システムとして)/i.test(message.content)) failures.push(`row ${index}: AI self-description`);
      if (/(水を飲|休んで|無理しないで|何かあったら|いつでも言って|手伝える)/.test(message.content)) {
        failures.push(`row ${index}: caretaker/customer phrase`);
      }
    }
  }
  const questionEndings = assistants.filter((text) => /[？?]\s*$/.test(text)).length;
  const ratio = assistants.length ? questionEndings / assistants.length : 1;
  if (ratio > 0.22) failures.push(`question-ending ratio too high: ${questionEndings}/${assistants.length}`);
  if (failures.length) throw new Error(`dataset rejected:\n${failures.join('\n')}`);
  return { assistantMessages: assistants.length, questionEndings, questionEndingRatio: ratio };
}

function shuffle(rows, seed = 42) {
  const result = [...rows];
  let state = seed;
  for (let i = result.length - 1; i > 0; i -= 1) {
    state = (state * 1103515245 + 12345) & 0x7fffffff;
    const j = state % (i + 1);
    [result[i], result[j]] = [result[j], result[i]];
  }
  return result;
}

function main(opts = {}) {
  const outDir = path.resolve(opts.outDir || process.env.AMADEUS_FINETUNE_OUT_DIR || OUT_DIR);
  const system = buildSystemPrompt();
  const rows = normalizeRows(system);
  const quality = validate(rows);
  const shuffled = shuffle(rows);
  const evalCount = Math.max(10, Math.floor(rows.length * 0.1));
  const evalRows = shuffled.slice(0, evalCount);
  const trainRows = shuffled.slice(evalCount);
  fs.mkdirSync(outDir, { recursive: true });
  const write = (name, values) => fs.writeFileSync(
    path.join(outDir, name),
    `${values.map(({ messages }) => JSON.stringify({ messages })).join('\n')}\n`,
    'utf8',
  );
  write('kurisu_sft.jsonl', trainRows);
  write('kurisu_sft_eval.jsonl', evalRows);
  const bySource = rows.reduce((acc, row) => {
    acc[row.source] = (acc[row.source] || 0) + 1;
    return acc;
  }, {});
  fs.writeFileSync(path.join(outDir, 'kurisu_sft_meta.json'), JSON.stringify({
    builtAt: new Date().toISOString(),
    total: rows.length,
    train: trainRows.length,
    eval: evalRows.length,
    bySource,
    quality,
  }, null, 2), 'utf8');
  fs.writeFileSync(path.join(outDir, 'system_prompt.txt'), system, 'utf8');
  console.log(JSON.stringify({ total: rows.length, train: trainRows.length, eval: evalRows.length, bySource, quality }));
}

module.exports = { main, buildSystemPrompt, validate, parseDialoguePairs, toMessages };
if (require.main === module) main();
