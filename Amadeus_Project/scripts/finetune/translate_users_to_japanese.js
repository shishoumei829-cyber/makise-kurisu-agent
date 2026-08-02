#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '../..');
const MODEL = process.env.AMADEUS_TRANSLATE_MODEL || 'qwen2.5:7b-instruct';
const OUTPUT_SUFFIX = process.env.AMADEUS_TRANSLATE_SUFFIX || 'ja';
const MODEL_SLUG = MODEL.replace(/[^a-z0-9]+/gi, '_').replace(/^_+|_+$/g, '').toLowerCase();
const CACHE_PATH = path.join(__dirname, `user_ja_translation_cache_${MODEL_SLUG}.json`);
const SPECIAL = {
  '亲爱的': 'ねえ、紅莉栖',
  '嗯。': 'うん。',
  '嗯': 'うん',
  '抱抱': '抱きしめて',
  '抱抱。': '抱きしめて。',
};

function readJsonl(file) {
  return fs.readFileSync(file, 'utf8').split(/\r?\n/).filter(Boolean).map(JSON.parse);
}

function isCleanJapanese(text) {
  const value = String(text || '');
  const withoutAllowedTerms = value.replace(/\b(?:AI|TTS)\b/gi, '');
  return /[\u3040-\u30ff]/.test(value)
    && !/[A-Za-z\u0400-\u04ff`{}]/.test(withoutAllowedTerms)
    && !/[个这机器么为说没还让帮给们时觉烦随便]/.test(value);
}

function loadCache() {
  try { return JSON.parse(fs.readFileSync(CACHE_PATH, 'utf8')); } catch { return {}; }
}

async function translate(text) {
  if (SPECIAL[text]) return SPECIAL[text];
  const response = await fetch('http://127.0.0.1:11434/api/chat', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      model: MODEL,
      stream: false,
      messages: [
        {
          role: 'system',
          content: [
            '把中国情侣聊天中的一句话翻成自然日语口语，保留原意和关系语气。',
            '不要逐字翻译中文夸张说法：“累死了”表示“疲れ切った”，不是自杀或死亡。',
            '问句仍是问句，不能把试探性的共同经历写成已经发生的事实。',
            '只输出日语正文，不解释，不加引号，必须包含假名。',
          ].join('\n'),
        },
        { role: 'user', content: text },
      ],
      options: { temperature: 0.05, num_predict: 100, num_ctx: 1024 },
    }),
  });
  const data = await response.json();
  const result = String(data.message?.content || '')
    .replace(/^["「]|["」]$/g, '')
    .trim();
  if (!/[\u3040-\u30ff]/.test(result)) throw new Error(`bad translation: ${text} -> ${result}`);
  return result;
}

async function mapLimit(values, limit, fn) {
  const result = new Array(values.length);
  let next = 0;
  async function worker() {
    while (next < values.length) {
      const index = next++;
      result[index] = await fn(values[index], index);
    }
  }
  await Promise.all(Array.from({ length: limit }, worker));
  return result;
}

async function main() {
  const cache = loadCache();
  for (const name of ['kurisu_sft', 'kurisu_sft_eval']) {
    const input = path.join(ROOT, 'data', 'finetune', `${name}.jsonl`);
    const rows = readJsonl(input);
    const unique = [...new Set(rows.flatMap((row) => row.messages
      .filter((message) => message.role === 'user')
      .map((message) => message.content)))];
    await mapLimit(unique.filter((text) => !cache[text]), 3, async (text) => {
      try {
        cache[text] = await translate(text);
      } catch (error) {
        cache[text] = null;
        console.warn(`[translate/skip] ${text}: ${error.message}`);
      }
      fs.writeFileSync(CACHE_PATH, JSON.stringify(cache, null, 2), 'utf8');
    });
    const translated = rows.filter((row) => row.messages.every((message) => (
      message.role !== 'user'
      || (typeof cache[message.content] === 'string' && isCleanJapanese(cache[message.content]))
    ))).map((row) => ({
      messages: row.messages.map((message) => (
        message.role === 'user'
          ? { ...message, content: cache[message.content] }
          : message
      )),
    }));
    const output = path.join(ROOT, 'data', 'finetune', `${name}_${OUTPUT_SUFFIX}.jsonl`);
    fs.writeFileSync(output, `${translated.map(JSON.stringify).join('\n')}\n`, 'utf8');
    console.log(JSON.stringify({ output, rows: rows.length, translatedUsers: unique.length }));
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
