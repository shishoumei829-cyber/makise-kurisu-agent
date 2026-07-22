'use strict';

const { describe, it } = require('node:test');
const assert = require('node:assert/strict');
const path = require('path');
const os = require('os');
const fs = require('fs');
const {
  buildSystemPrompt,
  parseDialoguePairs,
  toMessages,
} = require('../scripts/finetune/build_sft_dataset');

describe('finetune dataset', () => {
  it('buildSystemPrompt keeps identity, language, and reasoning rules compact', () => {
    const prompt = buildSystemPrompt();
    assert.match(prompt, /牧瀬紅莉栖/);
    assert.match(prompt, /日本語/);
    assert.match(prompt, /正しい結論/);
    assert.ok(prompt.length < 800, `system prompt should stay compact, got ${prompt.length}`);
  });

  it('parseDialoguePairs parses Okabe and Kurisu turns', () => {
    const sample = `岡部：クリスティーナ！
紅莉栖：だから"ティーナ"って付けるな`;
    const rows = parseDialoguePairs(sample);
    assert.equal(rows.length, 1);
    assert.match(rows[0].assistant, /ティーナ/);
    assert.ok(rows[0].user.length > 0);
  });

  it('toMessages creates a complete conversation', () => {
    const result = toMessages('sys', { user: '在吗', assistant: 'いるけど。' });
    assert.equal(result.messages.length, 3);
    assert.equal(result.messages[1].role, 'user');
    assert.equal(result.messages[2].role, 'assistant');
  });

  it('build_sft_dataset generates enough clean samples', () => {
    const { main } = require('../scripts/finetune/build_sft_dataset');
    const outDir = fs.mkdtempSync(path.join(os.tmpdir(), 'amadeus-sft-'));
    main({ outDir });
    const output = path.join(outDir, 'kurisu_sft.jsonl');
    assert.ok(fs.existsSync(output), 'kurisu_sft.jsonl should exist after build');
    const lines = fs.readFileSync(output, 'utf8').split(/\n/).filter(Boolean);
    assert.ok(lines.length >= 50, `expected >=50 samples, got ${lines.length}`);
    for (const line of lines) {
      const row = JSON.parse(line);
      const user = row.messages.find((message) => message.role === 'user')?.content || '';
      const assistant = row.messages.find((message) => message.role === 'assistant')?.content || '';
      assert.match(assistant, /[\u3040-\u30ff]/);
      assert.doesNotMatch(assistant, /[\u0400-\u04ff]/);
      assert.notEqual(user.trim(), assistant.trim());
    }
  });
});
