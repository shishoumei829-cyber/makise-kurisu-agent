'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');

const {
  scoreOoc,
  sanitizeOocSurface,
  replyNeedsOocRepair,
  repairKurisuReply,
} = require('../lib/oocGuard');

test('scoreOoc should detect assistant-like identity leakage', () => {
  const out = scoreOoc('作为一个AI，我很高兴为您服务。');
  assert.ok(out.score >= 3);
  assert.ok(out.hits.includes('ai_identity'));
});

test('sanitizeOocSurface should remove obvious assistant traces', () => {
  const out = sanitizeOocSurface('作为一个AI，我很高兴为您服务。*叹气*');
  assert.equal(out.includes('作为一个AI'), false);
  assert.equal(out.includes('很高兴为您服务'), false);
  assert.equal(out.includes('*叹气*'), false);
});

test('replyNeedsOocRepair should mark generic lecture style on casual input', () => {
  const out = replyNeedsOocRepair(
    '在吗',
    '首先，你需要注意的是，综上所述，我们应该讨论一个完整框架。',
  );
  assert.equal(out.needs, true);
});

test('repairKurisuReply should leave empty empty (no template filler)', () => {
  const out = repairKurisuReply('你好', '');
  assert.equal(out, '');
});
