'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { SubjectCore } = require('../brain/subjectCore');

function createCore() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'amadeus-subject-'));
  return { dir, core: new SubjectCore({ statePath: path.join(dir, 'subject_core.json') }) };
}

test('style criticism becomes a repair intent, never a fixed self-defence', () => {
  const { core } = createCore();
  const result = core.deliberate({ userText: '\u4f60\u600e\u4e48\u8fd8\u662f\u4e00\u80a1\u5ba2\u670d\u5473\uff0c\u592a\u5047\u4e86' });
  assert.equal(result.intent.action, 'repair');
  assert.match(result.promptBlock, /\u8fd4\u7b54\u306e\u4f5c\u308a\u65b9\u3084\u81ea\u5206\u306e\u6027\u8cea\u306b\u3064\u3044\u3066\u8a71\u3055\u305a/);
  assert.doesNotMatch(result.promptBlock, /\u5ba2\u670d|\u30c6\u30f3\u30d7\u30ec\u30fc\u30c8|AI/);
});

test('subject core gives a concrete user turn one stable center and speech act', () => {
  const { core } = createCore();
  const first = core.deliberate({ userText: '我今天终于把报告写完了', pad: { P: 0.2, A: 0.1, D: 0.1 } });
  const second = core.deliberate({ userText: '我今天终于把报告写完了', pad: { P: 0.2, A: 0.1, D: 0.1 } });
  assert.equal(first.shouldSpeak, true);
  assert.equal(first.intent.action, 'tease');
  assert.equal(second.intent.action, 'tease');
  assert.match(first.promptBlock, /报告写完/);
  assert.match(first.promptBlock, /質問にして/);
});

test('proactive speech is silent without a lasting subject and respects social boundaries', () => {
  const { core } = createCore();
  assert.equal(core.planProactive({ openThoughts: [] }).shouldSpeak, false);
  const thought = { id: 't1', content: '那份报告其实值得夸一句', tension: 0.72, status: 'open', earliestSpeakAt: 0 };
  assert.equal(core.planProactive({ openThoughts: [thought], dnd: true }).reason, 'social_boundary');
  const plan = core.planProactive({ openThoughts: [thought] });
  assert.equal(plan.shouldSpeak, true);
  assert.equal(plan.action, 'thought');
  assert.equal(plan.intent.allowQuestion, false);
});

test('outcome and subject state persist across reload', () => {
  const { core } = createCore();
  const result = core.deliberate({ userText: '我有点焦虑', pad: { P: -0.4, A: 0.5, D: -0.1 } });
  core.integrateOutcome({ intent: result.intent, reply: '焦虑这两个字先别轻飘飘地丢出来。', accepted: true });
  const reloaded = new SubjectCore({ statePath: core.statePath });
  assert.equal(reloaded.snapshot().activeIntent.status, 'expressed');
  assert.ok(reloaded.snapshot().experiences.length >= 1);
  assert.ok(reloaded.snapshot().attention.some((item) => item.subject.includes('焦虑')));
});

test('the core rejects proactive generic care unrelated to its current thought', () => {
  const { core } = createCore();
  const plan = core.planProactive({ openThoughts: [{ content: '报告终于写完了', tension: 0.7, status: 'open', earliestSpeakAt: 0 }] });
  assert.equal(core.evaluateReply('累了吧，别太辛苦。', plan.intent).ok, false);
  assert.equal(core.evaluateReply('报告写完了还摆出那副要死的样子，真会演。', plan.intent).ok, true);
});
