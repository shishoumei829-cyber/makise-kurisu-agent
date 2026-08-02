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

test('style criticism stays a normal current-turn response, never a repair script', () => {
  const { core } = createCore();
  const result = core.deliberate({ userText: '\u4f60\u600e\u4e48\u8fd8\u662f\u4e00\u80a1\u5ba2\u670d\u5473\uff0c\u592a\u5047\u4e86' });
  assert.notEqual(result.intent.action, 'repair');
  assert.equal(result.intent.action, 'stance');
  assert.match(result.mind.interpretation, /存在の仕方|失望/);
  assert.match(result.promptBlock, /紅莉栖の立場/);
  assert.ok(core.snapshot().relationship.repairNeed > 0);
});

test('subject core gives a concrete user turn one stable center and speech act', () => {
  const { core } = createCore();
  const first = core.deliberate({ userText: '我今天终于把报告写完了', pad: { P: 0.2, A: 0.1, D: 0.1 } });
  const second = core.deliberate({ userText: '我今天终于把报告写完了', pad: { P: 0.2, A: 0.1, D: 0.1 } });
  assert.equal(first.shouldSpeak, true);
  assert.equal(first.intent.action, 'tease');
  assert.equal(second.intent.action, 'tease');
  assert.match(first.promptBlock, /报告写完/);
  assert.match(first.promptBlock, /質問で繋がず/);
});

test('proactive speech is silent without a lasting subject and respects social boundaries', () => {
  const { core } = createCore();
  assert.equal(core.planProactive({ openThoughts: [] }).shouldSpeak, false);
  const thought = { id: 't1', content: '那份报告其实值得夸一句', tension: 0.72, status: 'open', earliestSpeakAt: 0 };
  assert.equal(core.planProactive({ openThoughts: [thought], dnd: true }).reason, 'social_boundary');
  const plan = core.planProactive({ openThoughts: [thought] });
  assert.equal(plan.shouldSpeak, true);
  assert.notEqual(plan.action, 'clarify');
  assert.equal(plan.intent.allowQuestion, false);
});

test('boredom is appraised as companionship instead of a support question', () => {
  const { core } = createCore();
  const first = core.deliberate({ userText: '不知道干什么', expressionText: '何をすればいいのか分からない' });
  core.integrateOutcome({ intent: first.intent, reply: '……暇なら、少しくらい付き合ってあげる。', accepted: true });
  const second = core.deliberate({ userText: '就是无聊' });
  assert.equal(first.intent.action, 'accompany');
  assert.equal(second.intent.action, 'accompany');
  assert.notEqual(first.mind.perception, second.mind.perception);
  assert.equal(first.intent.allowQuestion, false);
  assert.equal(second.intent.allowQuestion, false);
  assert.match(second.mind.interpretation, /一緒に過ごしたい/);
  assert.match(first.promptBlock, /何をすればいいのか分からない/);
  assert.doesNotMatch(first.promptBlock, /不知道干什么/);
  assert.match(first.intent.spokenNucleus, /私と少し話すか同じ時間/);
  assert.equal(core.evaluateReply('ふん、別に退屈してないし。', first.intent).ok, false);
  assert.equal(core.evaluateReply('決まらないなら、少しくらい私と話してれば。', first.intent).ok, true);
  assert.equal(core.evaluateReply('それなら、私と少し話さない？', first.intent).ok, true);
});

test('relationship changes the choice but never removes the subject freedom to refuse', () => {
  const { core } = createCore();
  core.state.affect.irritation = 1;
  core.state.affect.energy = 0;
  core.state.relationship.friction = 1;
  core.state.relationship.reciprocity = 0;
  const result = core.deliberate({ userText: '不知道干什么', expressionText: '何をすればいいのか分からない' });
  assert.equal(result.intent.action, 'decline');
  assert.ok(result.intent.semanticTags.includes('autonomous_decline'));
  assert.equal(core.evaluateReply('私は退屈してないし、今は付き合う気分じゃない。', result.intent).ok, true);
});

test('a harsh autonomous reply leaves a relationship consequence', () => {
  const { core } = createCore();
  const result = core.deliberate({ userText: '克里斯蒂娜' });
  const before = core.snapshot().relationship;
  core.integrateOutcome({ intent: result.intent, reply: 'うるさい、馬鹿岡部。', accepted: true });
  const after = core.snapshot().relationship;
  assert.ok(after.friction > before.friction);
  assert.ok(after.repairNeed > before.repairNeed);
  assert.match(after.lastShift, /強く突き放した|きつい言葉/);
});

test('an interaction changes the persistent relationship and later inner state', () => {
  const { core } = createCore();
  const before = core.snapshot().relationship.repairNeed;
  core.deliberate({ userText: '你说话完全像机器，我很失望' });
  const hurt = core.snapshot();
  assert.ok(hurt.relationship.repairNeed > before);
  assert.match(hurt.currentMind.emotion, /痛いところ/);
  core.deliberate({ userText: '不过我还是喜欢你' });
  const after = core.snapshot();
  assert.ok(after.relationship.intimacy > hurt.relationship.intimacy);
  assert.match(after.currentMind.relationshipMeaning, /近づいた/);
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
