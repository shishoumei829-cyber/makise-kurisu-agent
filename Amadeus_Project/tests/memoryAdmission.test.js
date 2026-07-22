'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const { MemoryAdmissionPolicy } = require('../lib/memoryAdmission');
const { AutonomySubsystem } = require('../digital_life/autonomy');
const { UnderstandingSubsystem } = require('../digital_life/understanding');
const { UserModel, ConversationAnalytics } = require('../user_model');

test('short casual commands stay in working memory only', () => {
  const policy = new MemoryAdmissionPolicy();
  const out = policy.assessUserText('刷抖音吧', { source: 'user' });
  assert.equal(out.tier, 'working');
  assert.equal(out.allowProfile, false);
  assert.equal(out.allowCuriosity, false);
  assert.equal(out.allowInference, false);
});

test('explicit user preferences can enter durable memory', () => {
  const policy = new MemoryAdmissionPolicy();
  const out = policy.assessUserText('我平时很喜欢看科幻电影', { source: 'user' });
  assert.equal(out.tier, 'durable');
  assert.equal(out.allowProfile, true);
  assert.equal(out.allowInference, true);
});

test('synthetic initiative input is rejected by every memory channel', () => {
  const policy = new MemoryAdmissionPolicy();
  const out = policy.assessUserText('（想说话）继续旧话题', { source: 'synthetic', synthetic: true });
  assert.equal(out.tier, 'reject');
  assert.equal(out.allowEvent, false);
  assert.equal(out.allowProfile, false);
});

test('assistant-led topic repetition is quarantined instead of self-reinforced', () => {
  const policy = new MemoryAdmissionPolicy();
  const now = Date.now();
  policy.observe('user', '刷抖音吧', { now });
  policy.observe('proactive', '又在刷抖音？', { now: now + 1000 });
  policy.observe('proactive', '刷抖音很浪费时间。', { now: now + 2000 });
  const out = policy.assessUserText('刷抖音吧', { source: 'user' });
  assert.equal(out.reason, 'topic_quarantined');
  assert.ok(out.contaminated.length > 0);
  assert.equal(out.allowCuriosity, false);
});

test('working-only text does not leak into autonomy, understanding, or analytics text history', () => {
  const policy = new MemoryAdmissionPolicy();
  const admission = policy.assessUserText('刷抖音吧', { source: 'user' });
  const autonomy = new AutonomySubsystem();
  autonomy.onConversationTurn({ userText: '刷抖音吧', memoryAdmission: admission, pad: {}, memory: { events: [] } });
  assert.doesNotMatch(JSON.stringify(autonomy.snapshot()), /刷抖音/);

  const understanding = new UnderstandingSubsystem();
  understanding.onConversationTurn({ userText: '刷抖音吧', memoryAdmission: admission });
  assert.doesNotMatch(JSON.stringify(understanding.getPublicState()), /刷抖音/);

  const analytics = new ConversationAnalytics(new UserModel());
  analytics.analyze('刷抖音吧', { persistText: false });
  assert.equal(analytics._log.at(-1).text, '');
});
