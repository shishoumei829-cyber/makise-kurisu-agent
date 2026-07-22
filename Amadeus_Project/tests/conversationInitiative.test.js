'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const { ConversationInitiativeEngine } = require('../cognitive/conversationInitiative');

function engine() {
  return new ConversationInitiativeEngine({ cooldownMs: 15000 });
}

test('ordinary task answer releases the floor and stays silent', () => {
  const out = engine().decide({
    userText: '帮我检查一下这个文件',
    replyText: '已经检查完了，没有语法错误。',
    entropy: 0,
  });
  assert.equal(out.shouldSpeak, false);
});

test('assistant question does not stack another initiative', () => {
  const out = engine().decide({
    userText: '我最近有点烦',
    replyText: '是工作上的事，还是别的什么？',
    entropy: 0,
  });
  assert.equal(out.shouldSpeak, false);
  assert.equal(out.reason, 'assistant_already_passed_floor');
  assert.ok(out.reevaluateAfterMs >= 2800 && out.reevaluateAfterMs <= 7000);
});

test('after her question, silence phase can still care', () => {
  const out = engine().decide({
    userText: '我最近总觉得做什么都没意思，其实我也说不上来',
    replyText: '是工作上的事，还是别的什么？',
    phase: 'silence',
    entropy: 0.05,
    relScore: 0.5,
  });
  assert.equal(out.shouldSpeak, true);
  assert.equal(out.action, 'care');
});

test('vulnerable personal turn can create care motive', () => {
  const out = engine().decide({
    userText: '我最近总觉得做什么都没意思，其实我也说不上来',
    replyText: '这听起来不只是普通的无聊。',
    phase: 'silence',
    entropy: 0.1,
    relScore: 0.5,
  });
  assert.equal(out.shouldSpeak, true);
  assert.equal(out.action, 'care');
});

test('science opinion produces stance instead of universal follow-up', () => {
  const out = engine().decide({
    userText: '我觉得记忆最有趣的地方是它会不断重构事实',
    replyText: '至少你没有把记忆当成录像带。',
    entropy: 0.1,
    pad: { A: 0.5 },
  });
  assert.equal(out.shouldSpeak, true);
  assert.equal(out.action, 'stance');
});

test('same turn cannot speak twice and repeated action is discouraged', () => {
  const e = engine();
  const input = {
    userText: '我觉得时间机器真正有趣的是因果关系',
    replyText: '这次总算抓到一个值得讨论的点。',
    entropy: 0,
    pad: { A: 0.6 },
    now: 100000,
  };
  assert.equal(e.decide(input).shouldSpeak, true);
  assert.equal(e.decide({ ...input, phase: 'silence', now: 130000 }).shouldSpeak, false);
});

test('bored chat can produce a lightweight poke instead of a task', () => {
  const out = engine().decide({
    userText: '好无聊啊',
    replyText: '那就别一直发呆。',
    phase: 'floor_release',
    entropy: 0,
    relScore: 0.5,
  });
  assert.equal(out.shouldSpeak, true);
  assert.equal(out.action, 'poke');
  assert.ok(out.delivery.bubbleCount >= 1 && out.delivery.bubbleCount <= 3);
  assert.equal(out.delivery.allowNonSemantic, true);
  assert.ok(out.delivery.maxCharsPerBubble <= 24);
});

test('initiative reevaluation happens in seconds rather than a long countdown', () => {
  const out = engine().decide({
    userText: '好无聊啊',
    replyText: '我知道。',
    phase: 'floor_release',
    entropy: 0.99,
    relScore: 0.5,
  });
  assert.equal(out.shouldSpeak, false);
  assert.ok(out.reevaluateAfterMs >= 2200 && out.reevaluateAfterMs <= 7000);
});

test('idle initiative uses an adaptive state machine and learns from replies', () => {
  const e = engine();
  const plan = e.decideIdle({
    now: 100000,
    idleMs: 60000,
    contextFresh: true,
    proactiveQuotaOk: true,
    entropy: 0,
    relScore: 0.5,
  });
  assert.equal(plan.shouldSpeak, true);
  assert.ok(plan.delivery.bubbleCount >= 1 && plan.delivery.bubbleCount <= 3);
  e.registerSent({ now: 100000, text: '喂，别发呆。', action: plan.action });
  const state = e.registerFeedback({ now: 120000, type: 'reply', text: '怎么了' });
  assert.equal(state.ignoredStreak, 0);
  assert.equal(state.engagedStreak, 1);
  assert.equal(state.activeThread, null);
});

test('ignored proactive messages increase the next idle threshold', () => {
  const e = engine();
  e.registerSent({ now: 100000, text: '喂。', action: 'poke' });
  const expired = e.decideIdle({
    now: 100000 + 11 * 60000,
    idleMs: 20000,
    contextFresh: false,
    proactiveQuotaOk: true,
    entropy: 0,
  });
  assert.equal(expired.shouldSpeak, false);
  assert.equal(e.snapshot().ignoredStreak, 1);

  e.state.ignoredStreak = 3;
  e.state.lastSpokenAt = 100000 + 11 * 60000;
  const out = e.decideIdle({
    now: 100000 + 11 * 60000 + 8000,
    idleMs: 40000,
    contextFresh: false,
    proactiveQuotaOk: true,
    entropy: 0,
  });
  assert.equal(out.shouldSpeak, false);
  assert.equal(out.reason, 'adaptive_cooldown');
});

test('presence phase: opening the session often speaks first', () => {
  const e = engine();
  e.markSessionStart(100000);
  const out = e.decidePresence({
    now: 100000 + 5000,
    facePresent: true,
    proactiveQuotaOk: true,
    entropy: 0,
    relScore: 0.4,
  });
  assert.equal(out.shouldSpeak, true);
  assert.ok(['poke', 'tease', 'share'].includes(out.action));
  assert.equal(out.phase, 'presence');
  assert.equal(out.delivery.allowNonSemantic, true);
});

test('presence phase respects a short notice cooldown', () => {
  const e = engine();
  e.markSessionStart(100000);
  assert.equal(e.decidePresence({
    now: 105000,
    facePresent: true,
    proactiveQuotaOk: true,
    entropy: 0,
  }).shouldSpeak, true);
  const held = e.decidePresence({
    now: 120000,
    facePresent: true,
    proactiveQuotaOk: true,
    entropy: 0,
  });
  assert.equal(held.shouldSpeak, false);
  assert.equal(held.reason, 'presence_cooldown');
});

test('coldness phase can poke or tease after short replies', () => {
  const out = engine().decideColdness({
    now: 200000,
    lastUserText: '嗯',
    proactiveQuotaOk: true,
    entropy: 0.2,
  });
  assert.equal(out.shouldSpeak, true);
  assert.ok(['poke', 'tease'].includes(out.action));
  assert.equal(out.phase, 'coldness');
});

test('user presence decays ignored streak without treating it as a direct reply', () => {
  const e = engine();
  e.state.ignoredStreak = 3;
  const state = e.registerFeedback({ type: 'presence', text: '还在' });
  assert.equal(state.ignoredStreak, 2);
  assert.ok(state.engagedStreak >= 0);
});
