'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');

const { BeliefRevision } = require('../digital_life/metacognition/belief_revision');
const { ReflectionLoop } = require('../digital_life/metacognition/reflection_loop');
const { MetacognitionSubsystem } = require('../digital_life/metacognition');

test('BeliefRevision: intimate event reduces user unknown', () => {
  const b = new BeliefRevision();
  const before = b.beliefs.get('user_unknown').confidence;
  b.updateWorldview({ type: 'intimate', content: '喜欢你' });
  assert.ok(b.beliefs.get('user_unknown').confidence > before);
});

test('ReflectionLoop: detects habit bias', () => {
  const r = new ReflectionLoop();
  for (let i = 0; i < 6; i++) {
    r.reflectOnDecision({ action: 'WITHDRAW', reasoning: '回避', factors: ['fear'] });
  }
  assert.ok(r.biases.some((x) => x.type === 'habit'));
  const insight = r.generateInsight();
  assert.ok(insight?.content);
});

test('MetacognitionSubsystem: surfaces insight on turn', () => {
  const m = new MetacognitionSubsystem();
  for (let i = 0; i < 5; i++) {
    m.reflection.reflectOnDecision({ action: 'ENGAGE', reasoning: 'test', factors: ['a'] });
  }
  const out = m.onConversationTurn({
    mainEvent: { type: 'scientific', content: '物理' },
    decision: { action: 'ENGAGE', reasoning: '好奇', factors: ['curiosity'] },
    chatTurnCounter: 8,
    chatMinimal: true,
  });
  assert.ok(out.insight || out.beliefLines.length >= 1);
});
