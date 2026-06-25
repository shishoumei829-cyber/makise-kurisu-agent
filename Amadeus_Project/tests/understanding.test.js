'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');

const { EmotionalResonance } = require('../digital_life/understanding/emotional_resonance');
const { MentalModel } = require('../digital_life/understanding/mental_model');
const { SubtextDetector } = require('../digital_life/understanding/subtext');
const { UnderstandingSubsystem } = require('../digital_life/understanding');

test('EmotionalResonance: aggressive triggers shield mode', () => {
  const er = new EmotionalResonance();
  const rec = er.recognizeEmotion('你闭嘴滚');
  assert.equal(rec.emotion, 'aggressive');
  const delta = er.adjustEmotionalState(rec, 0.3);
  assert.ok(delta.D > 0);
  assert.equal(er.regulationMode, 'shield');
});

test('MentalModel: extracts beliefs and preferences', () => {
  const mm = new MentalModel();
  mm.ingestUserText('我觉得量子物理很有趣，我想要了解更多');
  assert.ok(mm.beliefs.length >= 1);
  assert.ok(mm.desires.length >= 1);
  assert.ok(mm.topPreferences().some((p) => p.topic.includes('量子')));
});

test('SubtextDetector: detects withdrawal subtext', () => {
  const st = new SubtextDetector();
  const analysis = st.analyze('算了不说了', { emotion: 'negative', intensity: 0.6 });
  assert.ok(analysis.hits.some((h) => h.label === '撤回'));
  assert.ok(analysis.primaryNeed.includes('安全'));
});

test('UnderstandingSubsystem: full turn produces pad delta and lines', () => {
  const u = new UnderstandingSubsystem();
  const out = u.onConversationTurn({
    userText: '你怎么不回我，算了随便吧',
    userModel: { model: { patterns: { emotion_history: [] }, relationship: { closeness: 0.4 } } },
    closeness: 0.4,
  });
  assert.ok(out.recognized);
  assert.ok(out.resonanceLine || out.subtextLine);
  assert.ok(out.subtextLine.includes('言外之意') || out.pendingNeed);
});
