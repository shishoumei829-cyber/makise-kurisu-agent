'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');

const { ReinforcementLearning } = require('../learning_engine');
const { MemoryReorganization } = require('../digital_life/memory_reorganization');
const { DreamEngine } = require('../digital_life/dream_engine');
const { BeliefRevision } = require('../digital_life/belief_revision');
const { EmotionalResonance } = require('../digital_life/emotional_resonance');
const { AutonomousBehaviorEngine } = require('../digital_life/drive_engine');
const { DigitalLifeOrchestrator } = require('../digital_life');

test('ReinforcementLearning.buildStateKey should bucket PAD and relationship', () => {
  const rl = new ReinforcementLearning();
  const key = rl.buildStateKey({ P: 0.5, A: 0.4, D: 0, S: 0.6 }, 0.5);
  assert.match(key, /^P\+_A\+_R\+$/);
});

test('MemoryReorganization should discover associations from events', () => {
  const reorg = new MemoryReorganization();
  const events = [
    { type: 'scientific', content: '讨论量子物理实验' },
    { type: 'scientific', content: '量子纠缠理论' },
    { type: 'positive', content: '今天心情不错' },
  ];
  const found = reorg.discoverAssociations(events);
  assert.ok(found.length >= 1);
});

test('DreamEngine should dream after sufficient idle', () => {
  const dream = new DreamEngine();
  assert.equal(dream.shouldDream(10 * 60 * 1000), false);
  assert.equal(dream.shouldDream(30 * 60 * 1000), true);
  const d = dream.generateDream({ insights: ['记忆碎片'], pad: { P: 0.1 }, idleMin: 30 });
  assert.ok(d.text.length > 0);
});

test('BeliefRevision should update on worldview events', () => {
  const beliefs = new BeliefRevision();
  beliefs.updateWorldview({ type: 'intimate', content: '喜欢你' });
  const user = beliefs.beliefs.get('user_unknown');
  assert.ok(user.confidence > 0.7);
});

test('EmotionalResonance should mirror negative emotion', () => {
  const er = new EmotionalResonance();
  const rec = er.recognizeEmotion('我今天很难过');
  assert.equal(rec.emotion, 'negative');
  const delta = er.adjustEmotionalState(rec, 0.5);
  assert.ok(delta.P < 0);
});

test('AutonomousBehaviorEngine should produce behavior boosts', () => {
  const drive = new AutonomousBehaviorEngine();
  drive.updateInternalState({ P: 0.3, A: 0.6, S: 0.5, D: 0.4 }, {
    getRelationshipScore: () => 0.6,
    events: [{ content: '科学实验' }],
  }, { curiosity: 0.8 });
  drive.generateUrge();
  const boosts = drive.behaviorBoosts();
  assert.ok(Object.keys(boosts).length > 0);
});

test('DigitalLifeOrchestrator onUserTurn returns drive boosts and goal seeds', () => {
  const dl = new DigitalLifeOrchestrator();
  const out = dl.onUserTurn({
    pad: { P: 0, A: 0.4, S: 0.3, D: 0.3 },
    memory: { getRelationshipScore: () => 0.2, events: [] },
    motivationState: { curiosity: 0.6 },
    userText: '量子物理有意思吗',
    userModel: { model: { patterns: { emotion_history: [] }, relationship: { closeness: 0.3 } } },
    mainEvent: { type: 'scientific', content: '量子' },
    selfModel: { get: () => ({ identity_tags: [], relationship_perception: '' }) },
    relScore: 0.2,
  });
  assert.ok(out.driveBoosts);
  assert.ok(out.goalSeeds?.length >= 0);
  assert.ok(out.recognized.emotion);
});
