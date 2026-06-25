'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');

const { MemoryConsolidation } = require('../digital_life/evolution/memory_consolidation');
const { DreamEngine } = require('../digital_life/evolution/dream_engine');
const { PersonalityTrajectory } = require('../digital_life/evolution/personality_trajectory');
const { RlBridge } = require('../digital_life/evolution/rl_bridge');
const { EvolutionSubsystem } = require('../digital_life/evolution');
const { ReinforcementLearning } = require('../learning_engine');

test('MemoryConsolidation: segments episodes and discovers associations', () => {
  const mem = new MemoryConsolidation();
  const events = [
    { type: 'scientific', content: '讨论量子物理实验', timestamp: 1000 },
    { type: 'scientific', content: '量子纠缠理论', timestamp: 2000 },
    { type: 'positive', content: '今天心情不错', timestamp: 3000 },
  ];
  const eps = mem.segmentEpisodes(events);
  assert.ok(eps.length >= 1);
  const found = mem.discoverAssociations(events);
  assert.ok(found.length >= 1);
});

test('DreamEngine: REM phase and carryover on long idle', () => {
  const dream = new DreamEngine();
  assert.equal(dream.shouldDream(30 * 60 * 1000), true);
  const d = dream.generateDream({
    insights: ['记忆碎片'],
    pad: { P: -0.1, A: 0.5 },
    idleMin: 95,
    associations: [{ a: '量子', b: '实验' }],
  });
  assert.ok(d.text.includes('REM') || d.phase);
  assert.ok(dream.getCarryover());
});

test('PersonalityTrajectory: shifts on scientific events', () => {
  const p = new PersonalityTrajectory();
  const before = p.traits.openness;
  p.updateFromEvent({ type: 'scientific', content: '论文与实验' });
  assert.ok(p.traits.openness >= before);
  assert.ok(p.getDescription().length > 0);
});

test('RlBridge: records turn with ReinforcementLearning', () => {
  const rl = new ReinforcementLearning();
  const bridge = new RlBridge();
  bridge.recordTurn(rl, {
    pad: { P: 0.2, A: 0.3 },
    relScore: 0.4,
    behaviorId: 'ENGAGE',
    userText: '谢谢你，我很开心',
    recognizedEmotion: { emotion: 'positive', intensity: 0.7 },
  });
  assert.ok(bridge.lastStateKey);
  assert.ok(typeof bridge.lastReward === 'number');
});

test('EvolutionSubsystem: idle cycle consolidates and dreams', () => {
  const evo = new EvolutionSubsystem();
  const memorySystem = {
    events: [
      { type: 'scientific', content: '量子物理' },
      { type: 'scientific', content: '量子实验' },
      { type: 'scientific', content: '物理理论' },
      { type: 'positive', content: '开心' },
    ],
    getRecentSignificant: () => ['[x] 重要'],
    addObservation: () => {},
  };
  evo._lastConsolidation = 0;
  const result = evo.runIdleCycle({
    idleMs: 35 * 60 * 1000,
    pad: { P: 0, A: 0.2 },
    memorySystem,
  });
  assert.equal(result.consolidated, true);
  assert.ok(result.insights.length >= 1);
  assert.ok(result.dream);
});
