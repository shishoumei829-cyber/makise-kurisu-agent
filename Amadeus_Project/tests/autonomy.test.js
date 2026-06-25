'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');

const { DriveDynamics, Urge } = require('../digital_life/autonomy/drive_dynamics');
const { CuriosityEngine } = require('../digital_life/autonomy/curiosity');
const { CreativityModule } = require('../digital_life/autonomy/creativity');
const { AutonomousBehaviorLoop } = require('../digital_life/autonomy/behavior_loop');
const { AutonomySubsystem } = require('../digital_life/autonomy');
const { ACTION_INTENTS } = require('../digital_life/autonomy/constants');

test('DriveDynamics: CONNECTION rises with idle time', () => {
  const d = new DriveDynamics();
  d.tick(0, { pad: { P: 0, A: 0, S: 0.4 }, relScore: 0.4, idleMs: 45 * 60 * 1000 });
  assert.ok(d.activations.CONNECTION > 0.4);
});

test('DriveDynamics: cross-inhibition between CONNECTION and AUTONOMY', () => {
  const d = new DriveDynamics();
  d.activations.CONNECTION = 0.85;
  d.activations.AUTONOMY = 0.82;
  d._applyCrossInhibition();
  assert.ok(d.activations.CONNECTION < 0.85);
  assert.ok(d.activations.AUTONOMY < 0.82);
});

test('Urge: expires and effective intensity fades', () => {
  const u = new Urge({
    drive: 'CURIOSITY',
    intent: ACTION_INTENTS.ASK_QUESTION,
    intensity: 0.9,
    createdAt: Date.now() - 20 * 60 * 1000,
    expiresAt: Date.now() + 5 * 60 * 1000,
  });
  assert.ok(u.isActive());
  assert.ok(u.effectiveIntensity() < 0.9);
  u.satisfy('test');
  assert.equal(u.isActive(), false);
});

test('DriveDynamics: generates urges and behavior boosts', () => {
  const d = new DriveDynamics();
  d.tick(0, {
    pad: { P: 0.2, A: 0.6, S: 0.5, D: 0.3 },
    userText: '量子物理到底是什么',
    relScore: 0.5,
    motivationState: { curiosity: 0.8 },
  });
  d.generateUrges({ pad: { A: 0.6 }, relScore: 0.5, userText: '量子' });
  const urges = d.getActiveUrges();
  assert.ok(urges.length >= 1);
  const boosts = d.behaviorBoostsFromUrges();
  assert.ok(boosts.ENGAGE > 0 || boosts.APPROACH > 0);
});

test('CuriosityEngine: information gain higher for novel science topics', () => {
  const c = new CuriosityEngine();
  const memory = { events: [{ content: '聊过日常' }] };
  const gainSci = c.calculateInformationGain('量子纠缠', memory);
  const gainDaily = c.calculateInformationGain('日常', memory);
  c.ingestText('量子纠缠实验');
  c.ingestText('量子纠缠理论');
  const gaps = c.discoverKnowledgeGaps(memory, { get: () => ({ identity_tags: ['时间旅行'] }) });
  assert.ok(gainSci >= gainDaily);
  assert.ok(gaps.some((g) => g.topic === '时间旅行' || g.importance > 0.3));
});

test('CuriosityEngine: tracks open and answered questions', () => {
  const c = new CuriosityEngine();
  c.generateCuriousQuestions({ pad: { S: 0.4, A: 0.5 }, relScore: 0.3 }, { events: [] });
  const before = c.getOpenQuestions().length;
  assert.ok(before >= 1);
  const topic = c.getOpenQuestions()[0].topic;
  c.markAnswered(topic, `关于${topic}其实是因为工作压力大`);
  assert.ok(c.getOpenQuestions().every((q) => q.topic !== topic || q.answered));
});

test('CreativityModule: rejects off-persona ideas', () => {
  const cr = new CreativityModule();
  const idea = cr.generateIdea({ pad: { A: 0.5 }, userText: '今天好可爱呀宝贝' });
  assert.ok(!/可爱|宝贝/.test(idea));
});

test('CreativityModule: builds associations across turns', () => {
  const cr = new CreativityModule();
  cr.learnAssociation('量子', '意识', 2);
  const m = cr.generateMetaphor('量子');
  assert.ok(m.length > 0);
  assert.ok(cr.associations.has('量子|意识') || cr.associations.size > 0);
});

test('AutonomousBehaviorLoop: suppresses proactive under DND', () => {
  const d = new DriveDynamics();
  d.tick(0, { pad: { P: 0, A: 0, S: 0.6 }, relScore: 0.6, idleMs: 60 * 60 * 1000 });
  d.generateUrges({ relScore: 0.6, idleMs: 60 * 60 * 1000 });
  const loop = new AutonomousBehaviorLoop(d, new CuriosityEngine(), new CreativityModule());
  const decision = loop.execute({
    isAutonomyTick: true,
    idleMs: 60 * 60 * 1000,
    pad: { P: 0.1, A: 0.2, S: 0.6 },
    relScore: 0.6,
    dnd: true,
  });
  assert.equal(decision.shouldAct, false);
  assert.equal(decision.suppressProactive, true);
});

test('AutonomousBehaviorLoop: REACH_OUT may speak after long idle', () => {
  const d = new DriveDynamics();
  for (let i = 0; i < 3; i++) {
    d.tick(60000, { pad: { P: 0.1, A: 0.2, S: 0.55 }, relScore: 0.5, idleMs: 50 * 60 * 1000 });
  }
  d.generateUrges({ relScore: 0.5, idleMs: 50 * 60 * 1000 });
  const loop = new AutonomousBehaviorLoop(d, new CuriosityEngine(), new CreativityModule());
  let spoke = false;
  for (let i = 0; i < 20; i++) {
    const decision = loop.execute({
      isAutonomyTick: true,
      idleMs: 50 * 60 * 1000,
      pad: { P: 0.1, A: 0.3, S: 0.55 },
      relScore: 0.5,
      proactiveQuotaOk: true,
    });
    if (decision.shouldAct) spoke = true;
  }
  assert.ok(spoke, 'expected at least one speak decision in 20 trials');
});

test('AutonomySubsystem: conversation turn yields goal seeds and prompt', () => {
  const sub = new AutonomySubsystem();
  const out = sub.onConversationTurn({
    pad: { P: 0.1, A: 0.55, S: 0.45, D: 0.3 },
    memory: {
      getRelationshipScore: () => 0.45,
      events: [{ type: 'scientific', content: '量子计算' }],
    },
    motivationState: { curiosity: 0.7 },
    userText: '你觉得量子计算能模拟意识吗',
    selfModel: { get: () => ({ identity_tags: ['神经科学'], relationship_perception: '还不太熟' }) },
    relScore: 0.45,
  });
  assert.ok(Object.keys(out.behaviorBoosts).length > 0);
  assert.ok(out.goalSeeds.length >= 1);
  assert.ok(out.promptBlock.includes('内驱') || out.promptBlock.includes('好奇') || out.promptBlock.includes('创造'));
});
