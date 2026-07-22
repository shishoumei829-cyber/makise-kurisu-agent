'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const os = require('os');
const path = require('path');
const fs = require('fs');

const {
  GlobalWorkspace,
  collectCandidates,
  compete,
  CONTENT_KINDS,
} = require('../brain/workspace');
const { ConsciousnessLayer } = require('../brain/consciousness');
const { BrainSelfModel } = require('../brain/selfModel');
const { createDefaultAxioms } = require('../brain/axioms/fromSoul');

function debounce(ms, fn) {
  let t = null;
  return () => {
    if (t) clearTimeout(t);
    t = setTimeout(fn, ms);
  };
}

test('collectCandidates: intimacy raises percept salience', () => {
  const items = collectCandidates({
    perceived: { userContent: '有点想你', cognitiveInput: '有点想你' },
    worldSnapshot: { partner: { isOkabe: true, name: '冈部' }, relationship: { score: 0.5 } },
    selfSnapshot: { axioms: createDefaultAxioms(), tensions: {} },
    pad: { P: 0.2, A: 0.3, S: 0.5 },
  });
  const percept = items.find((i) => i.kind === CONTENT_KINDS.PERCEPT && /亲密/.test(i.content));
  assert.ok(percept);
  assert.ok(percept.salience >= 0.85);
});

test('collectCandidates: physical request boosts self axiom', () => {
  const items = collectCandidates({
    perceived: { userContent: '帮我带杯咖啡', cognitiveInput: '帮我带杯咖啡' },
    worldSnapshot: { partner: { isOkabe: true }, relationship: { score: 0.4 } },
    selfSnapshot: { axioms: createDefaultAxioms(), tensions: { physical_promise: 3 } },
    pad: { P: 0, A: 0.2, S: 0.3 },
  });
  const self = items.find((i) => i.kind === CONTENT_KINDS.SELF && /物理效应器/.test(i.content));
  assert.ok(self);
  assert.ok(self.salience >= 0.9);
  const meta = items.find((i) => i.kind === CONTENT_KINDS.META);
  assert.ok(meta);
});

test('compete: caps broadcast and diversifies kinds', () => {
  const many = [];
  for (let i = 0; i < 8; i++) {
    many.push({
      id: `d${i}`,
      kind: CONTENT_KINDS.DRIVE,
      content: `drive ${i}`,
      salience: 0.9 - i * 0.01,
      source: 't',
    });
  }
  many.push({
    id: 'p1',
    kind: CONTENT_KINDS.PERCEPT,
    content: '他说话了',
    salience: 0.95,
    source: 't',
  });
  const selected = compete(many, { maxBroadcast: 5 });
  assert.ok(selected.length <= 5);
  const drives = selected.filter((s) => s.kind === CONTENT_KINDS.DRIVE);
  assert.ok(drives.length <= 2);
  assert.ok(selected.some((s) => s.kind === CONTENT_KINDS.PERCEPT));
});

test('GlobalWorkspace: update produces narrative and broadcast', () => {
  const ws = new GlobalWorkspace();
  const snap = ws.update({
    perceived: { userContent: '在干嘛', cognitiveInput: '在干嘛', idleMsSinceUser: 0 },
    worldSnapshot: {
      partner: { isOkabe: true, name: '冈部' },
      relationship: { score: 0.5 },
      dialogue: { logExcerpt: 'hi', entryCount: 2 },
    },
    selfSnapshot: { axioms: createDefaultAxioms(), tensions: {} },
    pad: { P: 0.1, A: 0.2, S: 0.4 },
    autonomyPublic: {
      drives: {
        activations: { CURIOSITY: 0.8, CONNECTION: 0.4 },
        urgeQueue: [],
      },
    },
  });
  assert.ok(snap.broadcast.length >= 1);
  assert.ok(snap.narrative.length > 0);
  assert.ok(ws.toPromptBlock(snap).includes('意识广播'));
});

test('ConsciousnessLayer: cycle exposes public state', () => {
  const layer = new ConsciousnessLayer();
  const cycle = layer.cycle({
    perceived: { userContent: '帮我拿咖啡', cognitiveInput: '帮我拿咖啡' },
    worldSnapshot: { partner: { isOkabe: true, name: '冈部' }, relationship: { score: 0.5 } },
    selfSnapshot: { axioms: createDefaultAxioms(), tensions: {} },
    pad: { P: 0, A: 0.3, S: 0.4 },
  });
  assert.equal(cycle.intentionHint.intent, 'self_boundary');
  const pub = layer.toPublicState(cycle);
  assert.ok(pub.broadcast.length >= 1);
  assert.ok(pub.metrics.coherence > 0);
  assert.ok(layer.toPromptBlock(cycle).includes('自我') || layer.toPromptBlock(cycle).includes('物理'));
});

test('ConsciousnessLayer: proactive mode sets intention', () => {
  const layer = new ConsciousnessLayer();
  const cycle = layer.cycle({
    perceived: {
      userContent: '',
      autonomyInitiative: true,
      idleMsSinceUser: 15 * 60 * 1000,
    },
    worldSnapshot: { partner: { isOkabe: true }, relationship: { score: 0.5 } },
    selfSnapshot: { axioms: createDefaultAxioms(), tensions: {} },
    pad: { P: 0.1, A: 0.2, S: 0.5 },
    autonomyPublic: {
      drives: {
        activations: { CONNECTION: 0.82, CURIOSITY: 0.3 },
        urgeQueue: [{ id: 'u1', drive: 'CONNECTION', intensity: 0.8, promptHint: '想知道他在干嘛' }],
      },
    },
  });
  assert.equal(cycle.intentionHint.intent, 'proactive_from_drive');
  assert.equal(cycle.workspace.mode, 'proactive');
});

test('BrainSelfModel + workspace: tension enters consciousness', () => {
  const tmp = path.join(os.tmpdir(), `cons-${Date.now()}`);
  fs.mkdirSync(tmp, { recursive: true });
  const sm = new BrainSelfModel(path.join(tmp, 's.json'), debounce);
  sm.recordTension('physical_promise', 4);
  const layer = new ConsciousnessLayer();
  const cycle = layer.cycle({
    perceived: { userContent: '嗯', cognitiveInput: '嗯' },
    worldSnapshot: { partner: { isOkabe: true }, relationship: { score: 0.3 } },
    selfSnapshot: sm.snapshot(),
    pad: { P: 0, A: 0, S: 0.2 },
  });
  assert.ok(cycle.workspace.broadcast.some((b) => b.kind === 'meta' || b.source === 'learner'));
});
