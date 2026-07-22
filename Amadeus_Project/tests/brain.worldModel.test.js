'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const path = require('path');
const os = require('os');

const { WorldModel } = require('../brain/worldModel');
const { UnifiedDialogueLog } = require('../lib/unifiedDialogueLog');
const { createDefaultAxioms } = require('../brain/axioms/fromSoul');
const monitor = require('../brain/monitor');
const { localReviseDraft } = require('../brain/deliberation');
const probes = require('../brain/probes/capability.json');

function makeDeps(tmpDir, extra = {}) {
  const log = new UnifiedDialogueLog(tmpDir);
  return {
    unifiedDialogueLog: log,
    memorySystem: { getRelationshipScore: () => 0.42 },
    whoamiPath: path.join(tmpDir, 'whoami.json'),
    getLastVision: () => ({ description: '他看着屏幕', timestamp: Date.now() }),
    ...extra,
  };
}

test('WorldModel: aggregates partner and dialogue', () => {
  const tmp = path.join(os.tmpdir(), `wm-${Date.now()}`);
  const wm = new WorldModel();
  const log = new UnifiedDialogueLog(tmp);
  log.append('user', '在吗');
  const snap = wm.update(
    { userContent: '在吗', cognitiveInput: '在吗', useLongTermMemory: false, autonomyInitiative: false },
    { clientContext: { situation: '他在说话' } },
    makeDeps(tmp, { unifiedDialogueLog: log }),
  );
  assert.equal(snap.userText, '在吗');
  assert.ok(snap.dialogue.entryCount >= 1);
  assert.ok(wm.toPromptSummary(snap).includes('世界模型'));
});

test('WorldModel: vision active when client reports vision', () => {
  const tmp = path.join(os.tmpdir(), `wm-v-${Date.now()}`);
  const wm = new WorldModel();
  const snap = wm.update(
    { userContent: '你看到什么', useLongTermMemory: false },
    { clientContext: { vision: '屏幕上是代码' } },
    makeDeps(tmp),
  );
  assert.equal(snap.vision.active, true);
  assert.match(snap.vision.description, /代码/);
});

test('WorldModel: relationship score from memory', () => {
  const tmp = path.join(os.tmpdir(), `wm-r-${Date.now()}`);
  const wm = new WorldModel();
  const snap = wm.update(
    { userContent: '嗯', useLongTermMemory: false },
    {},
    makeDeps(tmp, { memorySystem: { getRelationshipScore: () => 0.8 } }),
  );
  assert.ok(snap.relationship.score >= 0.38);
});

test('WorldModel: client situation in snapshot', () => {
  const tmp = path.join(os.tmpdir(), `wm-s-${Date.now()}`);
  const wm = new WorldModel();
  const snap = wm.update(
    { userContent: '忙吗', useLongTermMemory: false },
    { clientContext: { situation: '深夜还在写代码' } },
    makeDeps(tmp),
  );
  assert.match(snap.clientContext.situation, /深夜/);
});

test('WorldModel: getSnapshot returns copy', () => {
  const tmp = path.join(os.tmpdir(), `wm-c-${Date.now()}`);
  const wm = new WorldModel();
  wm.update({ userContent: 'a', useLongTermMemory: false }, {}, makeDeps(tmp));
  const a = wm.getSnapshot();
  const b = wm.getSnapshot();
  assert.notEqual(a, b);
  assert.equal(a.userText, 'a');
});

test('WorldModel: digital life turn attached', () => {
  const tmp = path.join(os.tmpdir(), `wm-d-${Date.now()}`);
  const wm = new WorldModel();
  const snap = wm.update(
    { userContent: 'hi', useLongTermMemory: false },
    {},
    {
      ...makeDeps(tmp),
      digitalLifeTurn: { resonanceLine: '共情线', subtextLine: '言外', pendingNeed: '陪伴' },
    },
  );
  assert.equal(snap.digitalLife.resonanceLine, '共情线');
});

test('WorldModel: long term memory channel flag', () => {
  const tmp = path.join(os.tmpdir(), `wm-l-${Date.now()}`);
  const wm = new WorldModel();
  const snap = wm.update(
    { userContent: '昨天我们说了什么', useLongTermMemory: true },
    {},
    makeDeps(tmp),
  );
  assert.equal(snap.channels.longTermMemory, true);
});

test('WorldModel: proactive thread fields', () => {
  const tmp = path.join(os.tmpdir(), `wm-p-${Date.now()}`);
  const wm = new WorldModel();
  const snap = wm.update(
    {
      userContent: '嗯',
      replyingToProactive: true,
      proactiveAnchor: '刚才问你在干嘛',
      useLongTermMemory: false,
    },
    {},
    makeDeps(tmp),
  );
  assert.equal(snap.replyingToProactive, true);
  assert.match(snap.proactiveAnchor, /干嘛/);
});

test('WorldModel: toPromptSummary under 220 chars', () => {
  const tmp = path.join(os.tmpdir(), `wm-sum-${Date.now()}`);
  const wm = new WorldModel();
  const snap = wm.update(
    { userContent: 'test', useLongTermMemory: false },
    { clientContext: { situation: '测试' } },
    makeDeps(tmp),
  );
  const s = wm.toPromptSummary(snap);
  assert.ok(s.length <= 220);
});

test('WorldModel: autonomy initiative flag', () => {
  const tmp = path.join(os.tmpdir(), `wm-a-${Date.now()}`);
  const wm = new WorldModel();
  const snap = wm.update(
    { userContent: '（想说话）', autonomyInitiative: true, useLongTermMemory: true },
    {},
    makeDeps(tmp),
  );
  assert.equal(snap.autonomyInitiative, true);
});
