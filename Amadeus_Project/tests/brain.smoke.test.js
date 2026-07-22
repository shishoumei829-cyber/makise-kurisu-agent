'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const os = require('os');
const path = require('path');

const { Brain } = require('../brain');
const { BrainSelfModel } = require('../brain/selfModel');
const { WorldModel } = require('../brain/worldModel');
const { BrainLearner } = require('../brain/learner');
const { perceiveIncomingChat } = require('../brain/perceive');
const { UnifiedDialogueLog } = require('../lib/unifiedDialogueLog');

function debounce(ms, fn) {
  let t = null;
  return () => {
    if (t) clearTimeout(t);
    t = setTimeout(fn, ms);
  };
}

test('Brain: constructs with brain modules', () => {
  const tmpDir = path.join(os.tmpdir(), `amadeus-brain-${Date.now()}`);
  const log = new UnifiedDialogueLog(tmpDir);
  const state = {
    currentPAD: { P: 0, A: 0, D: 0, S: 0 },
    chatTurnCounter: 0,
    lastChatBehaviorId: '',
    lastRlStateKey: '',
    lastDigitalLifeTurn: null,
  };
  const brainSelfModel = new BrainSelfModel(path.join(tmpDir, 'self_v2.json'), debounce(50, () => {}));
  const worldModel = new WorldModel();
  const brainLearner = new BrainLearner(brainSelfModel);
  const brain = new Brain({
    runtime: {
      unifiedDialogueLog: log,
      needsLongTermMemory: () => false,
      memorySystem: { getRelationshipScore: () => 0 },
      whoamiPath: path.join(tmpDir, 'w.json'),
      getLastVision: () => ({}),
      brainSelfModel,
      worldModel,
      brainLearner,
      state,
    },
  });
  assert.ok(brain);
  assert.ok(brain.runtime.brainPipeline);
  assert.equal(typeof brain.turn, 'function');
});

test('perceiveIncomingChat: parses user message', () => {
  const tmpDir = path.join(os.tmpdir(), `amadeus-brain-p-${Date.now()}`);
  const log = new UnifiedDialogueLog(tmpDir);
  const out = perceiveIncomingChat(
    { messages: [{ role: 'user', content: '在干嘛' }] },
    { unifiedDialogueLog: log, needsLongTermMemory: () => false },
  );
  assert.equal(out.userContent, '在干嘛');
  assert.equal(out.cognitiveInput, '在干嘛');
  assert.equal(out.useLongTermMemory, false);
  assert.equal(log.entriesCount, 0);
});

test('perceiveIncomingChat: imports history only when explicitly authorized', () => {
  const tmpDir = path.join(os.tmpdir(), `amadeus-brain-import-${Date.now()}`);
  const log = new UnifiedDialogueLog(tmpDir);
  perceiveIncomingChat(
    { allowHistoryImport: true, messages: [{ role: 'user', content: '真实旧对话' }] },
    { unifiedDialogueLog: log, needsLongTermMemory: () => false },
  );
  assert.equal(log.entriesCount, 1);
});

test('Brain pipeline: monitor blocks physical draft', async () => {
  const tmpDir = path.join(os.tmpdir(), `amadeus-brain-m-${Date.now()}`);
  const log = new UnifiedDialogueLog(tmpDir);
  const state = { currentPAD: { P: 0, A: 0, D: 0, S: 0 }, chatTurnCounter: 0, lastChatBehaviorId: '', lastRlStateKey: '', lastDigitalLifeTurn: null };
  const brainSelfModel = new BrainSelfModel(path.join(tmpDir, 'sv2.json'), debounce(50, () => {}));
  const worldModel = new WorldModel();
  const brain = new Brain({
    runtime: {
      unifiedDialogueLog: log,
      needsLongTermMemory: () => false,
      memorySystem: { getRelationshipScore: () => 0.5 },
      whoamiPath: path.join(tmpDir, 'w.json'),
      getLastVision: () => ({}),
      brainSelfModel,
      worldModel,
      brainLearner: new BrainLearner(brainSelfModel),
      state,
    },
  });
  process.env.AMADEUS_BRAIN_MONITOR = '1';
  const out = await brain.runtime.brainPipeline.processReply('好，顺路给你带咖啡。', {
    userText: '带杯咖啡',
    oocOpts: {},
  });
  assert.ok(!/顺路给你带/.test(out));
  delete process.env.AMADEUS_BRAIN_MONITOR;
});
