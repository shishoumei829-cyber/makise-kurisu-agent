'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const os = require('os');
const path = require('path');
const fs = require('fs');

const { UnifiedDialogueLog } = require('../lib/unifiedDialogueLog');
const { InnerStateSix } = require('../cognitive/innerStateSix');
const { BehaviorIngest } = require('../lib/behaviorIngest');
const { validateJapaneseLine, validateChineseReply } = require('../cognitive/japanesePipeline');
const { buildSocialIdentityPrompt } = require('../cognitive/socialIdentity');
const { buildExpressionVariantBlock } = require('../cognitive/expressionVariants');
const { buildPrompt, ANCHOR } = require('../cognitive/prompts');
const { buildClientContextBlock } = require('../lib/clientContext');
const { buildJapaneseFirstPrompt, runJapaneseFirstPipeline } = require('../cognitive/japanesePipeline');

const tmpDir = path.join(os.tmpdir(), `amadeus-arch-${Date.now()}`);

test('UnifiedDialogueLog: allows same-side consecutive entries', () => {
  const log = new UnifiedDialogueLog(tmpDir);
  log.append('assistant', '第一句');
  log.append('assistant', '第二句');
  assert.equal(log.getRecent(2).length, 2);
  assert.equal(log.getRecent(2)[0].text, '第一句');
});

test('UnifiedDialogueLog: toOllamaDialogue merges adjacent same role', () => {
  const log = new UnifiedDialogueLog(tmpDir);
  log.append('user', '你好');
  log.append('user', '在吗');
  log.append('assistant', '嗯');
  const dlg = log.toOllamaDialogue({ maxMsgs: 10 });
  assert.equal(dlg.length, 2);
  assert.match(dlg[0].content, /你好/);
  assert.match(dlg[0].content, /在吗/);
});

test('InnerStateSix: persists and updates from turn', () => {
  const inner = new InnerStateSix(tmpDir);
  const before = inner.state.connection;
  inner.updateFromTurn({
    pad: { P: 0.3, A: 0.4, S: 0.6 },
    relScore: 0.5,
    userEmotion: 'intimate',
    mainEvent: { type: 'positive' },
    userText: '想你',
    behaviorId: 'APPROACH',
  });
  assert.ok(inner.state.connection >= before);
  const inner2 = new InnerStateSix(tmpDir);
  assert.ok(inner2.state.connection >= before);
});

test('BehaviorIngest: converts events to natural language', () => {
  const bi = new BehaviorIngest(tmpDir);
  const out = bi.ingest({
    events: [{ app: '抖音', durationMin: 32 }, { app: '微信', durationMin: 5 }],
  });
  assert.ok(out.summary.includes('抖音'));
});

test('Japanese pipeline: catches addressee issues', () => {
  const v = validateJapaneseLine('岡部さんは誰ですか', { partnerName: '冈部' });
  assert.equal(v.ok, false);
});

test('Chinese reply validation: catches 那还能是谁', () => {
  const v = validateChineseReply('那还能是谁？', '实录', '冈部');
  assert.equal(v.ok, false);
});

test('Social identity: lover cues on intimate text', () => {
  const block = buildSocialIdentityPrompt({
    userText: '我有点想你',
    pad: { S: 0.6 },
    relScore: 0.5,
    recentEvents: [],
  });
  assert.match(block, /恋人模式/);
});

test('Expression variants: produces band block', () => {
  const block = buildExpressionVariantBlock({ P: 0.4, A: 0.3, D: 0, S: 0.6 }, { state: { connection: 0.65 } });
  assert.match(block, /表达变体/);
});

test('ANCHOR includes motivation item 11', () => {
  assert.match(ANCHOR, /11\./);
  assert.match(ANCHOR, /核心动机/);
});

test('buildPrompt: single path includes digital life without clientPersona gate', () => {
  const p = buildPrompt({
    soulContent: '传记',
    emotion: { P: 0, A: 0, D: 0, S: 0.5 },
    relationship: { closeness: 0.3, trust: 0.5 },
    digitalLifeCtx: '生命层测试',
    innerStateSixBlock: '六维测试',
    socialIdentityBlock: '身份测试',
    clientContextBlock: '【视线里】他在看手机',
  });
  assert.match(p, /生命层测试/);
  assert.match(p, /六维测试/);
  assert.match(p, /身份测试/);
  assert.match(p, /视线里/);
  assert.match(p, /最高人格指令/);
});

test('ClientContext: vision and palace blocks', () => {
  const block = buildClientContextBlock({
    vision: '他正看着屏幕',
    palace: '[HALL]\n测试记忆',
    wantLongMemory: true,
    situation: '他在和你说话',
  });
  assert.match(block, /视线里/);
  assert.match(block, /记忆宫殿/);
  assert.match(block, /当下情境/);
});

test('ClientContext: skipTopic flag', () => {
  const block = buildClientContextBlock({ skipTopic: true });
  assert.match(block, /换题意图/);
});

test('JapaneseFirst: prompt includes user and log', () => {
  const { system, user } = buildJapaneseFirstPrompt({
    userText: '在干嘛',
    conversationLog: '他: 你好',
    partnerName: '冈部',
  });
  assert.match(system, /牧濑红莉栖/);
  assert.match(user, /在干嘛/);
  assert.match(user, /冈部/);
});

test('JapaneseFirst: pipeline validates bad jp', async () => {
  const out = await runJapaneseFirstPipeline({
    userText: '你好',
    conversationLog: '',
    partnerName: '冈部',
    generateJapanese: async () => '岡部さんは誰ですか',
    translateToChinese: async () => '冈部是谁',
  });
  assert.equal(out.ok, false);
});
