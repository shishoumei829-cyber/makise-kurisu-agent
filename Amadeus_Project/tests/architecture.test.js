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
const {
  utteranceFocusLine, isContinuityFollowup, resolveFollowupForModel, buildContinuityRepairMessages,
} = require('../cognitive/replyAlign');
const { buildJapaneseFirstPrompt, runJapaneseFirstPipeline } = require('../cognitive/japanesePipeline');

const tmpDir = path.join(os.tmpdir(), `amadeus-arch-${Date.now()}`);

test('UnifiedDialogueLog: allows same-side consecutive entries', () => {
  const log = new UnifiedDialogueLog(tmpDir);
  log.append('assistant', '第一句');
  log.append('assistant', '第二句');
  assert.equal(log.getRecent(2).length, 2);
  assert.equal(log.getRecent(2)[0].text, '第一句');
});

test('UnifiedDialogueLog: rejects stale or duplicate assistant commits by turn ownership', () => {
  const log = new UnifiedDialogueLog(tmpDir);
  log.entries = [];
  assert.ok(log.append('user', '第一问', { conversationId: 'c1', turnId: 't1' }));
  assert.ok(log.append('user', '第二问', { conversationId: 'c1', turnId: 't2' }));
  assert.equal(log.append('assistant', '迟到的第一答', { conversationId: 'c1', turnId: 't1' }), null);
  assert.ok(log.append('assistant', '第二答', { conversationId: 'c1', turnId: 't2' }));
  assert.equal(log.append('assistant', '重复第二答', { conversationId: 'c1', turnId: 't2' }), null);
});

test('utterance focus must catch story endings that resolve earlier worry', () => {
  const focus = utteranceFocusLine(
    '我昨天电脑坏了，给我心慌了一个晚上没睡着，现在早上起来，发现电脑好了',
  );
  assert.match(focus, /结尾已经说明结果/);
  assert.match(focus, /还在担心吗/);
});

test('Turn continuity: a follow-up must resolve the assistant previous claim', () => {
  const focus = utteranceFocusLine('\u8bb2\u8bb2\u770b', {
    lastAssistant: '\u6211\u4e5f\u6709\u8fc7\u7c7b\u4f3c\u7684\u7ecf\u5386\u3002',
  });
  assert.match(focus, /\u4e0a\u4e00\u53e5\u627f\u8bfa\u5fc5\u987b\u5151\u73b0/);
  assert.match(focus, /\u7f3a\u4e4f\u771f\u5b9e\u4f9d\u636e/);
  const resolved = resolveFollowupForModel('\u8bb2\u8bb2\u770b', '\u6211\u4e5f\u6709\u8fc7\u7c7b\u4f3c\u7ecf\u5386\u3002');
  assert.match(resolved, /\u3053\u306e\u767a\u8a71\u306e\u53c2\u7167\u5148/);
  assert.match(resolved, /\u7e70\u308a\u8fd4\u3057/);
  const legacy = fs.readFileSync(path.join(__dirname, '..', 'brain', 'legacyChat.js'), 'utf8');
  assert.match(legacy, /AMADEUS_CONTINUITY_MODEL/);
  assert.match(legacy, /if \(continuityModel\) model = continuityModel/);
  const repairMessages = buildContinuityRepairMessages({
    userText: '\u8bb2\u8bb2\u770b',
    lastAssistant: '\u6211\u4e5f\u6709\u8fc7\u7c7b\u4f3c\u7ecf\u5386\u3002',
    previousUser: '\u6628\u5929\u8fd0\u52a8\u592a\u72e0\u4e86',
  });
  assert.equal(repairMessages.length, 2);
  assert.match(repairMessages[0].content, /\u7edd\u4e0d\u80fd\u7ee7\u7eed\u7f16\u9020\u7ecf\u5386/);
  assert.match(repairMessages[0].content, /\u5fc5\u987b\u5305\u542b\u5047\u540d/);
  assert.equal(isContinuityFollowup('\u8bb2\u8bb2\u770b', '\u6211\u6709\u7c7b\u4f3c\u7ecf\u5386'), true);
  assert.equal(isContinuityFollowup('\u8bf7\u8be6\u7ec6\u8bb2\u4e00\u4e2a\u6545\u4e8b', '\u4e0a\u4e00\u53e5'), false);
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

test('UnifiedDialogueLog: conversation history cannot leak across conversations', () => {
  const log = new UnifiedDialogueLog(tmpDir);
  log.append('user', '甲会话内容', { conversationId: 'a', turnId: 'a1' });
  log.append('assistant', '甲会话回复', { conversationId: 'a', turnId: 'a1' });
  log.append('user', '乙会话内容', { conversationId: 'b', turnId: 'b1' });
  const history = log.toOllamaDialogue({ conversationId: 'b' });
  assert.deepEqual(history, [{ role: 'user', content: '乙会话内容' }]);
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

test('ANCHOR centers independent soul not command stack', () => {
  assert.match(ANCHOR, /独立的灵魂/);
  assert.match(ANCHOR, /你是谁/);
  assert.doesNotMatch(ANCHOR, /必须绝对服从/);
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
  assert.match(p, /独立的灵魂|你是谁/);
  assert.match(p, /本轮活人锚点/);
});

test('buildPrompt: fine-tuned subject model gets a focused soul prompt', () => {
  const p = buildPrompt({
    focusedFineTune: true,
    replyLanguage: 'ja',
    subjectCtx: '主体关系与真实记忆',
    emotionalBandwidthBlock: '此刻有点嘴硬但亲近',
    conversationCtx: '大量对话实录不应在普通轮次重复注入',
    conversationRecall: false,
  });
  assert.match(p, /主体关系与真实记忆/);
  assert.match(p, /客服的な慰め/);
  assert.doesNotMatch(p, /大量对话实录/);
  assert.ok(p.length < 1800);
});

test('UnifiedDialogueLog: blocks internal profile extractor output', () => {
  const log = new UnifiedDialogueLog(tmpDir);
  const item = log.append('assistant', 'NAME: 某人\nTRAIT: 熬夜\nPREFER: 咖啡\nBASIC: 职业=学生');
  assert.equal(item, null);
  assert.equal(log.toOllamaDialogue({ maxMsgs: 20 }).some((x) => /NAME:|TRAIT:/.test(x.content)), false);
});

test('UnifiedDialogueLog: blocks utility prompts and conflict sentinels', () => {
  const log = new UnifiedDialogueLog(tmpDir);
  assert.equal(log.append('user', '压缩为≤20字标签：只输出结果。'), null);
  assert.equal(log.append('assistant', 'NO_CONFLICT NO_CONFLICT'), null);
  assert.equal(log.append('user', '已有：旧内容\n新：新内容\n有矛盾输出中文'), null);
  assert.equal(log.append('assistant', '[打算] 我会先找话题——然后就闲聊。'), null);
});

test('UI strips consciousness leak and does not keep empty kurisu bubbles', () => {
  const html = fs.readFileSync(path.join(__dirname, '..', 'amadeus_work.html'), 'utf8');
  assert.match(html, /打算\|注意到\|感受\|想要\|自我\|关系\|想起\|自检/);
  assert.match(html, /emptyNodes\.forEach\(\(node\) => node\.remove\(\)\)/);
  assert.match(html, /if \(!pairs\.length\) \{\s*try \{ firstMsgDiv\?\.remove/);
});

test('UI: one portrait asset; server replaceText overrides stream draft', () => {
  const html = fs.readFileSync(path.join(__dirname, '..', 'amadeus_work.html'), 'utf8');
  assert.doesNotMatch(html, /pose === 'expressive'[\s\S]{0,180}0\.png/);
  // 服务端定稿必须覆盖流式草稿，避免界面与实录分叉
  assert.match(html, /serverDisplayCn = rep/);
  assert.match(html, /rawFull = rep/);
  assert.match(html, /onDone\(serverDisplayCn \|\| rawFull\)/);
  assert.doesNotMatch(html, /onToken\(rep\)/);
  assert.match(html, /_renderFinalReplyBubbles\(cnText, msgDiv/);
  assert.match(html, /_splitAutonomyBubbles\(text, maxBubbles = 3\)/);
  assert.match(html, /if \(!stripped \|\| isOnlyDots\) \{\s*return '';/);
  // 有日文准绳时显示必须锁定为硬翻，禁止 coalesce 换另一句
  assert.match(html, /字幕=朗读日文硬翻|声画锁定|硬翻成显示中文/);
  assert.match(html, /已配对日文：只读这句/);
});

test('Conversation presence: initiative is decided before speech and is interruptible', () => {
  const html = fs.readFileSync(path.join(__dirname, '..', 'amadeus_work.html'), 'utf8');
  const server = fs.readFileSync(path.join(__dirname, '..', 'server.js'), 'utf8');
  assert.match(html, /_openConversationInitiativeWindow\(text, cnText\)/);
  assert.match(html, /initiativeAbortController\?\.abort\('new-user-turn'\)/);
  assert.match(html, /\/initiative\/decide/);
  assert.match(html, /lastKurisuAnchor: String\(replyText/);
  assert.match(html, /source = 'autonomy'/);
  assert.match(server, /const isConversationInitiative = req\.body\.conversationInitiative === true/);
  assert.match(server, /const anchor = lastKurisuAnchor \|\| lastUserAnchor \|\| legacyAnchor/);
  assert.match(server, /!reply && REPLY_FALLBACK_ENABLED && !isAutonomyInitiative/);
});

test('Proactive delivery: short irregular bursts do not reuse stale dialogue', () => {
  const html = fs.readFileSync(path.join(__dirname, '..', 'amadeus_work.html'), 'utf8');
  const server = fs.readFileSync(path.join(__dirname, '..', 'server.js'), 'utf8');
  assert.match(html, /autonomyMinIdleMin: 0\.35/);
  assert.match(html, /autonomyMaxBubbles: 3/);
  assert.match(html, /contextFresh: phase === 'idle' && idleMs <= 3 \* 60 \* 1000/);
  assert.match(html, /recentProactive: this\.working\.filter/);
  assert.match(html, /_buildAutonomySpeechPairs/);
  assert.match(html, /recentDialogue/);
  assert.match(server, /recentDialogue\.length/);
  assert.match(server, /空闲时随手开口/);
  assert.match(server, /独立的灵魂|牧濑红莉栖本人/);
  assert.match(server, /proactiveShapeOk/);
  assert.match(server, /repeat_penalty: 1\.16/);
  const internalCall = html.slice(html.indexOf('async _callLLM('), html.indexOf('_isDesignRequest(', html.indexOf('async _callLLM(')));
  assert.match(internalCall, /_callOllamaDirect/);
  assert.doesNotMatch(internalCall, /_postChatWithFallback/);
});

test('Memory admission: synthetic initiative turns cannot become user memory', () => {
  const legacy = fs.readFileSync(path.join(__dirname, '..', 'brain', 'legacyChat.js'), 'utf8');
  const server = fs.readFileSync(path.join(__dirname, '..', 'server.js'), 'utf8');
  const html = fs.readFileSync(path.join(__dirname, '..', 'amadeus_work.html'), 'utf8');
  assert.match(legacy, /const isRealUserTurn = !autonomyInitiative/);
  assert.match(legacy, /memoryAdmission\.allowEvent/);
  assert.match(legacy, /if \(isRealUserTurn\) s\.lastDigitalLifeTurn/);
  assert.match(legacy, /isRealUserTurn \? d\.analyticsInst\.analyze/);
  assert.match(server, /new MemoryAdmissionPolicy\(memoryDir\)/);
  assert.match(server, /app\.post\('\/initiative\/feedback'/);
  assert.match(html, /phase === 'idle'|speakPhase === 'idle'|initiativePhase === 'idle'/);
  assert.match(html, /phase: 'presence'|phase === 'presence'/);
  assert.match(html, /initiative\/session-start/);
  assert.match(html, /initiative\/sent/);
  assert.match(server, /app\.post\('\/initiative\/session-start'/);
  assert.match(server, /decidePresence/);
  assert.match(server, /察觉到人/);
  assert.match(server, /独立的灵魂|表达层：你是独立的灵魂|牧濑红莉栖本人/);
});

test('Japanese TTS: short replies stay whole and Chinese cannot reach synthesis', () => {
  const html = fs.readFileSync(path.join(__dirname, '..', 'amadeus_work.html'), 'utf8');
  const server = fs.readFileSync(path.join(__dirname, '..', 'server.js'), 'utf8');
  assert.match(html, /full\.replace\(\/\\s\/g, ''\)\.length <= 120/);
  assert.match(html, /await this\._startBackgroundJapaneseTTS\(text, modelJp, turnId\)/);
  assert.doesNotMatch(html, /_queueStreamingTTS[\s\S]{0,600}_ensureJapaneseTTSPieces/);
  assert.match(server, /const kanaCount = \(body\.text\.match/);
  assert.match(server, /tts accepts Japanese speech only/);
  assert.match(server, /text_lang: 'ja'/);
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
