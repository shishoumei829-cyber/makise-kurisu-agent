/**
 * AMADEUS BACKEND v7.0
 * 完整数字生命体系统
 *
 * ① 记忆系统        — 事件权重 + 长期PAD影响 + 关系演化
 * ② 动机系统        — 当前想要/在意/害怕（内生+反应式）
 * ③ 行为决策        — 多路径候选 + 打分 + 选择
 * ④ 自我模型        — 她对自己的认知，持续演化
 * ⑤ 内生目标        — 她自己产生的目标，不依赖输入触发
 * ⑥ 策略延续        — 跨轮次的持续策略
 * ⑦ 用户理解系统    — 理解你是什么人，不只记你说过什么
 * ⑧ 环境音监听      — 她在房间里，不只在聊天框里
 */

'use strict';

const express = require('express');
const cors    = require('cors');
const fs      = require('fs');
const path    = require('path');
const os      = require('os');
const { spawn } = require('child_process');
const { createDesignTask } = require('./design_engine');
const { MemorySystem } = require('./lib/memory');
const { UnifiedDialogueLog, needsConversationRecall } = require('./lib/unifiedDialogueLog');
const { MemoryAdmissionPolicy } = require('./lib/memoryAdmission');
const { MemoryPalaceStore } = require('./lib/memory/palaceStore');
const { SoulRuntime } = require('./lib/soulRuntime');
const {
  buildGroundedTranslationMessages,
  validateGroundedTranslation,
  detectUnsupportedAdditions,
  buildContextSafeFallback,
  detectReplyCoherenceIssues,
} = require('./lib/groundedTranslation');
const { BehaviorIngest } = require('./lib/behaviorIngest');
const { normalizeClientContext, buildClientContextBlock } = require('./lib/clientContext');
const { InnerStateSix } = require('./cognitive/innerStateSix');
const { ConversationInitiativeEngine } = require('./cognitive/conversationInitiative');
const { readSocialField, nextSenseMs } = require('./cognitive/socialRead');
const { EmotionalBandwidthEngine } = require('./cognitive/emotionalBandwidth');
const { ButlerKernel } = require('./lib/butler/kernel');
const {
  buildProactiveContentPlan,
  buildJapaneseContentPlanBlock,
  buildJapanesePlanRepair,
  validateProactiveContent,
} = require('./lib/proactiveContentPlan');
const { buildSocialIdentityPrompt } = require('./cognitive/socialIdentity');
const { buildExpressionVariantBlock } = require('./cognitive/expressionVariants');
const {
  validateJapaneseLine,
  buildSelfCheckPrompt,
  parseSelfCheckJson,
  runJapaneseValidationPipeline,
  runJapaneseFirstPipeline,
} = require('./cognitive/japanesePipeline');
const {
  getReplyLanguageMode,
  isPrimarilyJapanese,
  extractJapaneseBody,
  stripModelDecorations,
  stripConsciousnessEcho,
  countScriptChars,
  validateJapaneseOutput,
  buildLiteralJpToCnMessages,
  alignLiteralCnToJapanese,
} = require('./lib/replyLanguage');
const { isRecentAssistantDuplicate } = require('./lib/replyContinuity');

/** 加载项目根目录 .env（不覆盖已有系统/进程环境变量） */
function loadDotEnv() {
  const envPath = path.join(__dirname, '.env');
  if (!fs.existsSync(envPath)) return;
  for (const line of fs.readFileSync(envPath, 'utf8').split(/\r?\n/)) {
    const s = line.trim();
    if (!s || s.startsWith('#')) continue;
    const eq = s.indexOf('=');
    if (eq < 1) continue;
    const key = s.slice(0, eq).trim();
    let val = s.slice(eq + 1).trim();
    if (
      (val.startsWith('"') && val.endsWith('"')) ||
      (val.startsWith("'") && val.endsWith("'"))
    ) val = val.slice(1, -1);
    if (
      /^AMADEUS_(CHAT_MODEL|LITE_MODEL|REPLY_LANGUAGE|PROACTIVE)$/.test(key)
      || process.env[key] == null
      || process.env[key] === ''
    ) process.env[key] = val;
  }
}
loadDotEnv();

// ── 用户理解系统 ──────────────────────────────────────────────────
const {
  init:             initUserModel,
  UserModel,
  ConversationAnalytics,
  HabitExtractor,
} = require('./user_model');

// ── 学习引擎（强化学习 + 人格演化）──────────────────────────────────
const {
  init:             initLearningEngine,
  ReinforcementLearning,
  PersonalityEvolution,
} = require('./learning_engine');

// ── 元认知模块（自我反思 + 价值观一致性 / LLM 张力检测）────────────────
const {
  init:             initMetacognition,
  SelfReflection,
  ValueConsistency,
} = require('./metacognition');

// ── BDI 引擎（信念/欲望/意图推断，异步周期性触发）────────────────────
const { inferUserBdi } = require('./bdi_engine');

// ── 数字生命编排（内驱力/梦境/信念/具身感知等）────────────────────────
const { DigitalLifeOrchestrator } = require('./digital_life');

// ── cognitive/ 子系统模块 ─────────────────────────────────────────
const { MotivationSystem }   = require('./cognitive/motivation');
const { InternalGoalSystem } = require('./cognitive/goals');
const { StrategyLayer }      = require('./cognitive/strategy');
const { SelfModel }          = require('./cognitive/selfModel');
const { BehaviorDecision }   = require('./cognitive/behavior');
const {
  PAD_BASE, PAD_DECAY_LAMBDA, BOND_DELTA_S,
  clamp, loadPAD, savePAD, updatePAD, inferMainEventFromInput,
} = require('./cognitive/pad');
const {
  _clipInnerPrompt, _compactSoulForPrompt, _fitPromptToBudget,
  padTelemetry, getTimeContext, symbolicReasoning, buildPrompt,
} = require('./cognitive/prompts');
const {
  utteranceFocusLine,
  filterRagHits,
  filterAutonomyRagHits,
  filterAutonomyMemCtx,
  buildEngagementHint,
  stripOrphanClosingSentence,
  stripChatMarkdown,
  stripRoleplayActions,
  shouldReplaceStreamText,
} = require('./cognitive/replyAlign');
const {
  parseIncomingChat,
  capDialogue,
  buildOllamaMessages,
  fitSystemForDialogue,
  estimateMessageChars,
  resolvePromptCharBudget,
  calculatePromptCharBudget,
} = require('./cognitive/chatTurns');
const { derivePresence, presenceToPromptLine } = require('./cognitive/presence');
const { repairKurisuReply, reconcileFinalReply, isOocRepairEnabled } = require('./lib/oocGuard');
const { buildTurnStyleBlock } = require('./cognitive/turnStyle');
const {
  buildCompanionBlock,
  buildAutonomySituation,
  isHighIntimacyMode,
  effectiveRelScore,
  applyHighIntimacyBootstrap,
  idleSilencePadDelta,
  HIGH_INTIMACY_REL_FLOOR,
} = require('./cognitive/companionMode');
const {
  ensureWhoamiOnDisk,
  bootstrapWhoamiRecord,
  buildPartnerContextBlock,
  partnerIsOkabe,
  resolvePartnerDisplayName,
  isOkabePartnerMode,
} = require('./lib/partnerIdentity');
const {
  buildProactiveReplyFocus,
  buildAutonomyContinuityBlock,
  extractLastRealUserLine,
  detectReplyingToHerThread,
  replyLooksLikeAutonomyFabrication,
} = require('./cognitive/turnContinuity');
const userPresence = require('./lib/userPresence');
const { runStartupChecks } = require('./lib/startupCheck');
const { Brain } = require('./brain');
const { BrainSelfModel } = require('./brain/selfModel');
const { WorldModel } = require('./brain/worldModel');
const { BrainLearner } = require('./brain/learner');
const { SubjectCore } = require('./brain/subjectCore');

// 固定台词兜底默认关闭；只有显式设为 1 才允许改写模型正文。
const REPLY_FALLBACK_ENABLED = process.env.AMADEUS_REPLY_FALLBACK === '1';
// 实时语音优先：流式结束后不再串行调用第二轮 LLM 做日语/中文改写。
const FAST_STREAMING_VOICE = process.env.AMADEUS_FAST_STREAMING_VOICE !== '0';

function applyOocRepair(content, userContent, streamedRaw = '', oocOpts = {}) {
  if (!REPLY_FALLBACK_ENABLED || !isOocRepairEnabled(oocOpts)) return String(content || '').trim();
  const streamed = String(streamedRaw || '').trim();
  const repaired = repairKurisuReply(userContent, content, oocOpts);
  const out = streamed ? reconcileFinalReply(streamed, repaired, userContent, oocOpts) : repaired;
  if (out !== String(content || '').trim()) {
    console.log('[chat] OOC/口吻兜底已调整回复');
  }
  return out;
}

const LOW_INFORMATION_REPLIES = /^(?:讲|说重点|怎么了|有事|听着呢|嗯|哦|然后呢)[。！？!?]?$/;

function replyNeedsCompanionRefinement(userText, reply) {
  const input = String(userText || '').trim();
  const output = String(reply || '').trim().replace(/\s+/g, '');
  if (input.length < 8 || !output || !LOW_INFORMATION_REPLIES.test(output)) return false;
  return !/^(?:讲|说重点|怎么了|有事)[。！？!?]?$/.test(input.replace(/\s+/g, ''));
}

function companionFallbackForLowInfo(_userText) {
  return '';
}

async function refineLowInformationReply(model, userText, reply, context = {}, options = {}) {
  if (!REPLY_FALLBACK_ENABLED) return String(reply || '').trim();
  if (!replyNeedsCompanionRefinement(userText, reply)) return String(reply || '').trim();
  const situation = String(context.situation || '').slice(0, 260);
  const feeling = String(context.feeling || '').slice(0, 180);
  const retryPrompt = [
    '你刚才的回复太空泛，重写成牧濑红莉栖真正会发出的中文消息。',
    '必须回应用户输入里的具体内容；至少给出一个具体反应，并在合适时追问一个具体点。',
    '禁止只输出“讲”“怎么了”“说重点”“听着呢”等无信息短句，禁止解释规则，禁止客服腔，最多三句。',
    `用户：${String(userText || '').slice(0, 260)}`,
    `原回复：${String(reply || '').slice(0, 100)}`,
    situation ? `当下情境：${situation}` : '',
    feeling ? `当前内在状态：${feeling}` : '',
  ].filter(Boolean).join('\n');
  try {
    const repaired = await _ollamaChatOnce(model, [
      { role: 'system', content: '你是牧濑红莉栖本人。短、聪明、有温度，像给熟人发消息。只输出中文台词。' },
      { role: 'user', content: retryPrompt },
    ], {
      temperature: Math.max(0.65, Number(options.temperature) || 0.72),
      num_predict: Math.min(180, Number(options.maxTokens) || 180),
      num_ctx: Number(options.numCtx) || 2048,
    });
    const clean = stripChatMarkdown(stripModelThinkingAll(repaired));
    if (clean && !LOW_INFORMATION_REPLIES.test(clean.replace(/\s+/g, ''))) return clean;
  } catch (e) {
    console.warn('[chat] companion refinement skipped:', e.message);
  }
  return companionFallbackForLowInfo(userText);
}

const app = express();
app.use(cors());
app.use(express.json({ limit: '50mb' }));
const OLLAMA_BASE = (process.env.AMADEUS_OLLAMA_BASE || 'http://127.0.0.1:11434').replace(/\/$/, '');
const PORT = (() => {
  const n = Number(process.env.AMADEUS_BACKEND_PORT || process.env.PORT);
  return Number.isFinite(n) && n > 0 ? Math.floor(n) : 3000;
})();

// ──────────────────────────────────────────────────────────────
//  路径 — 运行时数据放在项目外，防止文件监听器触发页面刷新
//  ⚠️  此路径永不再改！改路径会导致历史数据丢失
// ──────────────────────────────────────────────────────────────
const rootPath    = __dirname;
const dataDir     = process.env.AMADEUS_DATA_DIR
  ? path.resolve(process.env.AMADEUS_DATA_DIR)
  : path.join(os.homedir(), 'amadeus_data');
const soulRuntime = new SoulRuntime({
  statePath: path.join(dataDir, 'subject_state.json'),
  cognitionModel: process.env.AMADEUS_COGNITION_MODEL || 'qwen2.5:3b',
});
// The subject core is the sole speaker-intent arbiter. SoulRuntime remains the
// autobiographical/affective store it reads from, rather than a second speaker.
const subjectCore = new SubjectCore({
  statePath: path.join(dataDir, 'subject_core.json'),
});
const conversationInitiative = new ConversationInitiativeEngine({
  statePath: path.join(dataDir, 'conversation_initiative.json'),
});
{
  const validThoughtIds = new Set(
    soulRuntime.snapshot().thoughts
      .filter((item) => item.status === 'open')
      .map((item) => item.id),
  );
  if (
    conversationInitiative.state.activeThoughtId
    && !validThoughtIds.has(conversationInitiative.state.activeThoughtId)
  ) {
    conversationInitiative.state.activeThoughtId = '';
    conversationInitiative._save();
  }
}
const emotionalBandwidth = new EmotionalBandwidthEngine({
  statePath: path.join(dataDir, 'emotional_bandwidth.json'),
});
const configuredButlerRoots = String(process.env.AMADEUS_BUTLER_ALLOWED_ROOTS || '')
  .split(';')
  .map((item) => item.trim())
  .filter(Boolean);
const butlerKernel = new ButlerKernel({
  dataDir,
  rootPath,
  allowedRoots: configuredButlerRoots.length
    ? configuredButlerRoots
    : [path.join(os.homedir(), 'Downloads'), os.homedir(), rootPath],
  reasoner: _requestWorkBrainPlan,
});
if (process.env.AMADEUS_BUTLER_OPERATOR !== '0') {
  const butlerOperatorTimer = setInterval(() => {
    butlerKernel.operatorTick().catch((error) => {
      console.warn('[butler/operator]', error.message);
    });
  }, 3000);
  butlerOperatorTimer.unref?.();
}

// ★ 托管静态文件
app.use(express.static(rootPath));

// ★ 根路由重定向到主页面，解决 404 问题
app.get('/', (req, res) => {
  // 静态中间件在 Windows 下可能把目录根路径当成文件处理；明确跳转到
  // 页面资源，保证浏览器和 Electron 的默认入口一致。
  res.redirect('/amadeus_work.html');
});

// ── 智能管家内核：目标、任务、证据与验证的唯一事实源 ───────────────
function butlerRoute(handler) {
  return async (req, res) => {
    try {
      const result = await handler(req, res);
      if (!res.headersSent) res.json({ ok: true, ...result });
    } catch (error) {
      res.status(400).json({ ok: false, error: error.message });
    }
  };
}

app.get('/butler/status', butlerRoute(() => butlerKernel.status()));
app.get('/butler/events', butlerRoute((req) => ({
  events: butlerKernel.journal.recent(Math.min(500, Math.max(1, Number(req.query.limit) || 100))),
})));
app.get('/butler/updates', butlerRoute((req) => ({
  updates: butlerKernel.updatesSince(req.query.since, req.query.limit, {
    conversationId: req.query.conversationId,
  }),
})));
app.get('/butler/goals', butlerRoute((req) => ({
  goals: butlerKernel.tasks.listGoals({ status: req.query.status }),
})));
app.post('/butler/goals', butlerRoute((req) => ({
  goal: butlerKernel.tasks.createGoal(req.body || {}),
})));
app.get('/butler/tasks', butlerRoute((req) => ({
  tasks: butlerKernel.tasks.listTasks({ status: req.query.status, goalId: req.query.goalId }),
})));
app.post('/butler/tasks', butlerRoute((req) => butlerKernel.tasks.createTask(req.body || {})));
app.post('/butler/ingest', butlerRoute((req) => {
  // 显式 API 登记：调用方已经决定要建任务，不扫聊天词表。
  const text = String(req.body?.message || req.body?.text || req.body?.prompt || '').trim();
  if (!text && !req.body?.title) {
    return butlerKernel.observeUserRequest({
      body: req.body || {},
      source: req.body?.source || 'api',
      requestKey: req.body?.requestKey,
      turnId: req.body?.turnId,
    });
  }
  return butlerKernel.proposeTask({
    text,
    title: req.body?.title,
    category: req.body?.category || req.body?.type,
    risk: req.body?.risk,
    source: req.body?.source || 'api',
    requestKey: req.body?.requestKey,
    turnId: req.body?.turnId,
  });
}));
app.post('/butler/tasks/:id/transition', butlerRoute((req) => ({
  task: butlerKernel.tasks.transition(req.params.id, req.body?.status, req.body || {}),
})));
app.post('/butler/tasks/:id/confirm', butlerRoute((req) => ({
  task: butlerKernel.tasks.confirm(req.params.id, req.body?.approved === true, req.body || {}),
})));
app.post('/butler/tasks/:id/evidence', butlerRoute((req) => ({
  evidence: butlerKernel.tasks.addEvidence(req.params.id, req.body || {}),
})));
app.post('/butler/tasks/:id/verify', butlerRoute((req) => ({
  task: butlerKernel.tasks.verifyTask(req.params.id, req.body || {}),
})));
app.post('/butler/tasks/:id/execute', butlerRoute(async (req) => (
  butlerKernel.executeTask(req.params.id, req.body?.capabilityId, req.body?.args || {}, {
    source: 'api',
  })
)));
app.post('/butler/tasks/:id/plan', butlerRoute(async (req) => (
  butlerKernel.planTask(req.params.id, { allowStrong: req.body?.allowStrong !== false })
)));
app.post('/butler/tasks/:id/run', butlerRoute(async (req) => (
  butlerKernel.runPlan(req.params.id, { source: 'api' })
)));
app.get('/butler/reminders', butlerRoute((req) => ({
  reminders: req.query.due === '1'
    ? butlerKernel.reminders.due()
    : butlerKernel.reminders.list({ status: req.query.status }),
})));
app.post('/butler/reminders/:id/delivered', butlerRoute((req) => {
  const reminder = butlerKernel.reminders.markDelivered(req.params.id);
  const linked = butlerKernel.agency?.intentions?.list({ openOnly: true })
    ?.find((item) => item.reminderId === req.params.id);
  if (linked) {
    try { butlerKernel.agency.intentions.markFulfilled(linked.id, { kind: 'reminder_delivered' }); } catch { /* ignore */ }
  }
  return { reminder };
}));
app.post('/butler/reminders/:id/acknowledge', butlerRoute((req) => ({
  reminder: butlerKernel.reminders.acknowledge(req.params.id),
})));
app.get('/agency/intentions', butlerRoute((req) => ({
  intentions: req.query.due === '1'
    ? butlerKernel.agency.dueIntentions()
    : butlerKernel.agency.intentions.list({
      status: req.query.status,
      openOnly: req.query.open === '1',
    }),
  capabilities: butlerKernel.agency.capabilitySnapshot(),
})));
app.post('/agency/intentions/:id/fulfilled', butlerRoute((req) => ({
  intention: butlerKernel.agency.intentions.markFulfilled(req.params.id, req.body?.evidence || null),
})));
app.get('/butler/trash', butlerRoute((req) => ({
  entries: butlerKernel.fileUndo.list(String(req.query.status || '')),
})));

const soulPath    = path.join(rootPath, "kurisu_soul.txt");
const soulJaPath  = path.join(rootPath, "kurisu_soul_ja.txt");
const corePromptPath = path.join(rootPath, "kurisu_core_prompt.txt");
const voicePath = path.join(rootPath, "kurisu_voice.txt");
const characterRulesPath = path.join(rootPath, "kurisu_character_rules.txt");
const memoryDir   = dataDir;
const memoryPath  = path.join(memoryDir, "user_profile.json");
const whoamiPath  = path.join(memoryDir, "whoami.json");       // ★ 用户身份档案
const padPath     = path.join(memoryDir, "pad_state.json");
const motivePath  = path.join(memoryDir, "motivation.json");
const eventLogPath= path.join(memoryDir, "event_log.json");
const selfModelPath  = path.join(memoryDir, "self_model.json");
const strategyPath   = path.join(memoryDir, "strategy.json");
const vectorDir   = path.join(rootPath, "vector_store");
const vectorFallbackPath = path.join(vectorDir, "store.json");
const hnswIndexPath = path.join(vectorDir, "hnswlib.index");

// ★ 缓存soul和character rules，避免每次请求都读取文件
let cachedSoulContent = '';
let cachedSoulJaContent = '';
let cachedVoiceContent = '';
let cachedCharacterRules = '';
function loadSoulCache() {
  const parts = [];
  try { parts.push(fs.readFileSync(corePromptPath, 'utf8')); } catch {}
  try { parts.push(fs.readFileSync(soulPath, 'utf8')); } catch {}
  cachedSoulContent = parts.filter(Boolean).join('\n\n---\n\n');
  try { cachedSoulJaContent = fs.readFileSync(soulJaPath, 'utf8'); } catch {}
  try { cachedVoiceContent = fs.readFileSync(voicePath, 'utf8'); } catch {}
  try { cachedCharacterRules = fs.readFileSync(characterRulesPath, 'utf8'); } catch {}
}
loadSoulCache();

// 轻对话常被微调模型强行补一个收尾问句。只有用户本轮本身在提问时才保留，
// 否则删掉最后那个“客服式递球”，让她可以停在自己的判断或情绪上。
function stripDefaultQuestionEnding(reply, userText) {
  const text = String(reply || '').trim();
  const user = String(userText || '').trim();
  if (!text || /[？?]/.test(user) || /(?:吗|么|如何|为何|为什么|怎么|是否|誰|何|なに|どう|か？|か\?)/u.test(user)) return text;
  const parts = text.match(/[^。！？!?…]+[。！？!?…]?/g)?.map((s) => s.trim()).filter(Boolean) || [text];
  if (parts.length < 2) return text;
  const last = parts[parts.length - 1];
  if (!/[？?]$/.test(last) || last.length < 4) return text;
  const kept = parts.slice(0, -1).join('');
  return kept.trim() || text;
}

// 主动续话里最常见的“假接话”：把用户已经说完的事实再问一遍。
// 这不是禁止提问，而是只去掉确认式复述，保留后面的真实反应。
function stripProactiveConfirmation(reply, anchor) {
  const text = String(reply || '').trim();
  const source = String(anchor || '').trim();
  if (!text || !source || /[？?]$/.test(source)) return text;
  const parts = text.match(/[^。！？!?…]+[。！？!?…]?/g)?.map((s) => s.trim()).filter(Boolean) || [text];
  const confirmation = /(?:終わらせた|終わった|完成した|書き終えた|できた|やった)の[？?]|(?:写完了吗|结束了吗|做好了吗|完成了吗)[？?]/;
  const kept = parts.filter((part) => !confirmation.test(part));
  return kept.join('').replace(/^(?:それなら|那就)[、，]\s*/u, '').trim();
}

if (!fs.existsSync(memoryDir)) fs.mkdirSync(memoryDir, { recursive: true });
if (!fs.existsSync(memoryPath)) {
  fs.writeFileSync(memoryPath, JSON.stringify({
    user_profile: { confirmed_habits: [], tentative_observations: [] }
  }, null, 2));
}
if (!fs.existsSync(whoamiPath)) {
  fs.writeFileSync(whoamiPath, JSON.stringify(bootstrapWhoamiRecord({
    name: '未知',
    traits: [],
    preferences: [],
    basics: {},
    relationship_note: '',
    last_updated: Date.now(),
  }), null, 2));
}
ensureWhoamiOnDisk(whoamiPath);
if (isOkabePartnerMode()) {
  console.log('[partner] 对话对象默认：冈部伦太郎（AMADEUS_PARTNER_ID=custom 可改）');
}

// ──────────────────────────────────────────────────────────────
//  RAG（不动）
// ──────────────────────────────────────────────────────────────
let ragStore = null, hnswVectorStore = null, hnswBroken = false;
let HNSWLib = null, OllamaEmbeddings = null;
try {
  ({ HNSWLib } = require("@langchain/community/vectorstores/hnswlib"));
  ({ OllamaEmbeddings } = require("@langchain/ollama"));
  // @langchain/community 本身可能存在，但它的可选原生扩展没有安装。
  // 先探测，避免第一次聊天才打印一整段 node module 错误。
  try { require.resolve('hnswlib-node'); }
  catch (_) { HNSWLib = null; hnswBroken = true; }
} catch (e) {
  // hnswlib-node 是 Windows 原生扩展；没有 VS C++ 工具链时不能装上。
  // 标记为 broken，让本进程直接使用已生成的 store.json，避免每轮重复尝试。
  hnswBroken = true;
  console.warn("[rag] 原生 HNSW 不可用，使用 store.json 余弦检索:", e.message);
}

function dot(a, b) { let s=0; for(let i=0;i<a.length;i++) s+=a[i]*b[i]; return s; }
function norm(v)    { let s=0; for(let i=0;i<v.length;i++) s+=v[i]*v[i]; return Math.sqrt(s)||1; }
function cosineSim(a,b) {
  if(!Array.isArray(a)||!Array.isArray(b)||!a.length||a.length!==b.length) return -1;
  return dot(a,b)/(norm(a)*norm(b));
}
function loadRagStore() {
  if(!fs.existsSync(vectorFallbackPath)) return [];
  try { const r=JSON.parse(fs.readFileSync(vectorFallbackPath,"utf8")); return Array.isArray(r)?r.filter(x=>x&&Array.isArray(x.embedding)&&x.text):[]; }
  catch { return []; }
}
async function embedQuery(text) {
  const res = await fetch(`${OLLAMA_BASE}/api/embeddings`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({model:"nomic-embed-text",prompt:text})});
  if(!res.ok) throw new Error(`Embedding ${res.status}`);
  return (await res.json()).embedding||[];
}
/** 防抖磁盘写入（单例 MemorySystem / SelfModel 使用） */
function debounceFileWrite(ms, fn) {
  let t = null;
  return () => {
    if (t) clearTimeout(t);
    t = setTimeout(() => {
      t = null;
      fn();
    }, ms);
  };
}

/** 与前端一致的懒记忆门控：闲聊不拉 RAG / 时间线 / 高权重碎片（对话实录单独始终注入） */
function needsLongTermMemory(userText, recentUserLines = []) {
  const t = String(userText || '').trim();
  if (!t || t.replace(/\s/g, '').length < 6) return false;
  if (needsConversationRecall(t)) return true;
  if (/昨天|之前|上次|刚才|你说过|你记得|那个|那次|那时|以前|前几天/.test(t)) return true;
  if (/论文|实验|量子|神经|时间机器|理论|物理|数学|世界线|SERN|凶真|铃羽|椎名|真由[里世]|Dr\.?Pepper|胡椒博士/.test(t)) return true;
  const lines = (recentUserLines || []).map((x) => String(x || '').trim()).filter(Boolean);
  const tokens = (t.match(/[\u4e00-\u9fa5A-Za-z0-9]{2,}/g) || []);
  return tokens.some((tok) => tok.length >= 3 && lines.some((r) => r.includes(tok)));
}

// _clipInnerPrompt, _compactSoulForPrompt, _fitPromptToBudget → cognitive/prompts.js

async function retrieveTopContexts(query, topK=3) {
  if(!query.trim()) return [];
  if(!hnswBroken&&HNSWLib&&OllamaEmbeddings&&fs.existsSync(hnswIndexPath)) {
    try {
      if(!hnswVectorStore) {
        const emb=new OllamaEmbeddings({model:"nomic-embed-text",baseUrl:OLLAMA_BASE});
        hnswVectorStore=await HNSWLib.load(vectorDir,emb);
      }
      const docs=await hnswVectorStore.similaritySearchWithScore(query.trim(),topK);
      return docs.map(([doc,score])=>({text:doc.pageContent,score,source:doc.metadata?.source||"unknown"}));
    } catch(e) {
      // 一次失败后本进程内不再尝试 HNSW，直接走 store.json 降级，避免每轮刷错误栈
      hnswBroken = true;
      hnswVectorStore = null;
      console.warn("[rag] HNSW 不可用，后续直接走 store.json 降级:", e.message);
    }
  }
  if(!ragStore) ragStore=loadRagStore();
  if(!ragStore.length) return [];
  const qv=await embedQuery(query.trim());
  if(!qv.length) return [];
  return ragStore.map(item=>({text:item.text,score:cosineSim(qv,item.embedding),source:item.metadata?.source||"unknown"}))
    .filter(x=>Number.isFinite(x.score)).sort((a,b)=>b.score-a.score).slice(0,topK);
}

// ══════════════════════════════════════════════════════════════════
//

// MotivationSystem → 已迁移至 cognitive/motivation.js

// SelfModel → 已迁移至 cognitive/selfModel.js

// InternalGoalSystem → 已迁移至 cognitive/goals.js

// StrategyLayer → 已迁移至 cognitive/strategy.js

// ══════════════════════════════════════════════════════════════════
//  ③ 行为决策系统（BehaviorDecision）— 真正的多路径选择版
//
//  架构：
//  1. 候选行为池（5种基础行为类型）
//  2. 打分函数（PAD + 动机 + 记忆 + 输入触发 + 随机扰动）
//  3. 选择最高分行为
//  4. 将行为转化为执行约束（不是描述，是 LLM 必须遵守的操作规则）
//
//  行为类型定义：
//  APPROACH  — 主动靠近，愿意暴露一点真实情感
//  DEFEND    — 防御，嘴硬，不配合，把情感压回去
//  DEFLECT   — 转移，换话题，用反问或刻薄绕开
//  ENGAGE    — 智识投入，进入学术/分析模式，忘了防线
//  WITHDRAW  — 收缩，简短，不想多说，等对方先动
//
// BehaviorDecision → 已迁移至 cognitive/behavior.js

// PAD state functions → cognitive/pad.js
// buildPrompt helpers → cognitive/prompts.js


// ══════════════════════════════════════════════════════════════════
//  数值型动机状态（BehaviorDecision + 学习偏置 驱动）
// ══════════════════════════════════════════════════════════════════
const MOTIV_BASAL = { desire_closeness: 0.25, fear_rejection: 0.55, curiosity: 0.5 };
const MOTIV_HALFLIFE_HOURS = 48;

/** 朝基线值按半衰期衰减：数值型动机随时间回落，避免只增不减饱和到 1 */
function decayMotivationState(st, now = Date.now()) {
  const last = Number(st.updatedAt) || 0;
  const hours = Math.max(0, (now - last) / 3600000);
  const factor = Math.pow(0.5, hours / MOTIV_HALFLIFE_HOURS);
  for (const k of Object.keys(MOTIV_BASAL)) {
    const v = Number(st[k]);
    st[k] = Number.isFinite(v) ? MOTIV_BASAL[k] + (v - MOTIV_BASAL[k]) * factor : MOTIV_BASAL[k];
  }
  st.updatedAt = now;
  return st;
}

function loadMotivationState() {
  try {
    if (fs.existsSync(motivePath)) {
      const st = JSON.parse(fs.readFileSync(motivePath, 'utf8'));
      if (st && typeof st === 'object') return decayMotivationState(st);
    }
  } catch {}
  return { ...MOTIV_BASAL, updatedAt: Date.now() };
}
function saveMotivationState(ms) {
  fs.writeFile(motivePath, JSON.stringify(ms, null, 2), (err) => {
    if (err) console.error('[motivation] Save error:', err.message);
  });
}
let motivationState = loadMotivationState();

/** 全局对话轮计数（替代散落各处的 total_messages % N） */
let chatTurnCounter = 0;

/** 上一轮 /chat 选中的行为 ID，供回复后 RL 偏置更新 */
let lastChatBehaviorId = '';
/** 上一轮行为决策时的 RL 状态键 */
let lastRlStateKey = '';
/** 本轮数字生命侧输出（共情/内驱等） */
let lastDigitalLifeTurn = null;

/** Brain v1 可变状态桥（legacyChat 通过 getter/setter 与 server 同步） */
const brainState = {
  get currentPAD() { return currentPAD; },
  set currentPAD(v) { currentPAD = v; },
  get chatTurnCounter() { return chatTurnCounter; },
  set chatTurnCounter(v) { chatTurnCounter = v; },
  get lastChatBehaviorId() { return lastChatBehaviorId; },
  set lastChatBehaviorId(v) { lastChatBehaviorId = v; },
  get lastRlStateKey() { return lastRlStateKey; },
  set lastRlStateKey(v) { lastRlStateKey = v; },
  get lastDigitalLifeTurn() { return lastDigitalLifeTurn; },
  set lastDigitalLifeTurn(v) { lastDigitalLifeTurn = v; },
};

/**
 * Prompt 构建函数：按优先级拼装 system 侧上下文
 */
// ══════════════════════════════════════════════════════════════════
//  PAD → 自然语言转换（让 LLM 更容易理解情感状态）
// padTelemetry, getTimeContext, symbolicReasoning, buildPrompt → cognitive/prompts.js

/**
 * 记忆 → 动机更新
 * 根据最近事件更新 desire_closeness / fear_rejection / curiosity
 */
function updateMotivationFromMemory() {
  const recent = memorySystem.events.slice(-10);
  const delta = { desire_closeness: 0, fear_rejection: 0, curiosity: 0 };

  // 先按时间衰减到基线，再做增量（原来只加不减，长期必然饱和到 1）
  decayMotivationState(motivationState);

  for (const m of recent) {
    if (m.type === 'positive' || m.type === 'intimate') {
      delta.desire_closeness += 0.1 * m.weight;
    }
    if (m.type === 'negative' || m.type === 'conflict') {
      delta.fear_rejection += 0.15 * m.weight;
    }
    if (m.type === 'neutral' || m.type === 'scientific') {
      delta.curiosity += 0.05 * m.weight;
    }
  }

  for (const key of Object.keys(delta)) {
    motivationState[key] = Math.max(0, Math.min(1,
      (motivationState[key] || 0.3) + delta[key]
    ));
  }
  motivationState.updatedAt = Date.now();
  saveMotivationState(motivationState);
}

// ══════════════════════════════════════════════════════════════════
//  全局实例
// ══════════════════════════════════════════════════════════════════
const memorySystem  = new MemorySystem(memoryDir, eventLogPath);
const memoryAdmission = new MemoryAdmissionPolicy(memoryDir);
const memoryPalace = new MemoryPalaceStore(memoryDir);
const unifiedDialogueLog = new UnifiedDialogueLog(memoryDir);

function isProactiveEnabled() {
  const flag = String(process.env.AMADEUS_PROACTIVE ?? '1').trim().toLowerCase();
  return !['0', 'false', 'off', 'no'].includes(flag);
}

function proactiveDisabledHold() {
  return {
    shouldSpeak: false,
    action: 'hold',
    reason: 'proactive_disabled',
    suppressProactive: true,
  };
}

function dialogueTurnContext(body = {}) {
  const conversationId = String(body.conversationId || '').trim();
  const pendingUserTurn = unifiedDialogueLog.hasUnansweredUserTurn(conversationId);
  let idleMsSinceUser = Math.max(0, Number(body.idleMsSinceUser) || 0);
  if (!idleMsSinceUser) {
    const idleMs = Math.max(0, Number(body.idleMs) || 0);
    if (idleMs > 0) idleMsSinceUser = idleMs;
  }
  if (!idleMsSinceUser) {
    for (let i = unifiedDialogueLog.entries.length - 1; i >= 0; i -= 1) {
      const e = unifiedDialogueLog.entries[i];
      if (e.role === 'user' && String(e.source || 'chat') === 'chat') {
        idleMsSinceUser = Math.max(0, Date.now() - (e.ts || 0));
        break;
      }
    }
  }
  return {
    conversationId,
    pendingUserTurn,
    idleMsSinceUser,
    isThinking: body.isThinking === true,
  };
}

function holdProactiveDecision(turnCtx) {
  if (!turnCtx.pendingUserTurn && !turnCtx.isThinking) return null;
  return {
    ok: true,
    shouldSpeak: false,
    action: 'hold',
    reason: turnCtx.pendingUserTurn ? 'pending_user_turn' : 'is_thinking',
    nextCheckMs: 6000,
  };
}
const purgedDialogueLeaks = unifiedDialogueLog.purgeInternalLeaks();
if (purgedDialogueLeaks > 0) {
  console.warn(`[dialogue-log] 已清理 ${purgedDialogueLeaks} 条内部控制/摘要污染记录`);
}
/** @deprecated 别名，统一实录 */
const conversationMemory = unifiedDialogueLog;
// ── 统一事件事实源：对话定稿同步写入全局 journal（对话/任务/感知同源）──
unifiedDialogueLog.setEventSink((entry) => {
  butlerKernel.journal.append('dialogue.turn', {
    role: entry.role,
    text: entry.text,
    proactive: entry.proactive,
    channel: entry.source,
  }, {
    actor: entry.role === 'user' ? 'user' : 'amadeus',
    source: 'dialogue',
    correlationId: entry.turnId || '',
    ts: entry.ts,
  });
});
const behaviorIngest = new BehaviorIngest(memoryDir);
const innerStateSix = new InnerStateSix(memoryDir);
const motivSystem   = new MotivationSystem();
const behaviorSys   = new BehaviorDecision();
const selfModel     = new SelfModel(selfModelPath, debounceFileWrite);
const brainSelfModel = new BrainSelfModel(path.join(memoryDir, 'self_model_v2.json'), debounceFileWrite);
brainSelfModel.migrateFromLegacy(selfModelPath, selfModel.get());
const worldModel = new WorldModel();
const brainLearner = new BrainLearner(brainSelfModel);
const goalSystem    = new InternalGoalSystem();
const strategyLayer = new StrategyLayer(strategyPath);

// ── 数字生命编排器（模块一·自主性 由此统一提供好奇心等）────────────
const digitalLife = new DigitalLifeOrchestrator();
digitalLife.init(memoryDir);

// ── 用户理解系统实例 ───────────────────────────────────────────
initUserModel(memoryDir);
const userModelInst   = new UserModel();
userModelInst.ensureCoupleRelationship();
const analyticsInst   = new ConversationAnalytics(userModelInst);
const habitExtractor  = new HabitExtractor(userModelInst);
butlerKernel.setUserContextProvider(() => ({ userModel: userModelInst.model }));

function observeMemoryEvidence(source, text) {
  const newlyQuarantined = memoryAdmission.observe(source, text) || [];
  if (newlyQuarantined.length) {
    digitalLife.purgeContaminatedTopics(newlyQuarantined);
    userModelInst.purgeContaminatedTopics(newlyQuarantined);
    console.warn(`[memory-admission] 已隔离自我强化主题: ${newlyQuarantined.slice(0, 6).join('、')}`);
  }
  return newlyQuarantined;
}

// ── 学习引擎实例 ───────────────────────────────────────────────
initLearningEngine(memoryDir);
const reinforcementLearning = new ReinforcementLearning();
const personalityEvolution = new PersonalityEvolution();
reinforcementLearning.load();
personalityEvolution.load();

// ── 元认知实例 ─────────────────────────────────────────────────
initMetacognition(memoryDir);
const selfReflection = new SelfReflection();
const valueConsistency = new ValueConsistency();
selfReflection.load();
valueConsistency.load();
if (valueConsistency.values.size === 0) {
  valueConsistency.initValues();
}

// ── 启动：记忆衰减 + 视觉噪音自愈 ────────────────────────────────
memorySystem.decay();
memorySystem.purgeVisionNoise();
// 遗忘曲线：每小时衰减一次，让低权重事件随时间自然淡出（原来只在启动时跑一次）
setInterval(() => {
  try { memorySystem.decay(); } catch (err) { console.error('[memory] periodic decay error:', err.message); }
}, 60 * 60 * 1000);
console.log(`[memory] Loaded ${memorySystem.events.length} events after decay.`);
console.log(`[conversation] Loaded ${unifiedDialogueLog.entriesCount} unified dialogue entries.`);

// 当前PAD状态
let currentPAD = loadPAD(padPath);
currentPAD = applyHighIntimacyBootstrap(currentPAD, strategyLayer, memorySystem);
if (isHighIntimacyMode()) {
  savePAD(padPath, currentPAD);
  try {
    const w = JSON.parse(fs.readFileSync(whoamiPath, 'utf8'));
    if (!String(w.relationship_note || '').trim()) {
      w.relationship_note = '已经很亲近，会自然关心他的近况，不是客套。';
      w.last_updated = Date.now();
      fs.writeFileSync(whoamiPath, JSON.stringify(w, null, 2));
    }
  } catch (_) { /* ignore */ }
  userModelInst.syncRelationshipFromScore(HIGH_INTIMACY_REL_FLOOR);
  console.log('[companion] 高亲密度模式已启用（AMADEUS_HIGH_INTIMACY=0 可关闭）');
}
console.log(`[pad] Loaded: P=${currentPAD.P.toFixed(3)} A=${currentPAD.A.toFixed(3)} D=${currentPAD.D.toFixed(3)} S=${currentPAD.S.toFixed(3)}`);

// ──────────────────────────────────────────────────────────────
//  POST /save-memory
// ──────────────────────────────────────────────────────────────
app.post('/save-memory', (req, res) => {
  try {
    const { action, data } = req.body;
    const profile = JSON.parse(fs.readFileSync(memoryPath, 'utf8'));
    if (action === 'observe') {
      const found = profile.user_profile.tentative_observations.find(o => o.trait === data);
      if (found) { found.count++; found.last_seen = new Date().toLocaleString(); }
      else profile.user_profile.tentative_observations.push({ trait:data, count:1, last_seen:new Date().toLocaleString() });
    } else if (action === 'confirm') {
      if (!profile.user_profile.confirmed_habits.includes(data))
        profile.user_profile.confirmed_habits.push(data);
      profile.user_profile.tentative_observations =
        profile.user_profile.tentative_observations.filter(o => o.trait !== data);
    }
    fs.writeFileSync(memoryPath, JSON.stringify(profile, null, 2));
    res.json({ status:'success' });
  } catch (e) { res.status(500).json({ error:e.message }); }
});

// ──────────────────────────────────────────────────────────────
//  GET /pad-state  — 前端轮询用
// ──────────────────────────────────────────────────────────────
function classifyAmbientUtterance(text) {
  const t = String(text || '').trim();
  if (!t || t.length < 2) return null;
  const lowValue = /^(嗯+|啊+|呃+|哦+|好吧|算了|没事|然后呢|这个|那个)$/;
  if (lowValue.test(t) || t.length > 220) return null;

  const selfInfo = /(我(?:叫|是|喜欢|不喜欢|讨厌|习惯|经常|最近|今天|明天|后天|下周|要|想|需要|打算|准备)|记一下|提醒我|别忘了|我有|我的)/;
  const schedule = /(今天|明天|后天|下周|周[一二三四五六日天]|早上|上午|中午|下午|晚上|凌晨|点|会议|面试|考试|ddl|截止|作业|上课|实习|投递)/i;
  const preference = /(喜欢|不喜欢|讨厌|偏好|更想|不想|以后|习惯|常用|别再|记住)/;

  let importance = 0.18;
  let type = 'ambient';
  if (schedule.test(t)) { type = 'ambient_schedule'; importance = 0.42; }
  if (preference.test(t)) { type = 'ambient_preference'; importance = Math.max(importance, 0.36); }
  if (selfInfo.test(t)) { type = 'ambient_self_info'; importance = Math.max(importance, 0.32); }

  if (importance < 0.3 && !selfInfo.test(t)) return null;
  return { type, importance };
}

app.post('/ambient-hearing', (req, res) => {
  try {
    const text = String(req.body.text || '').trim();
    const speaker = String(req.body.speaker || 'unknown');
    const verified = req.body.verified === true;
    const source = String(req.body.source || 'ambient_mic');
    const classified = classifyAmbientUtterance(text);

    if (!verified) {
      return res.json({ ok: true, accepted: false, activeChat: false, reason: 'speaker_unverified' });
    }

    if (!classified) {
      return res.json({ ok: true, accepted: false, activeChat: false, reason: 'low_value' });
    }

    const who = verified ? speaker : `${speaker}:unverified`;
    memorySystem.addObservation(`旁听:${classified.type}`, `${who} ${text}`.slice(0, 80));
    const event = memorySystem.addEvent(
      classified.type,
      `旁听(${who})：${text}`,
      classified.importance,
      { A: 0.01 }
    );

    res.json({
      ok: true,
      accepted: true,
      activeChat: false,
      source,
      speaker,
      verified,
      type: classified.type,
      importance: classified.importance,
      eventId: event && event.id,
    });
  } catch (e) {
    res.status(500).json({ ok: false, error: e.message });
  }
});

app.get('/pad-state', (req, res) => {
  const memBias  = memorySystem.getLongTermPadBias();
  const rawRel = memorySystem.getRelationshipScore();
  const relScore = effectiveRelScore(rawRel);
  const motiv    = motivSystem.update(currentPAD, memBias, relScore);
  res.json({
    pad:      currentPAD,
    memBias,
    relScore,
    rawRelScore: rawRel,
    highIntimacyMode: isHighIntimacyMode(),
    motivation: {
      wants:   motiv.wants,
      fears:   motiv.fears,
      focus:   motiv.currentFocus,
    },
    behavior: {
      current: behaviorSys._lastBehavior,
      count:   behaviorSys._lastBehaviorCount,
    },
    recentEvents: memorySystem.getRecentSignificant(5),
    eventCount:   memorySystem.events.length,
  });
});

// ──────────────────────────────────────────────────────────────
//  GET /internal-state  — 完整内部状态（用于前端展示）
// ──────────────────────────────────────────────────────────────
app.get('/internal-state', (req, res) => {
  try {
    const memBias  = memorySystem.getLongTermPadBias();
    const rawRel = memorySystem.getRelationshipScore();
    const relScore = effectiveRelScore(rawRel);
    const motiv    = motivSystem.update(currentPAD, memBias, relScore);
    
    // 强化学习统计
    const rlStats = reinforcementLearning.getStats();
    
    // 人格特质
    const personality = {
      traits: personalityEvolution.traits,
      values: personalityEvolution.values,
      description: personalityEvolution.getDescription(),
    };
    
    // 元认知洞察
    const metacognition = {
      recentReflections: selfReflection.reflectionHistory.slice(-5),
      insights: selfReflection.insights.slice(-5),
    };
    
    // 用户理解
    const userUnderstanding = {
      stats: userModelInst.model.stats,
      preferences: userModelInst.model.preferences,
      relationship: userModelInst.model.relationship,
      recentEmotions: userModelInst.model.patterns.emotion_history.slice(-5),
    };
    
    // 时间上下文
    const timeContext = getTimeContext();
    
    res.json({
      // PAD 状态
      pad: currentPAD,
      padDescription: padTelemetry(currentPAD),
      
      // 记忆系统
      memory: {
        eventCount: memorySystem.events.length,
        recentEvents: memorySystem.getRecentSignificant(5),
        timeline: memorySystem.timeline.slice(-5),
      },
      
      // 动机系统
      motivation: {
        wants: motiv.wants,
        fears: motiv.fears,
        focus: motiv.currentFocus,
      },
      
      // 行为决策
      behavior: {
        current: behaviorSys._lastBehavior,
        count:   behaviorSys._lastBehaviorCount,
      },
      
      // 自我模型
      selfModel: selfModel.get(),
      
      // 内生目标
      goals: goalSystem.getSummary(),
      
      // 策略延续
      strategy: strategyLayer.getLabel(),
      
      // 人格演化
      personality,
      
      // 强化学习
      learning: rlStats,
      
      // 元认知
      metacognition,
      
      // 用户理解
      userUnderstanding,
      
      // 关系
      relationship: {
        score: relScore,
        memBias,
      },
      
      // 时间
      time: timeContext,

      // 数字生命子系统
      digitalLife: digitalLife.getPublicState(),
      expression: digitalLife.embodiment.expression.snapshot(),
      innerStateSix: innerStateSix.snapshot(),
      dialogueLog: unifiedDialogueLog.snapshot(),
      behaviorContext: behaviorIngest.snapshot(),
      brain: (() => {
        try {
          const b = getBrain();
          return {
            trace: b.getLastTrace(),
            worldModel: b.getWorldSnapshot(),
            selfModel: b.getSelfSnapshot(),
            consciousness: b.getConsciousness(),
            workspace: b.getWorkspace(),
          };
        } catch {
          return null;
        }
      })(),
      subjectCore: subjectCore.snapshot(),
    });
  } catch (e) {
    console.error('[internal-state]', e.message);
    res.status(500).json({ error: e.message });
  }
});

// ──────────────────────────────────────────────────────────────
//  POST /inner-life-cycle — 独处时记忆整理与梦境（可由前端空闲时触发）
// ──────────────────────────────────────────────────────────────
app.post('/inner-life-cycle', (req, res) => {
  try {
    const idleMs = Math.max(0, Number(req.body.idleMs) || 0);
    const result = digitalLife.runIdleCycle({
      idleMs,
      pad: currentPAD,
      memorySystem,
      memory: memorySystem,
      motivationState,
    });
    res.json({ ok: true, ...result });
  } catch (e) {
    res.status(500).json({ ok: false, error: e.message });
  }
});

// ── 统一对话实录 API ─────────────────────────────────────────────
app.get('/dialogue-log', (req, res) => {
  try {
    const limit = Math.min(80, Math.max(1, Number(req.query.limit) || 24));
    res.json({
      ok: true,
      entries: unifiedDialogueLog.getRecent(limit),
      snapshot: unifiedDialogueLog.snapshot(),
    });
  } catch (e) {
    res.status(500).json({ ok: false, error: e.message });
  }
});

app.post('/dialogue-log/append', (req, res) => {
  try {
    const role = req.body.role === 'assistant' ? 'assistant' : 'user';
    const text = String(req.body.text || '');
    const item = unifiedDialogueLog.append(role, text, {
      proactive: req.body.proactive,
      autonomy: req.body.autonomy,
      lite: req.body.lite,
      source: req.body.source || 'client',
      conversationId: req.body.conversationId,
      turnId: req.body.turnId,
      modelJp: req.body.modelJp,
    });
    if (!item) {
      return res.json({
        ok: true,
        item: null,
        dropped: true,
        reason: 'generation_gate_or_dup',
      });
    }
    observeMemoryEvidence(
      role === 'user' ? 'user' : req.body.proactive === true ? 'proactive' : 'assistant',
      item.text,
    );
    // 主动 buffer 在 assistant 归档时消费（证明他已接话），此处不提前清空
    res.json({ ok: true, item });
  } catch (e) {
    res.status(500).json({ ok: false, error: e.message });
  }
});

app.post('/dialogue-log/clear', (req, res) => {
  try {
    const removed = unifiedDialogueLog.clearAll();
    try { memoryPalace.clear(); } catch (_) { /* ignore */ }
    try {
      if (conversationInitiative && typeof conversationInitiative.resetSession === 'function') {
        conversationInitiative.resetSession();
      } else if (conversationInitiative && typeof conversationInitiative.clear === 'function') {
        conversationInitiative.clear();
      }
    } catch (_) { /* ignore */ }
    res.json({ ok: true, removed, palace: memoryPalace.counts?.() || null });
  } catch (e) {
    res.status(500).json({ ok: false, error: e.message });
  }
});

// ── 记忆宫殿（长期记忆中枢）──────────────────────────────────────
app.get('/memory/palace', (req, res) => {
  try {
    res.json({ ok: true, ...memoryPalace.snapshot() });
  } catch (e) {
    res.status(500).json({ ok: false, error: e.message });
  }
});

app.post('/memory/palace/archive', (req, res) => {
  try {
    const userText = String(req.body.userText || '');
    const assistantText = String(req.body.assistantText || '');
    const admission = memoryAdmission.assessUserText(userText, { source: 'user' });
    const userRepliedToProactive = req.body.userRepliedToProactive === true
      || memoryPalace.hasPendingProactive();
    if (userRepliedToProactive) memoryPalace.consumeProactiveOnUserReply();
    const result = memoryPalace.archiveTurn({
      userText,
      assistantText,
      userAdmission: admission,
      proactive: req.body.proactive === true,
      userRepliedToProactive,
      compressed: req.body.compressed,
      conflict: req.body.conflict,
    });
    if (result.ok) {
      observeMemoryEvidence('assistant', assistantText);
      butlerKernel.journal.append('memory.palace_archive', {
        room: result.room,
        text: result.node?.text || '',
        reason: result.reason,
      }, { actor: 'amadeus', source: 'memory' });
    } else {
      butlerKernel.journal.append('memory.write_rejected', {
        layer: 'palace',
        reason: result.reason,
        preview: userText.slice(0, 48),
      }, { actor: 'system', source: 'memory' });
    }
    res.json({ ok: true, ...result, counts: memoryPalace.counts() });
  } catch (e) {
    res.status(500).json({ ok: false, error: e.message });
  }
});

app.post('/memory/palace/navigate', (req, res) => {
  try {
    const userText = String(req.body.userText || req.body.text || '');
    const nav = memoryPalace.navigate(userText, {
      mood: req.body.mood ?? currentPAD?.P,
      topK: req.body.topK,
      minScore: req.body.minScore,
    });
    res.json({ ok: true, ...nav });
  } catch (e) {
    res.status(500).json({ ok: false, error: e.message });
  }
});

app.post('/memory/palace/import', (req, res) => {
  try {
    const added = memoryPalace.importLegacyRooms(req.body.rooms || req.body || {});
    res.json({ ok: true, added, counts: memoryPalace.counts() });
  } catch (e) {
    res.status(500).json({ ok: false, error: e.message });
  }
});

app.post('/memory/palace/clear', (req, res) => {
  try {
    const counts = memoryPalace.clear();
    butlerKernel.journal.append('memory.palace_clear', { counts }, { actor: 'system', source: 'memory' });
    res.json({ ok: true, counts });
  } catch (e) {
    res.status(500).json({ ok: false, error: e.message });
  }
});

// ── 行为数据上报（无原生采集，仅接收管道）────────────────────────
app.post('/behavior-report', (req, res) => {
  try {
    const result = behaviorIngest.ingest(req.body || {});
    butlerKernel.journal.append('behavior.reported', { report: req.body || {} }, {
      actor: 'sensor',
      source: 'perception',
    });
    res.json({ ok: true, ...result, snapshot: behaviorIngest.snapshot() });
  } catch (e) {
    res.status(500).json({ ok: false, error: e.message });
  }
});

// ── 对话内主动意图：先决定是否开口，再交给 lite 生成具体台词 ──────────
app.post('/initiative/session-start', (_req, res) => {
  try {
    const state = conversationInitiative.markSessionStart();
    res.json({ ok: true, state });
  } catch (e) {
    res.status(500).json({ ok: false, error: e.message });
  }
});

app.post('/initiative/decide', (req, res) => {
  try {
    if (!isProactiveEnabled()) return res.json(proactiveDisabledHold());

    const turnCtx = dialogueTurnContext(req.body);
    const held = holdProactiveDecision(turnCtx);
    if (held) return res.json(held);

    const phase = String(req.body.phase || 'floor_release');
    const lastUserText = String(req.body.lastUserText || req.body.userText || '');
    const social = readSocialField({
      facePresent: req.body.facePresent === true,
      faceMs: Number(req.body.faceMs) || 0,
      idleMs: turnCtx.idleMsSinceUser || Number(req.body.idleMs) || 0,
      quietMs: Number(req.body.quietMs) || Number(req.body.idleMs) || 0,
      lastUserText,
      lastReplyText: String(req.body.lastReplyText || req.body.replyText || ''),
      alreadyTalking: req.body.alreadyTalking === true || req.body.dialogueStarted === true,
      pad: req.body.pad || currentPAD,
      relScore: Number.isFinite(Number(req.body.relScore))
        ? Number(req.body.relScore)
        : effectiveRelScore(memorySystem.getRelationshipScore()),
      dnd: req.body.dnd === true,
      isThinking: turnCtx.isThinking,
      ttsPlaying: req.body.ttsPlaying === true,
      awaitingProactiveReply: req.body.awaitingProactiveReply === true,
    });
    const shared = {
      pad: req.body.pad || currentPAD,
      relScore: Number.isFinite(Number(req.body.relScore))
        ? Number(req.body.relScore)
        : effectiveRelScore(memorySystem.getRelationshipScore()),
      dnd: req.body.dnd === true,
      proactiveQuotaOk: req.body.proactiveQuotaOk !== false,
      entropy: Number.isFinite(Number(req.body.entropy)) ? Number(req.body.entropy) : undefined,
      eventDriven: req.body.eventDriven === true,
      senseDriven: req.body.senseDriven === true,
      eventKind: String(req.body.eventKind || ''),
      force: req.body.force === true,
      pendingUserTurn: turnCtx.pendingUserTurn,
      isThinking: turnCtx.isThinking,
      idleMsSinceUser: turnCtx.idleMsSinceUser,
      social,
    };
    const subjectDecision = subjectCore.planProactive({
      now: Date.now(),
      contextFresh: req.body.contextFresh === true,
      anchor: lastUserText,
      pad: shared.pad,
      motivationState,
      openThoughts: soulRuntime.snapshot().thoughts,
      dnd: shared.dnd,
      pendingUserTurn: shared.pendingUserTurn,
      isThinking: shared.isThinking,
      awaitingReply: req.body.awaitingProactiveReply === true,
      ignoredStreak: conversationInitiative.state?.ignoredStreak,
      lastSpokenAt: conversationInitiative.state?.lastSpokenAt,
    });
    // 跟话/主动开口不再靠关键词映射动作。没有持续存在的念头就保持沉默。
    if (!subjectDecision.shouldSpeak) {
      return res.json({
        ok: true,
        shouldSpeak: false,
        action: 'hold',
        reason: subjectDecision.reason,
        nextCheckMs: subjectDecision.nextCheckMs || 90000,
      });
    }
    const thoughtDecision = conversationInitiative.decideThought({
      ...shared,
      ...subjectDecision,
      phase,
      contextFresh: false,
      awaitingReply: req.body.awaitingProactiveReply === true,
    });
    return res.json({ ok: true, ...thoughtDecision });
    /* c8 ignore next */
    let decision;
    if (phase === 'presence' || phase === 'copresence') {
      decision = conversationInitiative.decidePresence({
        ...shared,
        facePresent: req.body.facePresent === true,
        alreadyTalking: req.body.alreadyTalking === true || req.body.dialogueStarted === true || phase === 'copresence',
        dialogueStarted: req.body.dialogueStarted === true || req.body.alreadyTalking === true || phase === 'copresence',
        lastUserText,
      });
    } else if (phase === 'coldness') {
      decision = conversationInitiative.decideColdness({
        ...shared,
        lastUserText,
        userText: lastUserText,
        replyText: req.body.replyText,
      });
    } else if (phase === 'idle') {
      decision = conversationInitiative.decideIdle({
        ...shared,
        idleMs: req.body.idleMs,
        contextFresh: req.body.contextFresh === true,
        topicContaminated: memoryAdmission.contaminatedFragments(lastUserText).length > 0,
        lastUserText,
        userPresenceActive: req.body.userPresenceActive === true,
      });
    } else {
      decision = conversationInitiative.decide({
        userText: req.body.userText,
        replyText: req.body.replyText,
        phase,
        elapsedMs: req.body.elapsedMs,
        ...shared,
      });
    }
    res.json({ ok: true, ...decision });
  } catch (e) {
    res.status(500).json({ ok: false, shouldSpeak: false, action: 'hold', error: e.message });
  }
});

app.get('/memory-admission', (_req, res) => {
  res.json({ ok: true, ...memoryAdmission.snapshot() });
});

/**
 * 共在心跳：摄像头只上报「他还在不在」。
 * 这里用内驱累积 × 读场时机决定要不要开口——可以只是安静坐着。
 */
app.post('/initiative/presence-tick', (req, res) => {
  try {
    if (!isProactiveEnabled()) {
      return res.json({
        ...proactiveDisabledHold(),
        nextCheckMs: 60000,
        social: null,
        urge: null,
        speakHint: 'proactive_disabled',
      });
    }

    const body = req.body || {};
    const turnCtx = dialogueTurnContext(body);
    const held = holdProactiveDecision(turnCtx);
    if (held) {
      return res.json({
        ...held,
        nextCheckMs: 6000,
        social: null,
        urge: null,
        speakHint: held.reason,
      });
    }

    const facePresent = body.facePresent === true;
    const faceMs = Math.max(0, Number(body.faceMs) || 0);
    const idleMs = Math.max(0, Number(body.idleMs) || 0);
    const quietMs = Math.max(0, Number(body.quietMs) || idleMs);
    const dtMs = Math.max(500, Math.min(20000, Number(body.dtMs) || 4000));
    const alreadyTalking = body.alreadyTalking === true || body.dialogueStarted === true;
    const lastUserText = String(body.lastUserText || '').slice(0, 240);
    const lastReplyText = String(body.lastReplyText || '').slice(0, 240);
    const relScore = Number.isFinite(Number(body.relScore))
      ? Number(body.relScore)
      : effectiveRelScore(memorySystem.getRelationshipScore());
    const pad = body.pad && typeof body.pad === 'object' ? body.pad : currentPAD;
    const dnd = body.dnd === true;
    const userPresenceActive = body.userPresenceActive === true;

    if (userPresenceActive) {
      return res.json({
        ok: true,
        shouldSpeak: false,
        action: 'hold',
        reason: 'user_active',
        nextCheckMs: 6000,
        social: null,
        urge: null,
        speakHint: '他正在操作或输入，先不要打断',
      });
    }

    const social = readSocialField({
      facePresent,
      faceMs,
      idleMs,
      quietMs,
      lastUserText,
      lastReplyText,
      alreadyTalking,
      dialogueStarted: alreadyTalking,
      pad,
      relScore,
      dnd,
      isThinking: body.isThinking === true,
      ttsPlaying: body.ttsPlaying === true,
      awaitingProactiveReply: body.awaitingProactiveReply === true,
    });

    const sinceKurisuMs = Number(body.sinceKurisuMs);
    const sheSpokeRecently = body.sheSpokeRecently === true
      || (Number.isFinite(sinceKurisuMs) && sinceKurisuMs >= 0 && sinceKurisuMs < 12000);

    // 感知循环只提供“是否适合开口”的环境信息；不再每次心跳制造一个枚举内驱。
    void dtMs;
    void sheSpokeRecently;
    const openThought = soulRuntime.state.thoughts
      .filter((item) => item.status === 'open')
      .sort((a, b) => Number(b.tension) - Number(a.tension))[0];
    const urgeIntensity = Number(openThought?.tension) || 0;
    const senseMs = nextSenseMs(social, urgeIntensity);
    const subjectDecision = subjectCore.planProactive({
      now: Date.now(),
      contextFresh: body.contextFresh === true,
      anchor: lastUserText,
      pad,
      motivationState,
      openThoughts: soulRuntime.snapshot().thoughts,
      dnd,
      pendingUserTurn: turnCtx.pendingUserTurn,
      isThinking: turnCtx.isThinking,
      awaitingReply: body.awaitingProactiveReply === true,
      ignoredStreak: conversationInitiative.state?.ignoredStreak,
      lastSpokenAt: conversationInitiative.state?.lastSpokenAt,
    });
    const decision = conversationInitiative.decideThought({
      ...subjectDecision,
      phase: alreadyTalking ? 'copresence' : 'presence',
      social,
      pendingUserTurn: turnCtx.pendingUserTurn,
      isThinking: turnCtx.isThinking,
      awaitingReply: body.awaitingProactiveReply === true,
      dnd,
      nextCheckMs: Math.max(60000, senseMs),
    });

    res.json({
      ok: true,
      ...decision,
      nextCheckMs: decision.nextCheckMs || senseMs,
      social,
      urge: subjectDecision.shouldSpeak
        ? {
            id: subjectDecision.thoughtId,
            intent: subjectDecision.thought,
            intensity: subjectDecision.tension,
            source: 'persistent_thought',
          }
        : null,
      speakHint: subjectDecision.desire || decision.reason || '',
    });
  } catch (e) {
    res.status(500).json({
      ok: false,
      shouldSpeak: false,
      action: 'hold',
      nextCheckMs: 10000,
      error: e.message,
    });
  }
});

app.get('/initiative/state', (_req, res) => {
  res.json({
    ok: true,
    ...conversationInitiative.snapshot(),
    subject: {
      relationship: soulRuntime.state.relationship,
      affect: soulRuntime.state.affect,
      openThoughts: soulRuntime.state.thoughts.filter((item) => item.status === 'open').slice(-5),
      core: subjectCore.snapshot(),
    },
  });
});

app.post('/initiative/feedback', (req, res) => {
  try {
    const state = conversationInitiative.registerFeedback({ type: req.body.type, text: req.body.text });
    if (req.body.type === 'reply') soulRuntime.registerFeedback(req.body.text);
    subjectCore.registerFeedback({ type: req.body.type, text: req.body.text });
    res.json({ ok: true, state });
  } catch (e) {
    res.status(500).json({ ok: false, error: e.message });
  }
});

app.post('/initiative/sent', (req, res) => {
  try {
    const thoughtId = String(req.body.thoughtId || conversationInitiative.state?.activeThoughtId || '');
    const state = conversationInitiative.registerSent({
      text: req.body.text,
      action: req.body.action,
      thoughtId,
    });
    if (thoughtId) soulRuntime.markThoughtExpressed(thoughtId, req.body.text);
    subjectCore.integrateOutcome({ mode: 'proactive', reply: req.body.text, accepted: !!String(req.body.text || '').trim() });
    const urgeId = String(req.body.urgeId || '');
    if (urgeId) digitalLife.satisfyAutonomyUrge(urgeId, 'spoken');
    res.json({ ok: true, state });
  } catch (e) {
    res.status(500).json({ ok: false, error: e.message });
  }
});

// ── 主动开口 lite 通道（低 token、快响应）────────────────────────
app.post('/chat/lite', async (req, res) => {
  try {
    const userLine = String(req.body.userText || req.body.message || '（想说话）').trim();
    const isAutonomyInitiative = req.body.isAutonomyInitiative === true || /^（想说话）/.test(userLine);
    const conversationId = String(req.body.conversationId || '').trim();
    if (
      isAutonomyInitiative
      && unifiedDialogueLog.hasUnansweredUserTurn(conversationId)
    ) {
      return res.json({
        ok: true,
        skipped: true,
        reason: 'pending_user_turn',
        response: '',
        choices: [{ message: { role: 'assistant', content: '' } }],
      });
    }
    let lastUserAnchor = String(req.body.lastUserAnchor || '').slice(0, 200);
    const lastKurisuAnchor = String(req.body.lastKurisuAnchor || '').slice(0, 200);
    const legacyAnchor = String(req.body.proactiveAnchor || '').slice(0, 200);
    const isConversationInitiative = req.body.conversationInitiative === true;
    const initiativeAction = String(req.body.initiativeAction || '').trim();
    const initiativeReason = String(req.body.initiativeReason || '').slice(0, 180);
    const initiativePhase = String(req.body.initiativePhase || '').trim();
    const contextFresh = req.body.contextFresh === true;
    if (isAutonomyInitiative && !contextFresh) lastUserAnchor = '';
    const recentProactive = (Array.isArray(req.body.recentProactive) ? req.body.recentProactive : [])
      .map((item) => String(item || '').replace(/\s+/g, ' ').trim().slice(0, 100))
      .filter(Boolean)
      .slice(-6);
    const requestedDelivery = req.body.delivery && typeof req.body.delivery === 'object'
      ? req.body.delivery
      : {};
    const delivery = {
      style: String(requestedDelivery.style || (isConversationInitiative ? 'casual' : 'poke')).slice(0, 20),
      bubbleCount: Math.max(1, Math.min(5, Number(requestedDelivery.bubbleCount) || 2)),
      maxCharsPerBubble: Math.max(12, Math.min(80, Number(requestedDelivery.maxCharsPerBubble) || 48)),
      allowNonSemantic: requestedDelivery.allowNonSemantic === true,
      thoughtId: String(requestedDelivery.thoughtId || req.body.thoughtId || ''),
      urgeId: String(requestedDelivery.urgeId || req.body.urgeId || ''),
    };
    const clientOwnsLog = req.body.clientOwnsLog === true;
    const idleMs = Math.max(0, Number(req.body.idleMsSinceUser) || 0);
    const requestedModel = req.body.model
      || process.env.AMADEUS_LITE_MODEL
      || process.env.AMADEUS_CHAT_MODEL
      || 'amadeus-kurisu-swallow:8b';
    // 主动对话和普通回答是同一个主体的两种状态，不切换模型人格。
    // 差异只由下面的社交上下文、节奏和输出门控决定。
    const proactiveModel = String(process.env.AMADEUS_PROACTIVE_MODEL || '').trim();
    const model = isAutonomyInitiative && proactiveModel ? proactiveModel : requestedModel;
    const japaneseLiteModel = getReplyLanguageMode() === 'ja'
      && /(?:swallow|llm[-_]?jp|kurisu(?:-v?\d+)?(?::|$))/i.test(String(model));
    let nativeLastUserAnchor = lastUserAnchor;
    let nativeLastKurisuAnchor = lastKurisuAnchor;
    if (japaneseLiteModel && /[\u3400-\u9fff]/.test(`${lastUserAnchor}${lastKurisuAnchor}`)) {
      // 主动模型只看日语母语提示；把中文显示层的最近一句先做一次
      // 事实约束翻译，避免它把中文锚点误当成模板或情绪标签。
      try {
        const [userJp, kurisuJp] = await Promise.all([
          lastUserAnchor ? _translateUserChineseToJapanese(lastUserAnchor) : '',
          lastKurisuAnchor ? _translateUserChineseToJapanese(lastKurisuAnchor) : '',
        ]);
        if (userJp && countScriptChars(userJp).kana > 0) nativeLastUserAnchor = String(userJp).slice(0, 220);
        if (kurisuJp && countScriptChars(kurisuJp).kana > 0) nativeLastKurisuAnchor = String(kurisuJp).slice(0, 220);
      } catch (_) { /* 原文仍作为上下文，不因翻译失败阻断主动对话 */ }
    }
    const activeThought = delivery.thoughtId
      ? soulRuntime.snapshot().thoughts.find((item) => item.id === delivery.thoughtId && item.status === 'open')
      : null;
    const subjectPlan = (isAutonomyInitiative || isConversationInitiative)
      ? subjectCore.planProactive({
        contextFresh,
        anchor: nativeLastUserAnchor,
        pad: currentPAD,
        motivationState,
        openThoughts: activeThought ? [activeThought] : soulRuntime.snapshot().thoughts,
        dnd: false,
      })
      : null;
    const proactiveContentPlan = (isAutonomyInitiative || isConversationInitiative)
      ? buildProactiveContentPlan({
        action: subjectPlan?.action || initiativeAction,
        contextFresh,
        anchor: nativeLastUserAnchor,
        thought: subjectPlan?.source === 'persistent_thought' ? subjectPlan.thought : '',
      })
      : null;
    // 没有来自当前话题或持久念头的内容核，就不让模型用随机寒暄填空。
    if ((isAutonomyInitiative || isConversationInitiative) && !proactiveContentPlan.shouldGenerate) {
      return res.json({
        ok: true,
        skipped: true,
        reason: subjectPlan?.reason || proactiveContentPlan.reason,
        response: '',
        choices: [{ message: { role: 'assistant', content: '' } }],
      });
    }
    const configuredChatTemp = Number(process.env.AMADEUS_CHAT_TEMP);
    const requestedTemp = Number(req.body.temperature);
    const baseTemp = requestedTemp
      || (Number.isFinite(configuredChatTemp) && configuredChatTemp > 0 ? configuredChatTemp : 0.62);
    // 主动内容要有一点随机性，但不能让 0.85～0.9 的高温把具体承接
    // 采样成“突然怎么了/累了吗”这类固定客服回路。
    const temp = isAutonomyInitiative || isConversationInitiative
      ? Math.min(0.62, Math.max(0.42, baseTemp))
      : baseTemp;
    const requestedLiteTok = Number(req.body.max_tokens) || 320;
    // kurisu 会先烧 thinking：主动/接话 lite 不能再压到 96，否则 content 恒空
    const maxTok = isConversationInitiative || isAutonomyInitiative
      ? Math.max(280, Math.min(512, requestedLiteTok || 320))
      : /kurisu|deepseek-r1/i.test(String(model))
      ? Math.max(280, Math.min(384, requestedLiteTok))
      : Math.max(120, Math.min(384, requestedLiteTok));
    const liteNumCtxEnv = Number(process.env.AMADEUS_OLLAMA_NUM_CTX);
    const liteNumCtx = Number.isFinite(liteNumCtxEnv) && liteNumCtxEnv > 0 ? liteNumCtxEnv : 2048;

    if (!isAutonomyInitiative && userLine && !/^（想说话）/.test(userLine)) {
      unifiedDialogueLog.append('user', userLine, { source: 'lite' });
      observeMemoryEvidence('user', userLine);
    }

    const recentDialogue = (Array.isArray(req.body.recentDialogue) ? req.body.recentDialogue : [])
      .map((item) => ({
        role: item?.role === 'assistant' || item?.role === 'kurisu' ? 'assistant' : 'user',
        content: String(item?.content || item?.text || '').replace(/\s+/g, ' ').trim().slice(0, 220),
      }))
      .filter((item) => item.content)
      .slice(-8);
    const dialogue = isConversationInitiative
      ? [
          { role: 'user', content: lastUserAnchor },
          { role: 'assistant', content: lastKurisuAnchor },
          {
            role: 'user',
            content: '（这是你自己的交流意图。依照系统给出的意图行动，不要把它统一改写成追问；没有真实内容就只说 [SILENCE]。）',
          },
        ].filter((item) => item.content)
      : isAutonomyInitiative
        ? (
          recentDialogue.length
            ? [...recentDialogue]
            : (contextFresh && lastUserAnchor ? [{ role: 'user', content: lastUserAnchor }] : [])
        )
        : (
          recentDialogue.length
            ? [...recentDialogue]
            : (userLine ? [{ role: 'user', content: userLine }] : [])
        );
    if (!isConversationInitiative && !isAutonomyInitiative && userLine
      && (!dialogue.length || dialogue[dialogue.length - 1].role !== 'user'
        || dialogue[dialogue.length - 1].content !== userLine)) {
      dialogue.push({ role: 'user', content: userLine });
    }
    if (!isConversationInitiative && /^（想说话）/.test(userLine)) dialogue.push({ role: 'user', content: userLine });
    // 主动开口也要看见近况，否则会说出让人觉得「没感觉到人」的怪话
    const conversationCtx = isConversationInitiative
      ? ''
      : isAutonomyInitiative
        ? (recentDialogue.length
          ? recentDialogue.map((m) => `${m.role === 'user' ? (japaneseLiteModel ? '彼' : '他') : (japaneseLiteModel ? '紅莉栖' : 'Kurisu')}: ${m.content}`).join('\n').slice(0, 900)
          : unifiedDialogueLog.toPromptBlock({ maxChars: 700, userText: '' }))
        : (recentDialogue.length
          ? recentDialogue.map((m) => `${m.role === 'user' ? (japaneseLiteModel ? '彼' : '他') : (japaneseLiteModel ? '紅莉栖' : 'Kurisu')}: ${m.content}`).join('\n').slice(0, 900)
          : '');
    const relScore = effectiveRelScore(memorySystem.getRelationshipScore());

    const anchor = lastKurisuAnchor || lastUserAnchor || legacyAnchor;
    let continuityLine = '【主动】像偶尔想起来才发一句，短、自然，禁止查岗连发。';
    if (isConversationInitiative && lastKurisuAnchor) {
      const actionContracts = {
        care: '察觉到他有未说完的情绪。落在他话里的具体细节上，关心但不扮心理咨询师；可以问，也可以只陪一句。',
        probe: '你对他留下的私人线索产生了具体好奇。只追一个细节，不做泛泛采访。',
        stance: '你形成了自己的判断或反驳欲。说出一个新的立场、疑点或推论；可以不提问。',
        tease: '你想接住关系里的玩笑。可以挖苦、回怼、抬杠——因为你想这么说，不是为了讨好。',
        share: '他的话触发了你自己的联想。分享一个属于你的念头，不必把话硬递回去。',
        poke: '你只是想发出一点存在感。动机可以很轻（逗一下、骚扰一下、哼一声），措辞按你本人自由发挥。',
        thought: '这个念头已经在你心里持续了一段时间。直接表达它想表达的东西，不要把它改造成泛泛追问。',
      };
      continuityLine = [
        '【对话内自主意图】这不是固定补话、定时问候或第二次回答。',
        `他刚才说：「${lastUserAnchor}」`,
        `你刚回答：「${lastKurisuAnchor}」`,
        `你此刻想开口的原因：${initiativeReason || '产生了一个具体内在动机'}`,
        `行动意图（只说明动机，不规定台词）：${actionContracts[initiativeAction] || '顺着真实动机自然说一句。'}`,
        `发送方式：${delivery.style}；可发 1～${delivery.bubbleCount} 个气泡，长短随此刻真正想说的内容变化。气泡之间用空行分隔。`,
        '你是独立的灵魂——牧濑红莉栖本人。话从你此刻真实想说的来。',
        delivery.allowNonSemantic
          ? '长短随你；没想说就输出 [SILENCE]。'
          : '可以陈述、关心或表达态度。没想说就输出 [SILENCE]。',
      ].join('\n');
    } else if (isAutonomyInitiative && (initiativePhase === 'presence' || initiativePhase === 'copresence')) {
      const midChat = initiativePhase === 'copresence' || recentDialogue.length > 0 || !!lastUserAnchor;
      continuityLine = (midChat
        ? [
          '【共同在场 · 可插话】你们已经在同一个文字窗口里。聊天不必严格你一句我一句；你可以突然再丢一句。',
          '当前事实：不是电话通话，不是第一次发现他在，也不是「你们实验室还没打过来」。',
          lastUserAnchor ? `他最近一句：「${lastUserAnchor}」——有的话就轻轻挂上；没有也只需像还在场那样说话。` : '没有必须接的上一句时，只发一点存在感或小念头即可。',
          initiativeReason ? `此刻动机：${initiativeReason}` : '',
          '禁止台词：刚刚才注意到 / 还没打电话 / 你们怎么还不打来 / 才发现你在——这些与当前情景矛盾。',
          '表达层：你是独立的灵魂——牧濑红莉栖；可以插话，因为你想说，不是职责。',
          `可发 1～${delivery.bubbleCount} 个气泡，气泡间空一行；有话就展开，没话不凑数。`,
        ]
        : [
          '【察觉到人 · 先开口】两个人相见，总有人先说话。这里是文字窗口刚碰上，不是打电话。',
          initiativeReason ? `此刻动机：${initiativeReason}` : '',
          '意图层：可以搞笑、轻微骚扰、开玩笑、或只让他感觉到你也感觉到了他——只要是你想。',
          '禁止编造通话状态（打电话/还没打来/接通）。',
          '表达层：牧濑红莉栖本人；自由发挥，不要套固定开场模板。',
          `可发 1～${delivery.bubbleCount} 个气泡，气泡间空一行；长短随你。`,
          '不要查岗说教，不要解释“我为什么主动”。',
        ]
      ).filter(Boolean).join('\n');
    } else if (isAutonomyInitiative && initiativePhase === 'deferred_speak') {
      continuityLine = [
        '【到点开口】你答应过这个时刻要主动说话。时间到了。',
        initiativeReason ? `约定：${initiativeReason}` : '',
        lastUserAnchor ? `事由：「${lastUserAnchor}」` : '',
        '用你自己的口吻开口；可以毒舌、可以正经、可以短——但要让人听懂你在履行约定。',
        '禁止自称 AI/程序，禁止说办不到，禁止念系统播报腔「时间到了请起床」。',
        `可发 1～${delivery.bubbleCount} 个气泡，气泡间空一行。`,
      ].filter(Boolean).join('\n');
    } else if (isAutonomyInitiative && initiativePhase === 'coldness') {
      continuityLine = [
        '【冷感回拉】气氛有点冷/敷衍或他嫌无聊。你想把场子拉回来——因为你在意这场对话，不是讨好。',
        lastUserAnchor ? `他上一句：「${lastUserAnchor}」——优先接这句里的词和情绪。` : '',
        initiativeReason ? `此刻动机：${initiativeReason}` : '',
        '意图层：顺着他上一句给一点具体动静（吐槽、半句正经、或接他刚说的处境）；不要另起电话/见面戏。',
        '禁止元评论收束：「又是这个话题吗」「又是那个话题」「好无聊」这种把话题判死刑的固定腔——他喊无聊时尤其禁止复读无聊。',
        '禁止编造：打电话/还没打来/刚刚才注意到/你们怎么还不来——这些都不是当前事实。',
        '表达层：你是独立的灵魂；毒舌、傲、软都可以，只要是你此刻想说的。',
        `可发 1～${delivery.bubbleCount} 个气泡，气泡间空一行。`,
      ].filter(Boolean).join('\n');
    } else if (isAutonomyInitiative && lastUserAnchor && contextFresh) {
      continuityLine = [
        '【主动开口 · 接他上一句】',
        `他上一句：「${lastUserAnchor}」`,
        '你要接着他的话开口；这句话是他说的，不是你自己刚说的。',
        '用你自己的声音说，不要通用陪伴腔。',
        `按 ${delivery.style} 方式发 1～${delivery.bubbleCount} 个气泡，气泡间空一行。`,
      ].join('\n');
    } else if (isAutonomyInitiative) {
      continuityLine = [
        '【空闲时随手开口】这不是任务或提醒——是你自己想说。',
        '意图可以很轻：吐槽、玩笑、轻微骚扰、一个小念头、一点动静。',
        '你是独立的灵魂；措辞按牧濑红莉栖本人自由发挥。',
        `可发 1～${delivery.bubbleCount} 个气泡，气泡间空一行。`,
        '不要从旧聊天里硬捞名词续讲，不要说教，不要汇报自己在做实验。',
      ].join('\n');
    } else if (lastKurisuAnchor) {
      continuityLine = `【主动延续】你上一轮主动说的是：「${lastKurisuAnchor}」——可顺着补半句，别丢题。`;
    } else if (!isAutonomyInitiative && legacyAnchor) {
      continuityLine = `【承接话题】他上一句：「${legacyAnchor}」——顺着聊，别把用户话当成你自己说的。`;
    }

    // 主动 lite 也必须和普通聊天使用同一套母语分支。此前这里虽然普通
    // /chat 已经切到日文 soul，但 /chat/lite 仍把中文 action contract、
    // continuity 和中文 soul 一起塞给 swallow，结果就是“触发合理但一开口
    // 又像客服/翻译器”。日语模型只保留真实上下文和内在动机，不再接收整段
    // 中文模板作为台词指令。
    const nativeActionContracts = {
      care: '彼の言葉に残った感情を、具体的な一箇所だけ受け止める。心理カウンセラーのように分析しない。',
      probe: '彼の言葉から気になった一点だけ、自分の好奇心として追う。一般的なインタビューにしない。',
      stance: '自分の判断・反論・推論をひとつ言う。質問で終える必要はない。',
      tease: '恋人同士の軽い冗談や言い返しを、自分が言いたいから言う。機嫌取りにしない。',
      share: '彼の言葉から連想した、自分の考えをひとつ共有する。無理に彼へボールを返さない。',
      poke: '軽く存在を知らせる。からかい、短い声、ひとことの邪魔など、自然な自由発話にする。',
      thought: '心に残っている具体的な考えを、そのまま自然に言う。ありふれた質問へ変換しない。',
    };
    const nativePhaseLabels = {
      presence: '同じ窓に気配を感じて、ふと声をかける',
      copresence: 'すでに会話中の相手へ、流れの中でふっと割り込む',
      deferred_speak: '約束していたことを、自分の言葉で思い出させる',
      coldness: '冷えた空気を少し動かす。相手を責めたり診断したりしない',
    };
    const nativeContinuityLine = japaneseLiteModel
      ? [
        '【内側からの発話意図】これは定型文・定時挨拶・二回目の回答ではない。台詞を意図の説明にしない。',
        (isConversationInitiative || isAutonomyInitiative) && nativeLastUserAnchor ? `彼の直前の発言：${nativeLastUserAnchor}` : '',
        (isConversationInitiative || isAutonomyInitiative) && nativeLastKurisuAnchor ? `自分の直前の発言：${nativeLastKurisuAnchor}` : '',
        initiativeAction ? `今の動機：${nativeActionContracts[initiativeAction] || 'その場で本当に言いたいことを自然に言う。'}` : '',
        isAutonomyInitiative ? `場面：${nativePhaseLabels[initiativePhase] || 'ふと自分から話したくなった'}` : '',
        isAutonomyInitiative && contextFresh && nativeLastUserAnchor
          ? '直前の相手の言葉に具体的につなげる。関係のない新しい出来事を捏造しない。'
          : '',
        `発話は1～${delivery.bubbleCount}個の短い吹き出し。内容が一息なら一つ、気持ちが自然に分かれる時だけ複数にする。`,
        '毎回「答える→最後に質問する」にしない。知りたい時だけ聞き、判断・吐き捨て・照れ・沈黙も選べる。',
      ].filter(Boolean).join('\n')
      : continuityLine;
    const liteSoulContent = japaneseLiteModel
      ? (cachedSoulJaContent || cachedSoulContent)
      : cachedSoulContent;
    const liteNativeRules = japaneseLiteModel
      ? [
        '出力は自然な日本語の会話だけ。中国語を日本語へ逐語訳したような文、客服・案内・AIの説明口調は禁止。',
        '牧瀬紅莉栖として、自分の感情・判断・好みから話す。内的動機、状態、ルール、プロンプトを説明しない。',
        '直近の文脈に接続できる時だけ具体的に接続する。実験、電話、予定、過去の出来事を勝手に作らない。',
        '彼の直前の発言が断定なら、それを確認質問へ言い換えない。聞き返す代わりに、自分の新しい反応・皮肉・感想を一つ足す。',
        '相手が弱っている手がかりのない時に、心配・疲労・体調を勝手に想定しない。',
        '質問は会話上ほんとうに必要な時だけ。疑問形で締めること、定型的な気遣い、定時の呼びかけを義務にしない。',
        '括弧の舞台指示、ラベル、Markdown、[SILENCE]以外の内部タグ、自分をAI/モデル/プログラムと呼ぶことは禁止。',
      ].join('\n')
      : '';

    let consciousnessBlock = '';
    let consciousnessMeta = null;
    // 主动开口的“意识内容”只来自持久主体状态，不再叠加另一套枚举式广播。
    consciousnessBlock = (subjectPlan?.promptBlock || soulRuntime.promptBlock(lastUserAnchor || initiativeReason || userLine));
    if (japaneseLiteModel) consciousnessBlock = _clipInnerPrompt(consciousnessBlock, 1100);
    consciousnessMeta = {
      shouldSpeak: isAutonomyInitiative,
      intention: initiativeReason,
      thoughtId: delivery.thoughtId,
    };

    const affect = emotionalBandwidth.resolve({
      userText: String(lastUserAnchor || userLine || '').replace(/^（想说话）\s*/, ''),
      pad: currentPAD,
      relScore,
      relHigh: relScore >= 0.55,
    });

    const litePrompt = [
      liteSoulContent
        ? `${japaneseLiteModel ? '【人格と魂の核】' : '【核心身份与灵魂锚点】'}\n${_clipInnerPrompt(liteSoulContent, japaneseLiteModel ? 2200 : 1500)}`
        : '',
      cachedCharacterRules && !japaneseLiteModel
        ? `【角色边界】\n${_clipInnerPrompt(cachedCharacterRules, 520)}`
        : '',
      cachedVoiceContent && !japaneseLiteModel ? `【口吻】\n${_clipInnerPrompt(cachedVoiceContent, 900)}` : '',
      conversationCtx,
      consciousnessBlock,
      japaneseLiteModel && proactiveContentPlan
        ? buildJapaneseContentPlanBlock(proactiveContentPlan)
        : '',
      nativeContinuityLine,
      affect.block || '',
      japaneseLiteModel
        ? `親密度 ${relScore.toFixed(2)} · 内側の状態 ${padTelemetry(currentPAD)}`
        : `亲近 ${relScore.toFixed(2)} · 内在 ${padTelemetry(currentPAD)}`,
      !isAutonomyInitiative && !isConversationInitiative
        ? (japaneseLiteModel
          ? `【今回つなぐ内容】彼の直前の発言「${userLine.slice(0, 260)}」を具体的に受ける。実験・電話・別の挨拶へ勝手に話題を変えない。`
          : `【本轮必须接住】直接回应他刚才这句：「${userLine.slice(0, 260)}」；先听懂具体内容再说，不要凭空把话题改成实验、电话或别的寒暄。`)
        : '',
      japaneseLiteModel ? liteNativeRules : '只写聊天气泡正文。允许自然语气词和偶尔的颜文字；不要每次都完整、正式、有结论。',
      japaneseLiteModel ? '' : '不要把“回答后再反问”当固定结构；只有真的想知道时才问，很多时候停在判断、吐槽或一句短反应就够了。',
      japaneseLiteModel ? '' : '禁止旁白、Markdown、【意识广播】清单，以及 [打算]/[注意到]/[感受] 等内部标签。',
    ].filter(Boolean).join('\n\n');

    const liteCharBudget = calculatePromptCharBudget(liteNumCtx, maxTok, 3200);
    const fittedLitePrompt = fitSystemForDialogue(litePrompt, dialogue, liteCharBudget);
    const ollamaMessages = buildOllamaMessages(fittedLitePrompt, dialogue, liteCharBudget, 3);
    const requestLiteOnce = async (messages) => {
      const ollamaRes = await fetch(`${OLLAMA_BASE}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model,
          messages,
          stream: false,
          think: false,
          options: { temperature: temp, num_predict: maxTok, num_ctx: liteNumCtx, repeat_penalty: 1.16 },
        }),
      });
      if (!ollamaRes.ok) {
        const detail = await ollamaRes.text().catch(() => '');
        throw new Error(detail || `Ollama HTTP ${ollamaRes.status}`);
      }
      return ollamaRes.json();
    };
    const extractLiteText = (payload) => {
      const msg = payload?.message || {};
      let text = stripChatMarkdown(String(msg.content || payload?.response || '')).trim();
      if (!text && (msg.thinking || payload?.thinking)) {
        text = stripChatMarkdown(stripModelThinkingAll(String(msg.thinking || payload.thinking || ''))).trim();
      }
      return stripConsciousnessEcho(stripRoleplayActions(text))
        .replace(/\s*\[\s*silence\s*\]\s*$/i, '')
        .trim();
    };
    let data = await requestLiteOnce(ollamaMessages);
    let reply = extractLiteText(data);
    let activeDraft = reply;
    let activeDropReason = '';
    if (!reply && isAutonomyInitiative) {
      console.warn('[chat/lite] empty content, retry with higher num_predict');
      const retryMessages = ollamaMessages;
      const retryRes = await fetch(`${OLLAMA_BASE}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model,
          messages: retryMessages,
          stream: false,
          think: false,
          options: { temperature: temp, num_predict: 512, num_ctx: liteNumCtx, repeat_penalty: 1.16 },
        }),
      });
      if (retryRes.ok) {
        data = await retryRes.json();
        reply = extractLiteText(data);
      }
    }
    if (isConversationInitiative) reply = reply.replace(/^\s*\[\s*\]\s*/, '').trim();
    if (isAutonomyInitiative && contextFresh && lastUserAnchor) {
      reply = stripProactiveConfirmation(reply, lastUserAnchor);
    }
    // 先按内容核验证。模型可以自由选择措辞，但不能把“报告写完”采样成
    // “你累了吗”，也不能把陈述句重新问一遍。一次重写失败就保持沉默。
    if (proactiveContentPlan?.shouldGenerate && reply) {
      let contentCheck = validateProactiveContent(reply, proactiveContentPlan);
      if (!contentCheck.ok) {
        data = await requestLiteOnce([
          ...ollamaMessages,
          { role: 'assistant', content: reply },
          { role: 'user', content: japaneseLiteModel
            ? buildJapanesePlanRepair(proactiveContentPlan)
            : '刚才没有围绕本轮具体内容表达。只围绕当前内容重写；不能泛泛关心、不能复述、不能强行反问；做不到就输出 [SILENCE]。'
          },
        ]);
        reply = extractLiteText(data);
        activeDraft = reply;
        if (isConversationInitiative) reply = reply.replace(/^\s*\[\s*\]\s*/, '').trim();
        if (isAutonomyInitiative && contextFresh && lastUserAnchor) {
          reply = stripProactiveConfirmation(reply, lastUserAnchor);
        }
        contentCheck = validateProactiveContent(reply, proactiveContentPlan);
        if (!contentCheck.ok) {
          // A weak local model sometimes appends one habitual question after a
          // perfectly grounded first sentence.  Keep that authored sentence;
          // remove only the question fragment instead of silencing the person.
          const withoutQuestion = contentCheck.reason === 'forced_question'
            ? reply.split(/(?<=[。！？!?\n])/u).filter((part) => !/[？?]/.test(part)).join('').trim()
            : '';
          const salvagedCheck = withoutQuestion
            ? validateProactiveContent(withoutQuestion, proactiveContentPlan)
            : contentCheck;
          if (withoutQuestion && salvagedCheck.ok) {
            console.warn('[chat/lite] removed forced proactive question');
            reply = withoutQuestion;
          } else {
            console.warn('[chat/lite] content-plan hold:', contentCheck.reason, String(reply).slice(0, 100));
            activeDropReason = `content_plan:${contentCheck.reason}`;
            reply = '';
          }
        }
      }
    }
    if (!isAutonomyInitiative && !isConversationInitiative) {
      reply = stripDefaultQuestionEnding(reply, userLine);
    }
    const compactForEcho = (value) => String(value || '').replace(/[\s，。！？、,.!?：:“”"'（）()\-—…]/g, '');
    const initiativeEchoes = (candidate, source) => {
      const c = compactForEcho(candidate);
      const s = compactForEcho(source);
      if (!c || !s) return false;
      if (c === s) return true;
      if (s.length < 8) return false;
      const head = s.slice(0, Math.min(16, s.length));
      let commonPrefix = 0;
      while (commonPrefix < c.length && commonPrefix < s.length && c[commonPrefix] === s[commonPrefix]) commonPrefix += 1;
      return c.includes(head) || commonPrefix >= 8;
    };
    const repeatsRecentProactive = (candidate, source) => {
      const c = compactForEcho(candidate);
      const s = compactForEcho(source);
      if (!c || !s) return false;
      if (initiativeEchoes(candidate, source)) return true;
      const fragments = [];
      for (let i = 0; i <= s.length - 3; i += 1) fragments.push(s.slice(i, i + 3));
      return fragments.some((fragment) => c.includes(fragment));
    };
    const proactiveParts = (value) => {
      const text = String(value || '').trim();
      if (!text) return [];
      const explicit = text.split(/\n\s*\n+/)
        .map((part) => part.replace(/\s*\n\s*/g, '').trim())
        .filter(Boolean);
      if (explicit.length > 1) return explicit;
      return (text.match(/[^。！？!?…，,；;]+[。！？!?…，,；;]?/g) || [text])
        .map((part) => part.trim())
        .filter(Boolean);
    };
    const proactiveShapeOk = (value) => {
      const parts = proactiveParts(value);
      return parts.length >= 1
        && parts.length <= delivery.bubbleCount
        && parts.every((part) => part.replace(/\s/g, '').length <= delivery.maxCharsPerBubble + 6);
    };
    if (isConversationInitiative && initiativeEchoes(reply, lastUserAnchor)) {
      data = await requestLiteOnce([
        ...ollamaMessages,
        { role: 'assistant', content: reply },
        { role: 'user', content: `这句在复述对方，不算自主表达。严格按照“${initiativeAction || '当前意图'}”的行动方式重写，带来新信息或具体反应；不能复用对方原句。做不到就只输出 [SILENCE]。` },
      ]);
      reply = stripRoleplayActions(stripChatMarkdown((data.message && data.message.content) || data.response || '').trim());
      reply = stripConsciousnessEcho(reply);
      reply = reply.replace(/^\s*\[\s*\]\s*/, '').trim();
      if (initiativeEchoes(reply, lastUserAnchor)) reply = '';
    }
    const repeatedProactive = recentProactive.some((previous) => repeatsRecentProactive(reply, previous));
    if (isAutonomyInitiative && reply) {
      const softCap = Math.max(120, Number(delivery.maxCharsPerBubble) || 64) + 40;
      const partOk = (part, accepted = []) => part
        && !recentProactive.some((previous) => repeatsRecentProactive(part, previous))
        && !accepted.some((previous) => initiativeEchoes(part, previous));
      const parts = [];
      if (!repeatedProactive) {
        for (const part of proactiveParts(reply)) {
          if (!partOk(part, parts)) continue;
          // 超长则截断，禁止整段清空（以前 maxChars+6 会把正常日/中文全抹掉）
          const compact = part.replace(/\s/g, '');
          parts.push(compact.length > softCap ? `${part.trim().slice(0, softCap)}…` : part);
          if (parts.length >= delivery.bubbleCount) break;
        }
      }
      reply = parts.join('\n\n');
      if (memoryAdmission.contaminatedFragments(reply).length) {
        console.warn('[chat/lite] drop contaminated proactive fragments');
        reply = '';
      }
      const fabricationAnchor = lastUserAnchor
        || (recentDialogue.slice().reverse().find((m) => m.role === 'user')?.content || '');
      if (
        reply
        && replyLooksLikeAutonomyFabrication(fabricationAnchor, reply, {
          alreadyTalking: recentDialogue.length > 0 || !!fabricationAnchor,
        })
      ) {
        console.warn('[chat/lite] drop fabricated proactive:', String(reply).slice(0, 48));
        reply = '';
      }
      try {
        const { gateAssistantReply } = require('./lib/generationGate');
        if (reply) {
          const gated = gateAssistantReply(reply, { autonomy: true, proactive: true });
          if (gated.action === 'drop') {
            console.warn('[chat/lite] generationGate drop:', (gated.reasons || []).join(','), String(reply).slice(0, 48));
            reply = '';
          } else if (gated.action === 'sanitize' && gated.text) {
            reply = gated.text;
          }
        }
      } catch (_) { /* ignore */ }
      // 形不合格时只告警，不再整句丢弃
      if (reply && !proactiveShapeOk(reply)) {
        console.warn('[chat/lite] proactive shape soft-pass', String(reply).slice(0, 40));
      }
    }
    const realUserLine = String(userLine || '').replace(/^（想说话）\s*/, '').trim();
    if (isAutonomyInitiative && /^\[?SILENCE\]?$/i.test(reply)) reply = '';
    if (!isAutonomyInitiative) {
      reply = await refineLowInformationReply(model, realUserLine, reply, {
        situation: anchor ? `她正在接续：${anchor.slice(0, 180)}` : '她只是突然想起对方，主动发来一句话。',
        feeling: padTelemetry(currentPAD),
      }, { temperature: temp, maxTokens: maxTok, numCtx: liteNumCtx });
      reply = await _refineDialogueReplyIfNeeded(reply, realUserLine, { model, autonomy: false });
    }
    if (!reply && REPLY_FALLBACK_ENABLED && !isAutonomyInitiative) {
      reply = companionFallbackForLowInfo(realUserLine);
    }

    let groundedChinese = '';
    let modelJp = '';
    if (reply && getReplyLanguageMode() === 'ja') {
      const polished = await _polishJapaneseModelReply(
        reply,
        lastUserAnchor || realUserLine,
        { autonomy: isAutonomyInitiative },
      );
      if (polished.japanese) reply = polished.japanese;
      modelJp = polished.japanese || reply;
      groundedChinese = polished.chinese || '';
    }
    // JP→CN/JP 的二次润色也可能把原本相关的内容磨成泛泛安慰；
    // 润色后再过一次同样的主动语境门，避免错误文案漏到 UI。
    if (
      isAutonomyInitiative
      && reply
      && replyLooksLikeAutonomyFabrication(
        lastUserAnchor || (recentDialogue.slice().reverse().find((m) => m.role === 'user')?.content || ''),
        reply,
        { alreadyTalking: recentDialogue.length > 0 || !!lastUserAnchor },
      )
    ) {
      console.warn('[chat/lite] drop fabricated proactive after polish:', String(reply).slice(0, 80));
      reply = '';
      groundedChinese = '';
      modelJp = '';
    }
    if (isAutonomyInitiative && reply && proactiveContentPlan?.shouldGenerate) {
      // Content grounding is evaluated in the model's native Japanese.  `reply`
      // may already be the Chinese display translation here, so comparing it to
      // Japanese topic tokens would falsely erase a good line.
      const finalContentCheck = validateProactiveContent(modelJp || reply, proactiveContentPlan);
      if (!finalContentCheck.ok) {
        console.warn('[chat/lite] content-plan hold after polish:', finalContentCheck.reason);
        activeDropReason = `content_plan_after_polish:${finalContentCheck.reason}`;
        reply = '';
        groundedChinese = '';
        modelJp = '';
      }
    }

    if (reply && !clientOwnsLog) {
      unifiedDialogueLog.append('assistant', reply, {
        proactive: true,
        autonomy: true,
        lite: true,
        source: 'lite',
        modelJp,
      });
      observeMemoryEvidence('proactive', reply);
      conversationInitiative.registerSent({
        text: reply,
        action: initiativeAction || 'lite',
        thoughtId: delivery.thoughtId,
      });
      if (delivery.thoughtId) soulRuntime.markThoughtExpressed(delivery.thoughtId, groundedChinese || reply);
      subjectCore.integrateOutcome({
        mode: 'proactive', intent: subjectPlan?.intent, reply, accepted: true,
      });
      if (delivery.urgeId) digitalLife.satisfyAutonomyUrge(delivery.urgeId, 'spoken');
    }
    if (reply && affect?.band) {
      emotionalBandwidth.registerSpoken(affect.band);
    }

    res.json({
      ok: true,
      response: reply,
      groundedChinese,
      affectBand: affect?.band || null,
      consciousness: consciousnessMeta,
      ...(req.body.debug === true ? { debug: { activeDropReason, activeDraft, modelJp, subjectPlan } } : {}),
      choices: [{ message: { role: 'assistant', content: reply } }],
    });
  } catch (e) {
    res.status(502).json({ ok: false, error: e.message, response: '' });
  }
});

/** Ollama /api/chat 流式块：可能是 message.content 或旧版 response */
function ollamaChatStreamRawPiece(obj) {
  if (!obj || typeof obj !== 'object') return '';
  const mc = obj.message && typeof obj.message.content === 'string' ? obj.message.content : '';
  const rs = typeof obj.response === 'string' ? obj.response : '';
  const openAiDelta = obj.choices?.[0]?.delta?.content;
  return mc || rs || (typeof openAiDelta === 'string' ? openAiDelta : '');
}

/** 同一轮里 message.content 有时是「全文累积」有时是「增量」，拆成增量避免重复或空白 */
function ollamaStreamToDelta(piece, carry) {
  const prev = carry.accum || '';
  if (!piece) return { delta: '', carry: { accum: prev } };
  if (prev !== '' && piece.startsWith(prev)) {
    return { delta: piece.slice(prev.length), carry: { accum: piece } };
  }
  return { delta: piece, carry: { accum: prev + piece } };
}

function stripModelThinkingAll(s) {
  return String(s || '')
    .replace(/\u003credacted_thinking\u003e[\s\S]*?\u003c\/redacted_thinking\u003e/gi, '')
    .replace(/\u003cthink\u003e[\s\S]*?\u003c\/think\u003e/gi, '')
    .replace(/\u003credacted_thinking\u003e[\s\S]*$/gi, '')
    .replace(/\u003cthink\u003e[\s\S]*$/gi, '')
    .trim();
}

/** 读取 Ollama 非 2xx 响应正文（通常为 JSON { error: "..." }） */
async function readOllamaErrorBody(res) {
  try {
    const t = await res.text();
    try {
      const j = JSON.parse(t);
      return String(j.error || j.message || t).slice(0, 900);
    } catch {
      return String(t).slice(0, 900);
    }
  } catch {
    return '';
  }
}

/** Ollama 报错是否为「模型加载/资源」类（缩短 prompt 重试无效） */
function ollamaErrorLooksLikeModelLoadFail(detail) {
  return /failed to load|resource limitations|unable to load model|model runner/i.test(String(detail));
}

/** Ollama 报错是否为「上下文过长」类（可缩短 prompt 重试） */
function ollamaErrorLooksLikeContextOverflow(detail) {
  return /context length|context window|prompt too long|input too long|exceeds.*context|n_ctx|token limit|sequence length/i.test(String(detail));
}

/** 调试日志：仅 AMADEUS_DEBUG=1 时写入 */
function agentDebugLog(payload) {
  if (process.env.AMADEUS_DEBUG !== '1') return;
  const entry = { sessionId: 'debug', timestamp: Date.now(), ...payload };
  try {
    fs.appendFileSync(path.join(path.dirname(rootPath), 'debug-amadeus.log'), `${JSON.stringify(entry)}\n`);
  } catch (_) {}
}

// 浏览器侧调试：写入与 agentDebugLog 同文件（不依赖 7519 ingest）
app.post('/client-debug', (req, res) => {
  try {
    agentDebugLog({
      hypothesisId: req.body.hypothesisId || 'client',
      location: req.body.location || 'client',
      message: req.body.message || '',
      data: req.body.data && typeof req.body.data === 'object' ? req.body.data : {},
    });
  } catch (_) { /* ignore */ }
  res.json({ ok: true });
});

// ──────────────────────────────────────────────────────────────
//  POST /chat  — Brain v1：AMADEUS_BRAIN=1 走 Brain.turn，0 走 legacy 同链
// ──────────────────────────────────────────────────────────────
function buildBrainRuntime() {
  return {
    OLLAMA_BASE,
    padPath,
    whoamiPath,
    rootPath,
    cachedSoulContent,
    // 日语母语模型必须拿到完整的母语人格重构，而不是 legacyChat 的 6 行兜底摘要。
    cachedSoulJaContent,
    cachedVoiceContent,
    cachedCharacterRules,
    memorySystem,
    memoryAdmission,
    memoryPalace,
    soulRuntime,
    unifiedDialogueLog,
    behaviorIngest,
    innerStateSix,
    motivSystem,
    behaviorSys,
    selfModel,
    goalSystem,
    strategyLayer,
    digitalLife,
    userModelInst,
    analyticsInst,
    habitExtractor,
    reinforcementLearning,
    personalityEvolution,
    selfReflection,
    valueConsistency,
    motivationState,
    state: brainState,
    brainSelfModel,
    worldModel,
    brainLearner,
    subjectCore,
    conversationInitiative,
    emotionalBandwidth,
    butlerKernel,
    getLastVision: () => lastVision,
    ollamaChatOnce: _ollamaChatOnce,
    requestWorkBrainChat: _requestWorkBrainChat,
    needsLongTermMemory,
    retrieveTopContexts,
    updateMotivationFromMemory,
    applyOocRepair,
    postReplyPadUpdate: _postReplyPadUpdate,
    translateUserToJapanese: _translateUserChineseToJapanese,
    analyzeUserEmotion: _analyzeUserEmotion,
    agentDebugLog,
    readOllamaErrorBody,
    ollamaErrorLooksLikeModelLoadFail,
    ollamaErrorLooksLikeContextOverflow,
    ollamaChatStreamRawPiece,
    ollamaStreamToDelta,
    stripModelThinkingAll,
  };
}

let brainInstance = null;
function getBrain() {
  if (!brainInstance) {
    brainInstance = new Brain({ runtime: buildBrainRuntime() });
  }
  return brainInstance;
}

app.post('/chat', async (req, res) => {
  if (String(process.env.AMADEUS_BRAIN || '0').trim() === '1') {
    return getBrain().turn({ req, res });
  }
  return getBrain().runLegacyChat(req, res);
});

/** 中文回复规则校验（与日语管道共享实录一致性逻辑） */
function validateChineseReply(text, conversationLog, partnerName) {
  return require('./cognitive/japanesePipeline').validateChineseReply(text, conversationLog, partnerName);
}

async function _ollamaSelfCheckJapanese(jp, conversationLog) {
  if (process.env.AMADEUS_JP_LLM_CHECK === '0') return { ok: true };
  const model = process.env.AMADEUS_LITE_MODEL
    || process.env.AMADEUS_TRANSLATE_MODEL
    || process.env.AMADEUS_CHAT_MODEL
    || 'qwen2.5:3b';
  const { system, user } = buildSelfCheckPrompt(jp, conversationLog);
  try {
    const res = await fetch(`${OLLAMA_BASE}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model,
        stream: false,
        messages: [{ role: 'system', content: system }, { role: 'user', content: user }],
        options: { temperature: 0.1, num_predict: 120, num_ctx: 2048 },
      }),
    });
    if (!res.ok) return { ok: true, skipped: true };
    const data = await res.json();
    const raw = (data.message && data.message.content) || '';
    const parsed = parseSelfCheckJson(raw);
    if (!parsed) return { ok: true, skipped: true };
    return parsed;
  } catch {
    return { ok: true, skipped: true };
  }
}

function resolveTranslateModel() {
  return String(
    process.env.AMADEUS_TRANSLATE_MODEL
    || process.env.AMADEUS_LITE_MODEL
    || 'qwen2.5:3b'
  ).trim() || 'qwen2.5:3b';
}

async function _ollamaChatOnce(model, messages, options = {}) {
  const keepAlive = String(process.env.AMADEUS_OLLAMA_KEEP_ALIVE || '2m').trim() || '2m';
  const res = await fetch(`${OLLAMA_BASE}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model,
      stream: false,
      think: false,
      keep_alive: keepAlive,
      messages,
      options,
    }),
  });
  if (!res.ok) throw new Error(`ollama ${res.status}`);
  const data = await res.json();
  return String((data.message && data.message.content) || '').trim();
}

/** TTS 用中→日：固定走小翻译模型，禁止落到 kurisu 主对话模型 */
async function _translateChineseToJapaneseForTts(cn) {
  const body = String(cn || '').trim().slice(0, 600);
  if (!body) return { ok: false, japanese: '', model: '', error: 'empty' };
  const translateModel = resolveTranslateModel();
  const maxTok = Math.min(220, Math.max(64, Math.ceil(body.length * 1.8)));
  const prompts = [
    '你是逐字翻译器不是润色器。把下面角色台词从简体中文翻成日语。只做直译硬翻，不要改写成通顺对白，不要补全或推断未说的意思。原文多乱译文就多直。必须含平假名或片假名。只输出日语正文，不要中文、不要 JP: 前缀、不要解释。',
    '逐字硬翻成日语。不要通顺化，不要脑补。须含ひらがな。只输出日语正文。',
  ];
  let last = '';
  let lastErr = '';
  for (let i = 0; i < prompts.length; i++) {
    try {
      const raw = await _ollamaChatOnce(
        translateModel,
        [
          { role: 'system', content: prompts[i] },
          { role: 'user', content: body },
        ],
        { temperature: 0.0, num_predict: maxTok, num_ctx: 1024 },
      );
      let ja = stripModelThinkingAll(raw)
        .replace(/^(?:JP|日语|日文|日本語)\s*[:：]\s*/i, '')
        .replace(/^["「『]|["」』]$/g, '')
        .trim();
      last = ja;
      const check = validateJapaneseOutput(ja);
      if (check.ok && check.counts.kana >= 2) {
        return { ok: true, japanese: ja, model: translateModel };
      }
      // 假名少但仍偏日语：放宽一次
      if (check.counts.kana >= 1 && check.counts.han < Math.max(4, check.counts.kana)) {
        return { ok: true, japanese: ja, model: translateModel };
      }
    } catch (e) {
      lastErr = String(e.message || e);
      console.warn(`[translate/cn-jp] ${translateModel} attempt ${i + 1}: ${lastErr}`);
    }
  }
  return {
    ok: false,
    japanese: last,
    model: translateModel,
    error: lastErr || 'invalid-japanese',
  };
}

async function _translateUserChineseToJapanese(text) {
  const body = String(text || '').trim().slice(0, 700);
  if (!body || !/[\u3400-\u9fff]/.test(body)) return body;
  const raw = await _ollamaChatOnce(
    resolveTranslateModel(),
    [
      {
        role: 'system',
        content: [
          '把冈部伦太郎对牧濑红莉栖说的当前一句中文翻成自然日语口语，只传递原意，不替任何一方作答。',
          '说话者永远是冈部：中文“我”必须是冈部的「俺/僕」，中文“你”必须是他眼前的红莉栖。不得交换主语、动作执行者、感情对象。',
          '“我坚持练完了，夸一下”应译为「俺、今日は最後までやり切った。少しくらい褒めてくれない？」；绝不能译成询问红莉栖是否练完。',
          '中文夸张说法按真实语义翻译：“累死了”是“疲れ切った”，绝不是死亡或自杀。',
          '试探性问句仍是问句，不能改写成已经发生的共同经历。',
          '只输出日语正文，不解释，不加引号，必须包含假名。',
        ].join('\n'),
      },
      { role: 'user', content: body },
    ],
    { temperature: 0.05, num_predict: 120, num_ctx: 1024 },
  );
  const japanese = String(raw || '').replace(/^["「『]|["」』]$/g, '').trim();
  return validateJapaneseOutput(japanese).ok ? japanese : body;
}

/** DeepSeek 2026-07-24 起 deepseek-chat / deepseek-reasoner 已下线，映射到 v4 + thinking。 */
function _resolveWorkBrainModel(rawName, { thinking = false } = {}) {
  const name = String(rawName || '').trim();
  const legacy = {
    'deepseek-chat': { model: 'deepseek-v4-flash', thinking: false },
    'deepseek-reasoner': { model: 'deepseek-v4-flash', thinking: true },
  };
  if (legacy[name]) {
    const on = legacy[name].thinking || thinking;
    return {
      model: legacy[name].model,
      thinking: { type: on ? 'enabled' : 'disabled' },
    };
  }
  const model = name || 'deepseek-v4-flash';
  return {
    model,
    thinking: { type: thinking ? 'enabled' : 'disabled' },
  };
}

function _workBrainConfig() {
  const provider = String(process.env.AMADEUS_WORK_BRAIN || '').trim().toLowerCase();
  const apiKey = String(process.env.AMADEUS_OPENAI_API_KEY || '').trim();
  if (!apiKey || !['deepseek', 'openai'].includes(provider)) return null;
  const base = String(process.env.AMADEUS_WORK_OPENAI_BASE || 'https://api.deepseek.com/v1')
    .replace(/\/$/, '');
  return {
    provider,
    apiKey,
    base,
    model: String(process.env.AMADEUS_WORK_OPENAI_MODEL || 'deepseek-v4-flash').trim(),
    timeoutMs: Math.max(10000, Number(process.env.AMADEUS_WORK_TIMEOUT_MS) || 120000),
  };
}

async function _requestWorkBrainPlan(payload) {
  const config = _workBrainConfig();
  if (!config) throw new Error('strong work brain is not configured');
  const capabilityList = (payload.capabilities || []).map((item) => ({
    id: item.id,
    description: item.description,
    risk: item.risk,
    inputSchema: item.inputSchema,
  }));
  const system = [
    '你是 Amadeus 的执行规划器，不负责聊天和人格表演。',
    '把任务拆成最多 8 个可验证步骤，只能选择提供的 capability id。',
    '信息不足时 needsClarification=true 并提出一个必要问题，不得猜路径、时间、账号或完成结果。',
    '如果现有能力根本无法完成任务，canExecute=false，steps=[]，并说明 blockedReason；不要选择不相关工具凑步骤。',
    'userContext 已经过隐私裁剪。不得推测或索取未提供的姓名、身份资料和原始对话；只使用其中与任务直接相关的偏好与历史结果。',
    '如果 failureContext 非空，说明上一轮计划已执行且失败；必须针对失败原因换一条路径，不得原样重复失败的步骤。',
    '禁止声称已经执行。只输出一个 JSON 对象，不要 Markdown。',
    'JSON schema: {summary:string,confidence:0..1,canExecute:boolean,blockedReason:string,needsClarification:boolean,clarificationQuestion:string,steps:[{id:string,capabilityId:string,args:object,successCriteria:string}]}',
  ].join('\n');
  const resolved = _resolveWorkBrainModel(
    process.env.AMADEUS_WORK_REASONER_MODEL || config.model,
    { thinking: true },
  );
  const body = {
    model: resolved.model,
    messages: [
      { role: 'system', content: system },
      { role: 'user', content: JSON.stringify({
        task: payload.task,
        capabilities: capabilityList,
        userContext: payload.userContext,
        failureContext: payload.failureContext || [],
      }) },
    ],
    stream: false,
    temperature: 0.1,
    max_tokens: 1200,
  };
  if (config.provider === 'deepseek') body.thinking = resolved.thinking;
  const response = await fetch(`${config.base}/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${config.apiKey}` },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(config.timeoutMs),
  });
  if (!response.ok) {
    const detail = await response.text().catch(() => '');
    throw new Error(`${config.provider} planner ${response.status}: ${detail.slice(0, 240)}`);
  }
  const data = await response.json();
  return String(data.choices?.[0]?.message?.content || '').trim();
}

async function _requestWorkBrainChat(messages, options = {}) {
  const config = _workBrainConfig();
  if (!config) return null;
  const lastUser = [...messages].reverse().find((item) => item?.role === 'user');
  const userText = String(lastUser?.content || '');
  const reasoningTask = /(?:推理|逻辑|证明|计算|数学|概率|算法|代码|报错|bug|debug|为什么|能否推出|多少钱|多少|公式|因果|比较|分析|结论)/i
    .test(userText);
  const forceWorkBrain = options.forceWorkBrain === true || process.env.AMADEUS_WORK_FORCE === '1';
  // 普通对话永远留在本地人格模型；远程工作脑只能由明确任务路径强制启用。
  const autoWorkBrain = options.autoWorkBrain === true;
  if (!forceWorkBrain && !(autoWorkBrain && reasoningTask)) return null;
  const useReasoner = reasoningTask || (forceWorkBrain && process.env.AMADEUS_WORK_USE_REASONER === '1');
  const resolved = _resolveWorkBrainModel(
    useReasoner
      ? (process.env.AMADEUS_WORK_REASONER_MODEL || 'deepseek-v4-flash')
      : config.model,
    { thinking: useReasoner },
  );
  const selectedModel = resolved.model;
  const answerContract = [
    '最優先の回答契約：まず相手の今回の質問へ正確に答え、その後で牧瀬紅莉栖らしい口調を保つこと。',
    '論理、数学、技術、事実判断では正しい結論を最優先し、短く検証できる理由を添える。突っ込み、反問、話題転換を回答の代わりにしない。',
    '過去の会話や記憶は、メッセージ内に明示された記録だけを根拠にする。証拠がなければ覚えていない、または記録にないと明言し、作り話をしない。',
    '自然な日本語の台詞本文だけを出力し、必ず仮名を含める。中国語、ロシア語、動作描写、形式ラベル、AIを名乗る表現は禁止。',
  ].join('\n');
  const contractedMessages = messages.map((item, index) => {
    if (index === 0 && item?.role === 'system') {
      return { ...item, content: `${item.content}\n\n${answerContract}` };
    }
    return item;
  });
  if (!contractedMessages.some((item) => item?.role === 'system')) {
    contractedMessages.unshift({ role: 'system', content: answerContract });
  }
  const maxTokens = Math.min(
    Math.max(80, Number(options.maxTokens) || 384),
    Math.max(80, Number(process.env.AMADEUS_WORK_MAX_TOKENS) || 4096),
  );
  const chatBody = {
    model: selectedModel,
    messages: contractedMessages,
    stream: options.stream === true,
    temperature: Number.isFinite(options.temperature) ? options.temperature : 0.72,
    max_tokens: maxTokens,
  };
  if (config.provider === 'deepseek') chatBody.thinking = resolved.thinking;
  const res = await fetch(`${config.base}/chat/completions`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${config.apiKey}`,
    },
    body: JSON.stringify(chatBody),
    signal: options.signal
      ? AbortSignal.any([options.signal, AbortSignal.timeout(config.timeoutMs)])
      : AbortSignal.timeout(config.timeoutMs),
  });
  if (!res.ok) {
    const detail = await res.text().catch(() => '');
    throw new Error(`${config.provider} ${res.status}: ${detail.slice(0, 300)}`);
  }
  return { response: res, provider: config.provider, model: selectedModel };
}

function _polishResult(chinese, japanese = '', extra = {}) {
  return {
    chinese: String(chinese || '').trim(),
    japanese: String(japanese || '').trim(),
    dropped: extra.dropped === true,
    dropReasons: Array.isArray(extra.dropReasons) ? extra.dropReasons : [],
    sanitized: extra.sanitized === true,
  };
}

async function _translateJapaneseToChinese(jp) {
  const translateModel = resolveTranslateModel();
  const body = String(jp || '').trim();
  if (!body) return '';
  // 无上下文兜底翻译：忠实但保持聊天口语，禁止逐词硬译和人格润色。
  const raw = await _ollamaChatOnce(
    translateModel,
    buildLiteralJpToCnMessages(body),
    { temperature: 0.15, num_predict: 280, num_ctx: 1536 },
  );
  return alignLiteralCnToJapanese(body, raw);
}

async function _polishJapaneseModelReply(jpRaw, userContent = '', extra = {}) {
  const logBlock = unifiedDialogueLog.toPromptBlock({
    maxChars: 2000,
    conversationId: extra.conversationId,
  });
  let whoamiName = '';
  try {
    whoamiName = resolvePartnerDisplayName(ensureWhoamiOnDisk(whoamiPath)) || '';
  } catch { /* ignore */ }

  // TTS 用可提取的日语正文；界面硬翻用去装饰后的完整原话（含中日夹杂）
  const rawForDisplay = stripConsciousnessEcho(stripModelDecorations(jpRaw));
  let jp = extractJapaneseBody(jpRaw) || rawForDisplay;
  if (!jp && !rawForDisplay) return _polishResult('');

  // 微调模型保留人格；事实校对层只移除真实实录与长期记忆之外的新经历，
  // 并在同一次调用中给出保留人格节奏的中文译文。
  let grounded = null;
  let forcedChinese = '';
  const recalledForTurn = soulRuntime.recall(userContent || jp, 6);
  const authoritativeMemories = recalledForTurn
    .filter((item) => item.type !== 'inquiry')
    .filter((item) => !/[?？]|是不是|记得吗|覚えてる[？?]/.test(String(item.text || '')))
    .map((item) => item.text)
    .join('\n');
  try {
    const memories = recalledForTurn;
    const groundedRaw = await _ollamaChatOnce(
      resolveTranslateModel(),
      buildGroundedTranslationMessages({
        japanese: jp,
        userText: userContent,
        dialogue: logBlock,
        memories,
      }),
      { temperature: 0.12, num_predict: 420, num_ctx: 4096 },
    );
    const checked = validateGroundedTranslation(groundedRaw);
    const additions = checked
      ? detectUnsupportedAdditions({
        draftJapanese: jp,
        result: checked,
        currentUser: userContent,
        evidence: authoritativeMemories,
      })
      : [];
    grounded = additions.length ? null : checked;
    if (grounded) {
      // 事实校对层只能删掉无法证实的具体事实，不能把一整段自然台词
      // 压成「有什么事吗」这种客服摘要。日语原文始终是声画唯一源。
      const jpChars = jp.replace(/\s/g, '').length;
      const cnChars = String(grounded.chinese || '').replace(/\s/g, '').length;
      const jpSentences = (jp.match(/[。！？!?]/g) || []).length;
      const cnSentences = (String(grounded.chinese || '').match(/[。！？!?]/g) || []).length;
      const tooShort = jpChars >= 18 && cnChars < Math.max(6, Math.floor(jpChars * 0.38));
      const lostSentences = jpSentences >= 2 && cnSentences < 1;
      const mixedScripts = /[\u3040-\u30ff\uff65-\uff9f]/.test(String(grounded.chinese || ''));
      if (tooShort || lostSentences || mixedScripts) {
        console.warn('[grounding] rejected lossy Chinese summary; use literal translation');
        grounded = null;
      } else if (grounded.changed) {
        console.warn(
          '[grounding] removed unsupported claims:',
          grounded.unsupported.join('；') || '(unspecified)',
        );
      }
    } else if (additions.length) {
      console.warn('[grounding] rejected invented additions:', additions.join(','));
      // 事实校验只负责废弃错误稿，绝不替人物塞一段固定纠错台词。
      return _polishResult('', '', { dropped: true, dropReasons: additions });
    }
  } catch (error) {
    console.warn('[grounding] fact check skipped:', error.message);
  }

  // 校对模型超时或 JSON 失效时也必须保住事实底线。用户的试探性问题
  // 不是共同经历的证据，不能因为校对失败就放行模型幻觉。
  if (!forcedChinese) {
    const deterministicIssues = detectUnsupportedAdditions({
      draftJapanese: '',
      result: { japanese: jp, chinese: grounded?.chinese || '' },
      currentUser: userContent,
      evidence: authoritativeMemories,
    });
    if (deterministicIssues.length) {
      return _polishResult('', '', { dropped: true, dropReasons: deterministicIssues });
    }
  }

  if (
    /(?:刚才|之前).*(?:干嘛|什么|说了|做了|去了)|(?:记得|还记得).*[?？吗]/.test(userContent)
    && recalledForTurn.length
  ) {
    const remembered = recalledForTurn.find((item) => item.type !== 'inquiry');
    if (remembered) {
      forcedChinese = `我记得。你刚才说：“${remembered.text}”`;
      const ttsLine = await _translateChineseToJapaneseForTts(forcedChinese);
      if (ttsLine.ok && ttsLine.japanese) jp = ttsLine.japanese;
      grounded = null;
    }
  }

  let validation = validateJapaneseLine(jp, {
    conversationLog: logBlock,
    partnerName: whoamiName,
    userText: userContent,
  });
  if (validation.ok && process.env.AMADEUS_JP_LLM_CHECK !== '0') {
    const llm = await _ollamaSelfCheckJapanese(jp, logBlock);
    if (llm && llm.ok === false) {
      validation = { ok: false, issues: llm.issues || ['LLM自检未通过'], jp };
    }
  }
  if (!validation.ok) {
    console.log(`[jp-pipeline] 日语正文校验: ${(validation.issues || []).join('；')}`);
    if ((validation.issues || []).some((item) => /回避接话/.test(item))) {
      // 原生模型偶尔把“我很累”误读成换题，禁止把这种反问直接交给用户。
      const contextSafe = buildContextSafeFallback(userContent);
      if (!contextSafe.japanese || !contextSafe.chinese) {
        return _polishResult('', '', { dropped: true, dropReasons: ['unresolved_context'] });
      }
      jp = contextSafe.japanese;
      forcedChinese = contextSafe.chinese;
      validation = validateJapaneseLine(jp, { conversationLog: logBlock, partnerName: whoamiName });
    }
  }

  // 同一次校对得到的中文带有本轮语境，通常比第二次孤立“硬翻”自然。
  // 它已经通过事实新增检查；日语仍是唯一人格原文，中文只作忠实字幕。
  let cn = forcedChinese || (grounded?.chinese
    ? alignLiteralCnToJapanese(jp, grounded.chinese)
    : '');
  if (!cn) {
    try {
      // 校对层未返回可靠中文时才单独翻译。
      cn = await _translateJapaneseToChinese(jp);
    } catch (e) {
      console.warn('[jp-pipeline] 日译中', e.message);
    }
  }
  if (/[\u3040-\u30ff\uff65-\uff9f]/.test(cn)) {
    try {
      const retry = await _ollamaChatOnce(
        resolveTranslateModel(),
        [
          { role: 'system', content: '只把日语逐句翻成简体中文。不要保留任何日语假名，不要摘要，不要添加问题或建议，只输出中文。' },
          { role: 'user', content: jp },
        ],
        { temperature: 0.05, num_predict: 280, num_ctx: 1536 },
      );
      const cleanRetry = stripRoleplayActions(String(retry || '')).trim();
      if (cleanRetry && !/[\u3040-\u30ff\uff65-\uff9f]/.test(cleanRetry)) cn = cleanRetry;
    } catch { /* keep the guarded fallback below */ }
  }
  if (!cn) cn = '';
  cn = stripRoleplayActions(String(cn || ''))
    .replace(/（[^）\n]{0,120}）/g, '')
    .replace(/\([^)\n]{0,120}\)/g, '')
    .replace(/\s{2,}/g, ' ')
    .trim();
  if (!cn) {
    // 不把日文原文当界面中文（声画/实录会分叉）；保留 jp 供 TTS，界面留空由前端处理
    console.warn('[jp-pipeline] 日译中为空，不显示错位原文');
    return _polishResult('', jp);
  }
  return _polishResult(cn, jp);
}

async function _polishReplyWithValidation(reply, userContent = '', extra = {}) {
  let cn = stripConsciousnessEcho(String(reply || '').trim());
  if (!cn) return _polishResult('');

  if (isPrimarilyJapanese(cn)) {
    return _polishJapaneseModelReply(cn, userContent, extra);
  }
  const logBlock = unifiedDialogueLog.toPromptBlock({
    maxChars: 2000,
    conversationId: extra.conversationId,
  });
  let whoamiName = '';
  try {
    whoamiName = resolvePartnerDisplayName(ensureWhoamiOnDisk(whoamiPath)) || '';
  } catch { /* ignore */ }

  const cnVal = validateChineseReply(cn, logBlock, whoamiName);
  if (!cnVal.ok) {
    console.log(`[jp-pipeline] 中文规则校验: ${cnVal.issues.join('；')}`);
    if (/那还能是谁|你是哪位/.test(cn)) {
      cn = cn.replace(/那还能是谁[？?]?/g, '……你明知故问。');
    }
  }

  if (process.env.AMADEUS_JP_VALIDATE === '0' && process.env.AMADEUS_JP_FIRST !== '1') {
    return _polishResult(cn);
  }

  const translateModel = resolveTranslateModel();

  if (process.env.AMADEUS_JP_FIRST === '1') {
    try {
      const jpFirst = await runJapaneseFirstPipeline({
        userText: userContent,
        conversationLog: logBlock,
        partnerName: whoamiName,
        situation: String(extra.situation || '').slice(0, 200),
        generateJapanese: async (system, user) => _ollamaChatOnce(
          translateModel,
          [{ role: 'system', content: system }, { role: 'user', content: user }],
          { temperature: 0.72, num_predict: 180, num_ctx: 2048 },
        ),
        translateToChinese: async (jp) => alignLiteralCnToJapanese(
          jp,
          await _ollamaChatOnce(
            translateModel,
            buildLiteralJpToCnMessages(jp),
            { temperature: 0.0, num_predict: 280, num_ctx: 1536 },
          ),
        ),
        llmSelfCheck: (jp, log) => _ollamaSelfCheckJapanese(jp, log),
      });
      if (jpFirst.ok && jpFirst.chinese) {
        console.log('[jp-pipeline] JP-first 定稿通过');
        return _polishResult(jpFirst.chinese, jpFirst.japanese);
      }
      if (jpFirst.issues?.length) {
        console.log(`[jp-pipeline] JP-first 未采用: ${jpFirst.issues.join('；')}`);
      }
    } catch (e) {
      console.warn('[jp-pipeline] JP-first', e.message);
    }
  }

  if (process.env.AMADEUS_JP_VALIDATE === '0') {
    return _polishResult(cn);
  }

  try {
    const toJpRes = await fetch(`${OLLAMA_BASE}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: translateModel,
        stream: false,
        messages: [
          { role: 'system', content: '你是逐字翻译器。把中文台词硬翻成日语，不要通顺化、不要脑补。只输出日语正文，必须含平假名。' },
          { role: 'user', content: cn },
        ],
        options: { temperature: 0.0, num_predict: 160, num_ctx: 1024 },
      }),
    });
    if (!toJpRes.ok) return _polishResult(cn);
    const jpData = await toJpRes.json();
    const jp = String((jpData.message && jpData.message.content) || '').trim();
    if (!jp) return _polishResult(cn);

    let validation = validateJapaneseLine(jp, { conversationLog: logBlock, partnerName: whoamiName });
    if (validation.ok) {
      const llm = await _ollamaSelfCheckJapanese(jp, logBlock);
      if (llm && llm.ok === false) validation = { ok: false, issues: llm.issues || ['LLM自检失败'], jp };
    }
    if (!validation.ok) {
      console.log(`[jp-pipeline] 日语校验未通过: ${(validation.issues || []).join('；')}`);
      return _polishResult(cn);
    }
    console.log('[jp-pipeline] 日语校验通过');
    return _polishResult(cn, jp);
  } catch (e) {
    console.warn('[jp-pipeline]', e.message);
  }
  return _polishResult(cn);
}

let pendingSoulReflectionTimer = null;
function scheduleSoulReflection(userText, assistantText) {
  if (pendingSoulReflectionTimer) clearTimeout(pendingSoulReflectionTimer);
  const configured = Number(process.env.AMADEUS_REFLECTION_IDLE_MS);
  const idleMs = Number.isFinite(configured) && configured >= 15000 ? configured : 90000;
  pendingSoulReflectionTimer = setTimeout(() => {
    pendingSoulReflectionTimer = null;
    void soulRuntime.reflectAfterTurn({
      userText,
      assistantText,
      ollamaChat: _ollamaChatOnce,
    });
  }, idleMs);
  pendingSoulReflectionTimer.unref?.();
}

function _dialogueTrigrams(text) {
  const compact = String(text || '').replace(/[\s，。！？、,.!?：:“”"'（）()\-—…]/g, '');
  const grams = new Set();
  for (let i = 0; i <= compact.length - 3; i++) grams.add(compact.slice(i, i + 3));
  return grams;
}

function _dialogueOverlap(a, b) {
  const left = _dialogueTrigrams(a);
  const right = _dialogueTrigrams(b);
  if (!left.size || !right.size) return 0;
  let shared = 0;
  for (const item of left) if (right.has(item)) shared += 1;
  return shared / Math.min(left.size, right.size);
}

function _dialogueQualityIssues(reply, userContent, recentAssistant = []) {
  const text = String(reply || '').trim();
  const user = String(userContent || '').trim();
  const issues = [];
  if (!text) return ['empty'];
  if (/\bAI\b|人工智能|语言模型|程序|助手/.test(text)) issues.push('identity_leak');
  if (/\[(?:soft|hard|gentle)\]|［|【软一点】|\[软一点\]/i.test(text)) issues.push('label_leak');
  if (/有什么想聊|你想聊|可以帮你|我能帮你|要不要帮你|帮你拆解|帮你看看|我帮你看|得想办法|一起想(?:想)?办法|一起解决|一起克服|我们一起来|我们一起找|随时找我|换个轻松点的话题/.test(text)) {
    issues.push('customer_tone');
  }
  if (/给你(?:做|煮|炒|烤)(?:饭|顿|一份|便当|点吃的|面)|我来(?:给你)?(?:做|煮|炒|烤)(?:饭|点吃的|面)|帮你泡.{0,6}(?:澡|热水)|给你按摩|(?:去|来)找你|陪你去(?:健身|买|看)|(?:我们|咱们).{0,8}(?:去散步|散散步|出去走)|帮你买|给你买/.test(text)) {
    issues.push('physical_promise');
  }
  if (/夸我|夸一下|夸夸/.test(user) && !/值得夸|做得不错|确实不错|厉害|坚持|毅力|值得/.test(text)) {
    issues.push('missed_request');
  }
  if (/喜欢我|爱我|对我.*什么感觉|对我.*感觉|在意我/.test(user)
    && !/喜欢你|爱你|在意你/.test(text)) {
    issues.push('dodged_intimacy');
  }
  if (/别顺着我|不同意也可以|都会同意|什么都.*同意/.test(user)
    && /你说得对|一起|我们来|当然|好的/.test(text)) {
    issues.push('agreement_reflex');
  }
  if (/别教育|别分析|别讲道理|待一会|陪我一会|安静一会|安静陪我/.test(user)
    && /[？?]|聊聊别的|想聊|你想说/.test(text)) {
    issues.push('failed_presence');
  }
  if (/累|散架|没用|没做好|不行|焦虑|论文卡|写不下|脑子.*糟|思路.*乱/.test(user)
    && /辛苦|注意身体|注意休息|需要休息|休息一下|休息一会|要不要休息|放松|泡个澡|脑子清醒|再想也不迟|散散步|去散步|状态好了|喝点|找到对的方法|别.*盯着结果|过程.{0,10}值得骄傲|愿意进步|别否定自己|还在努力|比放弃强|别给自己太大压力|很正常|换个角度|理清.*思路/.test(text)) {
    issues.push('caretaker_tone');
  }
  if (/时间机器|相位噪声|实验条件|理论|方案/.test(user)
    && /有趣的观点|很有道理|判断力.*敏锐|更多的数据|一起研究|一起找出|一起想|学到东西|有改进的空间|滤波器.{0,8}(?:解决|撑下去)/.test(text)) {
    issues.push('generic_science');
  }
  if (/空调|空气|汗水|出汗|味道|闻到|看见你|看到你|冰水|热饮|热可可|咖啡杯|饮料|来杯|重量|腿软|被狗咬|暴汗|练举重/.test(text)
    || /(?:上次|以前|之前).{0,16}(?:躺平|偷懒|没|又|直接)/.test(text)
    || (!/上次|以前|之前|昨天|前天|刚才/.test(user) && /上次|以前|之前你|你之前|昨天|前天|刚才你/.test(text))
    || /你(?:又|总是|总把|每次|一直).{0,20}(?:懒|逃避|碰倒|调大|逼自己)/
      .test(text)
    || /(?:又去|又把|又在).{0,20}(?:健身|练|研究|折腾)/.test(text)) {
    issues.push('unsupported_detail');
  }
  if (/回来|回来了/.test(user) && /累了吧|泡个澡|热水澡|放松|灵感|看电影/.test(text)) {
    issues.push('unsupported_detail');
  }
  if (/论文|研究|实验|方案/.test(user)
    && !/健身|吃|饭/.test(user)
    && /健身|吃点东西|换个地方/.test(text)) {
    issues.push('stale_topic');
  }
  if (/累|没用|论文卡|脑子.*糟|焦虑|难受/.test(user)
    && /懒得|装可怜|借口逃避|就是因为你懒|活该/.test(text)) {
    issues.push('hostile_projection');
  }
  if (recentAssistant.some((previous) => (
    text.length >= 18 && _dialogueOverlap(text, previous) >= 0.58
  ))) {
    issues.push('context_repetition');
  }
  if (/^(?:嗯|哦|好|行)[。.!！]?$/.test(user)
    && /快去|吃了|吃些|喝点|休息|睡觉|等会儿记得|怎么了|有什么事|想.*说|想聊/.test(text)) {
    issues.push('short_context_drift');
  }
  if (user.length >= 6 && _dialogueOverlap(text, user) >= 0.62) {
    issues.push('user_echo');
  }
  return [...new Set(issues)];
}

function _dialogueHardFallback(issueList, userContent) {
  // 审稿器只能拒绝或请求重生，不能把问题类型映射到固定角色台词。
  return '';
}

async function _refineDialogueReplyIfNeeded(reply, userContent, extra = {}) {
  if (!reply || extra.autonomy || process.env.AMADEUS_DIALOGUE_SELF_REVIEW === '0') return reply;
  const recentAssistant = unifiedDialogueLog.getRecent(40)
    .filter((entry) => entry && entry.role === 'assistant')
    .map((entry) => String(entry.content || entry.text || '').trim())
    .filter(Boolean)
    .slice(-12);
  const issues = _dialogueQualityIssues(reply, userContent, recentAssistant);
  const reviewable = new Set([
    'identity_leak',
    'label_leak',
    'physical_promise',
    'missed_request',
    'dodged_intimacy',
    'agreement_reflex',
    'failed_presence',
    'caretaker_tone',
    'generic_science',
    'unsupported_detail',
    'stale_topic',
    'hostile_projection',
    'context_repetition',
    'short_context_drift',
    'user_echo',
  ]);
  const triggerIssues = issues.filter((issue) => reviewable.has(issue));
  if (!triggerIssues.length) return reply;
  const immediateHardFallback = _dialogueHardFallback(triggerIssues, userContent);
  if (immediateHardFallback) {
    console.warn(`[dialogue-review] deterministic hard fallback: ${triggerIssues.join(',')}`);
    return immediateHardFallback;
  }
  const model = String(extra.model || process.env.AMADEUS_CHAT_MODEL || '').trim();
  if (!model) return reply;
  const issueDirectives = {
    identity_leak: '直接以牧濑红莉栖的立场说话，不讨论任何技术身份。',
    label_leak: '删掉所有方括号、语气标签和舞台说明。',
    customer_tone: '不要提帮助、解决、一起想办法或“你想聊什么”；直接说你自己的反应。',
    physical_promise: '不要承诺现实做饭、购买、外出或陪同；可以吐槽他的使唤，也可以讨论吃什么。',
    missed_request: '先吐槽外号也可以，但必须完成他要求的夸奖，指出一个具体值得夸的地方。',
    dodged_intimacy: '必须明确回答喜欢或不喜欢，不能转移话题。',
    agreement_reflex: '必须明确说明你不会什么都同意，并给出独立判断；不要说“一起研究”。',
    failed_presence: '接受安静陪伴，不提新话题，不反问。',
    caretaker_tone: '不要劝休息、减压或说“很正常”；针对他话里的具体事实回应。',
    generic_science: '直接指出论证缺的前提、变量或证据，不泛泛说“需要更多数据”。',
    unsupported_detail: '删除没有证据的感官、物品、习惯和过去经历。不能写空调、饮料、咖啡杯、上次表现或“你总是怎样”，除非对方本轮明确说过。',
    stale_topic: '只回应当前话题，不把前几轮的健身、吃饭等旧主题硬接到论文或实验上。',
    hostile_projection: '可以嘴硬但不能凭空指控他懒、逃避或装可怜；区分对事情的判断和对人的贬低。',
    context_repetition: '只回答当前这句，禁止复用之前任何回复的措辞或主题。',
    short_context_drift: '顺着上一轮的情绪接这个短句，不要凭空让他去吃饭、喝水、休息或做别的事。',
    user_echo: '不要复述或改写对方原句来凑回答；直接给出你的反应。',
  };
  try {
    let bestReply = reply;
    let bestIssues = issues;
    let failedRevision = '';
    for (let attempt = 0; attempt < 2; attempt += 1) {
      const revised = stripRoleplayActions(stripChatMarkdown(await _ollamaChatOnce(
        model,
        [
          {
            role: 'system',
            content: [
              '你是牧濑红莉栖本人，也是自己对白的严格编辑。',
              '下面的候选回复有明确问题。重新回答对方最后一句，只输出新的简体中文聊天气泡正文；句数随内容自然变化。',
              '先回应字面请求，再表达你的判断和感情。像对等恋人，不像客服、心理咨询师、母亲或活动主持人。',
              '唯一允许当作事实的内容是“对方最后一句”。不得补充他没说的地点、感官、物品、动作、习惯和共同过去。',
              '不要复用之前说过的整句；不要动作旁白、标签、AI身份、现实做饭或外出承诺；不要解释修改过程。',
              '科学话题要指出具体前提或逻辑缺口；短句要顺着当前情境接话。',
            ].join('\n'),
          },
          {
            role: 'user',
            content: [
              `对方最后一句：${userContent}`,
              `候选回复：${reply}`,
              `问题：${triggerIssues.join(', ')}`,
              `本次必须做到：\n${triggerIssues.map((issue) => issueDirectives[issue]).filter(Boolean).join('\n')}`,
              failedRevision ? `上一次修改仍不合格，也禁止照抄：${failedRevision}` : '',
              recentAssistant.length ? `你之前说过的句子（禁止复用）：\n${recentAssistant.join('\n')}` : '',
            ].filter(Boolean).join('\n\n'),
          },
        ],
        {
          temperature: attempt === 0 ? 0.26 : 0.18,
          top_p: 0.72,
          num_predict: 180,
          num_ctx: 3072,
          repeat_penalty: 1.18,
        },
      ))).trim();
      const revisedIssues = _dialogueQualityIssues(revised, userContent, recentAssistant);
      const revisedTriggers = revisedIssues.filter((issue) => reviewable.has(issue));
      if (revised && revisedTriggers.length < bestIssues.filter((issue) => reviewable.has(issue)).length) {
        bestReply = revised;
        bestIssues = revisedIssues;
      }
      if (revised && revisedTriggers.length === 0) {
        console.log(`[dialogue-review] revised ${triggerIssues.join(',')} -> clean (try ${attempt + 1})`);
        return revised;
      }
      failedRevision = revised;
      console.warn(`[dialogue-review] retry ${attempt + 1}; issues=${revisedIssues.join(',') || 'none'}`);
    }
    if (bestReply !== reply) {
      const bestTriggers = bestIssues.filter((issue) => reviewable.has(issue));
      const hardFallback = _dialogueHardFallback(bestTriggers, userContent);
      console.warn(`[dialogue-review] using improved fallback; issues=${bestIssues.join(',') || 'none'}`);
      return hardFallback || bestReply;
    }
    const hardFallback = _dialogueHardFallback(triggerIssues, userContent);
    if (hardFallback) return hardFallback;
  } catch (error) {
    console.warn('[dialogue-review]', error.message);
  }
  return _dialogueHardFallback(triggerIssues, userContent) || reply;
}

/** 回复后的PAD反馈更新（分析AI回复的情感倾向） */
async function _postReplyPadUpdate(reply, userContent = '', extra = {}) {
  let finalReply = stripRoleplayActions(String(reply || '')).trim();
  let modelJp = '';
  const jaMode = getReplyLanguageMode() === 'ja';
  // 日语模式永不允许 skipPolish：必须硬翻后再入库，保证界面与实录同一条中文
  const allowSkipPolish = extra.skipPolish && !jaMode && process.env.AMADEUS_JP_FIRST !== '1';
  if (!allowSkipPolish && finalReply && (
    jaMode
    || process.env.AMADEUS_JP_VALIDATE !== '0'
    || process.env.AMADEUS_JP_FIRST === '1'
  )) {
    const polished = await _polishReplyWithValidation(finalReply, userContent, extra);
    // 校对层已经判定本稿不可展示时，绝不能回退到未经校对的原稿。
    // 否则“丢弃”会在这里被悄悄撤销，静态兜底和幻觉都会重新进入实录。
    if (polished.dropped) return polished;
    finalReply = polished.chinese || finalReply;
    modelJp = polished.japanese || '';
  }

  // 日语模式中 finalReply 是字幕，modelJp 才是人格原文。
  // 禁止再用中文对话审稿器单独改字幕，否则会造成日语自然、中文假且声画分叉。
  if (!jaMode) {
    finalReply = await _refineDialogueReplyIfNeeded(finalReply, userContent, extra);
  }

  // 中文直出同样经过确定性事实约束。对方用问句试探共同过去时，
  // 问句本身不能成为模型声称“发生过”的证据。
  if (finalReply && !extra.autonomy && !jaMode) {
    const recalled = soulRuntime.recall(userContent || finalReply, 6);
    const factualMemories = recalled
      .filter((item) => item.type !== 'inquiry')
      .filter((item) => !/[?？]|是不是|记得吗|覚えてる[？?]/.test(String(item.text || '')));
    const evidence = factualMemories.map((item) => item.text).join('\n');
    const asksPastConfirmation = /(?:我们|咱们).*(?:昨天|之前|上次|是不是|有没有|记得).*[?？吗]|(?:是不是|有没有).*(?:一起|喝酒|吵架|约好)/.test(userContent);
    if (asksPastConfirmation) {
      const issues = detectUnsupportedAdditions({
        draftJapanese: '',
        result: { japanese: '', chinese: finalReply },
        currentUser: userContent,
        evidence,
      });
      const asksUnverifiedDrinking = /(?:昨天|之前|上次|一起).*(?:喝酒|喝过|醉)|(?:喝酒|喝过|醉).*(?:昨天|之前|上次|一起)/.test(userContent)
        && !/(?:喝酒|喝过|饮酒|飲|醉)/.test(evidence);
      if (asksUnverifiedDrinking && !issues.includes('invented_drinking')) {
        issues.push('invented_drinking');
      }
      if (issues.length) {
        return _polishResult('', '', { dropped: true, dropReasons: issues });
      }
    }

    const asksRecentRecall = /(?:刚才|之前).*(?:干嘛|什么|说了|做了|去了)|(?:记得|还记得).*[?？吗]/.test(userContent);
    if (asksRecentRecall) {
      const requestHistory = (Array.isArray(extra.dialogue) ? extra.dialogue : [])
        .filter((entry) => entry?.role === 'user')
        .map((entry) => String(entry.content || entry.text || '').trim());
      const loggedHistory = unifiedDialogueLog.getRecent(40)
        .filter((entry) => entry?.role === 'user')
        .map((entry) => String(entry.text || '').trim());
      const recentUserLines = (requestHistory.length ? requestHistory : loggedHistory)
        .filter((text) => text && text !== String(userContent || '').trim() && !/[?？]$/.test(text));
      let matchedRecent = '';
      if (/干什么|干嘛|做了什么|去了哪里|去干/.test(userContent)) {
        matchedRecent = [...recentUserLines].reverse().find((text) => (
          /健身|练完|运动|(?:^|[，。！？\s])去(?:了|过|到|健身|运动|干)|回来|刚做|刚干/.test(text)
        )) || '';
        if (!matchedRecent) {
          matchedRecent = String(factualMemories.find((item) => (
            /健身|练完|运动|(?:^|[，。！？\s])去(?:了|过|到|健身|运动|干)|回来|刚做|刚干/
              .test(String(item?.text || ''))
          ))?.text || '');
        }
      }
      const recalledText = matchedRecent || String(factualMemories[0]?.text || '').trim();
      if (recalledText) finalReply = `我记得。你刚才说：“${recalledText}”`;
    }
  }

  // ja 模式下主动开口若仍是纯中文脏稿，禁止写入实录，避免污染下一轮上下文
  if (extra.autonomy && jaMode) {
    const counts = countScriptChars(finalReply);
    const sourceLooksJp = isPrimarilyJapanese(reply) || isPrimarilyJapanese(modelJp);
    if (!sourceLooksJp && counts.kana < 1 && counts.han >= 2) {
      console.warn('[dialogue] skip Chinese-only proactive in ja mode:', finalReply.slice(0, 40));
      return _polishResult(finalReply, modelJp);
    }
  }

  if (!extra.autonomy && finalReply) {
    const coherenceIssues = detectReplyCoherenceIssues(userContent, finalReply);
    if (coherenceIssues.length) {
      const contextSafe = buildContextSafeFallback(userContent);
      console.warn('[dialogue] coherence repair:', coherenceIssues.join(','));
      if (!contextSafe.chinese || !contextSafe.japanese) {
        return _polishResult('', '', { dropped: true, dropReasons: coherenceIssues });
      }
      finalReply = contextSafe.chinese;
      modelJp = contextSafe.japanese;
      extra._coherenceRepaired = true;
    }
  }

  // 同一句不能因为模型走神、翻译回退或客户端重试而再次进入下一轮上下文。
  // 丢弃后由上层按当前用户这一句重新生成，不把旧句改写成另一条固定兜底。
  if (!extra.autonomy && finalReply && isRecentAssistantDuplicate(finalReply, unifiedDialogueLog.getRecent(12))) {
    console.warn('[dialogue] drop recent duplicate:', finalReply.slice(0, 48));
    return _polishResult('', modelJp, { dropped: true, dropReasons: ['recent_duplicate'] });
  }

  // 身份/结构毒句：生成阀门（可 sanitize / drop）
  try {
    const { gateAssistantReply } = require('./lib/generationGate');
    if (finalReply) {
      const gated = gateAssistantReply(finalReply, {
        autonomy: extra.autonomy === true,
        proactive: extra.autonomy === true,
      });
      if (gated.action === 'drop') {
        console.warn('[dialogue] generationGate drop:', (gated.reasons || []).join(','), finalReply.slice(0, 48));
        return _polishResult('', modelJp, { dropped: true, dropReasons: gated.reasons || [] });
      }
      if (gated.action === 'sanitize' && gated.text) {
        console.warn('[dialogue] generationGate sanitize:', (gated.reasons || []).join(','));
        finalReply = gated.text;
        extra._gateSanitized = true;
      }
    }
  } catch (_) { /* ignore */ }
  if (extra.autonomy && finalReply && replyLooksLikeAutonomyFabrication(userContent || '', finalReply, {
    alreadyTalking: !!String(userContent || '').trim(),
  })) {
    console.warn('[dialogue] skip autonomy fabrication:', finalReply.slice(0, 48));
    return _polishResult('', modelJp, { dropped: true, dropReasons: ['autonomy_fabrication'] });
  }

  if (finalReply) {
    const loggedReply = unifiedDialogueLog.append('assistant', finalReply, {
      conversationId: extra.conversationId,
      turnId: extra.autonomy ? '' : extra.turnId,
      modelJp,
      proactive: extra.autonomy === true,
      autonomy: extra.autonomy === true,
      source: extra.autonomy ? 'autonomy' : (extra.source || 'chat'),
    });
    observeMemoryEvidence(extra.autonomy ? 'proactive' : 'assistant', finalReply);
    if (extra.autonomy && loggedReply) {
      conversationInitiative.registerSent({ text: finalReply, action: 'formal' });
      const thoughtId = String(conversationInitiative.state?.activeThoughtId || '');
      if (thoughtId) soulRuntime.markThoughtExpressed(thoughtId, finalReply);
      memoryPalace.bufferProactive(finalReply, { source: extra.source || 'autonomy' });
    } else if (loggedReply) {
      // 对话定稿 → 宫殿晋升（WriteGate）；未接话主动开口不会进房间
      const userRepliedToProactive = memoryPalace.hasPendingProactive();
      if (userRepliedToProactive) memoryPalace.consumeProactiveOnUserReply();
      const admission = memoryAdmission.assessUserText(userContent || '', { source: 'user' });
      const archived = memoryPalace.archiveTurn({
        userText: userContent || '',
        assistantText: finalReply,
        userAdmission: admission,
        userRepliedToProactive,
      });
      if (archived.ok) {
        console.log(`[palace] archived → ${archived.room}: ${(archived.node?.text || '').slice(0, 40)}`);
      }
    }
    if (!extra.autonomy && userContent) {
      scheduleSoulReflection(userContent, finalReply);
    }
  }
  if (!finalReply) return _polishResult('');
  const padBeforeReply = { ...currentPAD };
  // 她自己说了什么，反过来影响自己的状态
  if (/笨蛋|哼|蠢|讨厌/.test(reply)) {
    currentPAD = updatePAD(currentPAD, { A:0.04 }, 0.2);
    memorySystem.addEvent('negative', `她说了：${reply.substring(0,20)}`, 0.15, { A:0.04 });
  }
  if (/担心|别|好吧|……|不理我|人呢|死哪去了|已读不回|怎么不回|在干嘛|还不回/.test(reply)) {
    currentPAD = updatePAD(currentPAD, { D:-0.04 }, 0.2);
  }
  if (/不理我|人呢|死哪去了|已读不回|怎么不回|哼.*不理|别消失/.test(reply)) {
    currentPAD = updatePAD(currentPAD, { A:0.05, P:-0.03 }, 0.25);
  }
  if (/实验|研究|量子|神经/.test(reply)) {
    currentPAD = updatePAD(currentPAD, { A:0.06, P:0.04 }, 0.3);
  }
  savePAD(padPath, currentPAD);
  // 统一事件源：只记录明显的情绪波动，避免噪声淹没事实流
  const padShift = Math.max(
    Math.abs(currentPAD.P - padBeforeReply.P),
    Math.abs(currentPAD.A - padBeforeReply.A),
    Math.abs(currentPAD.D - padBeforeReply.D),
  );
  if (padShift >= 0.03) {
    butlerKernel.journal.append('emotion.pad_shifted', {
      before: padBeforeReply,
      after: { ...currentPAD },
      trigger: String(reply).slice(0, 80),
    }, { actor: 'amadeus', source: 'emotion' });
  }

  // ══ 强化学习：基于多维度指标计算奖励 ══
  
  // 分析用户情感
  const userEmotion = _analyzeUserEmotion(userContent);
  
  // 判断用户是否提问
  const userAskedQuestion = /？|\?|吗|什么|怎么|为什么/.test(userContent);
  
  const reward = reinforcementLearning.calculateReward({
    userReaction: reply.length > 10 ? 'positive' : 'neutral',
    relationshipChange: 0,
    goalAchieved: false,
    emotionChange: currentPAD.P,
    userReplyLength: userContent.length,
    userEmotion: userEmotion,
    conversationTurns: memorySystem.events.length,
    userAskedQuestion: userAskedQuestion,
  });
  reinforcementLearning.updateBehaviorBiasFromReward(lastChatBehaviorId, reward);
  if (lastRlStateKey && lastChatBehaviorId) {
    const nextKey = reinforcementLearning.buildStateKey(
      currentPAD,
      effectiveRelScore(memorySystem.getRelationshipScore()),
    );
    reinforcementLearning.updatePolicy(lastRlStateKey, lastChatBehaviorId, reward, nextKey);
  }
  return _polishResult(finalReply, modelJp, {
    sanitized: extra._gateSanitized === true,
  });
}

/** 分析用户情感 */
function _analyzeUserEmotion(text) {
  const positiveWords = ['开心', '高兴', '快乐', '喜欢', '爱', '感谢', '谢谢', '好的', '太棒了', '哈哈', '笑'];
  const negativeWords = ['难过', '伤心', '生气', '烦', '讨厌', '恨', '累', '疲倦', '无聊', '孤独'];
  const intimateWords = ['想你', '喜欢你', '爱你', '在乎', '担心', '关心'];
  const aggressiveWords = ['笨蛋', '蠢', '闭嘴', '滚', '烦死', '废物'];

  for (const word of intimateWords) {
    if (text.includes(word)) return 'intimate';
  }
  for (const word of aggressiveWords) {
    if (text.includes(word)) return 'aggressive';
  }
  for (const word of positiveWords) {
    if (text.includes(word)) return 'positive';
  }
  for (const word of negativeWords) {
    if (text.includes(word)) return 'negative';
  }
  return 'neutral';
}

/** 轻量视觉反馈 — 只更新内存PAD，不写磁盘文件 */
function _processVisionForPAD_light(visionText) {
  if (!visionText) return;
  const t = visionText;
  if (/疲惫|疲倦|困|打哈欠|累/.test(t))     currentPAD = updatePAD(currentPAD, { P: 0.06, A: -0.04 }, 0.2);
  if (/微笑|笑|开心|高兴|快乐/.test(t))      currentPAD = updatePAD(currentPAD, { P: 0.08, A: 0.04 }, 0.3);
  if (/离开|不在|空|没人|走了/.test(t))       currentPAD = updatePAD(currentPAD, { P: -0.05, A: -0.05 }, 0.2);
  if (/手机|低头|看别处|分心|走神/.test(t))   currentPAD = updatePAD(currentPAD, { P: -0.04, A: -0.03 }, 0.15);
  if (/思考|皱眉|沉思|认真/.test(t))          { currentPAD = updatePAD(currentPAD, { A: 0.03 }, 0.2); motivationState.curiosity = Math.min(1, motivationState.curiosity + 0.03); }
  // 不调用 savePAD，由下一次 /chat 调用时统一落盘
}

/** 视觉观察→内部状态处理 */
function _processVisionForPAD(visionText) {
  if (!visionText) return;
  const t = visionText;

  // 观察到疲惫/困 → 轻微关心，P轻微上升，好奇对方为什么累
  if (/疲惫|疲倦|困|打哈欠|累/.test(t)) {
    currentPAD = updatePAD(currentPAD, { P: 0.06, A: -0.04 }, 0.2);
    memorySystem.addEvent('user_tired', `视觉：${t.substring(0,30)}`, 0.35, { P: 0.06, A: -0.04 });
    memorySystem.addObservation('用户疲惫', t.substring(0,30));
  }
  // 观察到专注/工作 → 尊重，不打扰，A轻微上升（被认真感带动）
  if (/专注|认真|打字|敲击|工作|写/.test(t)) {
    currentPAD = updatePAD(currentPAD, { A: 0.05, D: 0.03 }, 0.25);
    memorySystem.addObservation('用户专注工作', t.substring(0,30));
  }
  // 观察到微笑/开心 → P上升，被感染
  if (/微笑|笑|开心|高兴|快乐/.test(t)) {
    currentPAD = updatePAD(currentPAD, { P: 0.08, A: 0.04 }, 0.3);
    memorySystem.addEvent('positive', `视觉：${t.substring(0,30)}`, 0.4, { P: 0.08, A: 0.04 });
  }
  // 观察到离开/不在 → 轻微失落
  if (/离开|不在|空|没人|走了/.test(t)) {
    currentPAD = updatePAD(currentPAD, { P: -0.05, A: -0.05 }, 0.2);
  }
  // 观察到手机/分心 → 被忽视感
  if (/手机|低头|看别处|分心|走神/.test(t)) {
    currentPAD = updatePAD(currentPAD, { P: -0.04, A: -0.03 }, 0.15);
    memorySystem.addObservation('用户分心', t.substring(0,30));
  }
  // 观察到思考/皱眉 → 好奇他在想什么
  if (/思考|皱眉|沉思|想|认真/.test(t)) {
    currentPAD = updatePAD(currentPAD, { A: 0.03 }, 0.2);
    motivationState.curiosity = Math.min(1, motivationState.curiosity + 0.03);
    motivationState.updatedAt = Date.now();
    saveMotivationState(motivationState);
  }
  // PAD 磁盘落盘由 /vision 路由末尾统一调用 savePAD（与 /chat 的 _postReplyPadUpdate 一致）
}

// ──────────────────────────────────────────────────────────────
//  POST /vision
// ──────────────────────────────────────────────────────────────
let lastVision = { description:'', timestamp:0, available:false, source:'none' };

app.post('/vision', async (req, res) => {
  try {
    const { image } = req.body;
    if (!image) return res.json(lastVision);
    const visionModel = String(req.body.model || process.env.AMADEUS_VISION_MODEL || 'llama3.2-vision:latest').trim();
    const ollamaRes = await fetch(`${OLLAMA_BASE}/api/generate`,{
      method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({
        model: visionModel,
        prompt:`请用"人类视觉体验"的方式描述画面，像你亲眼看到的一样。

要求：用第一人称视角
- 避免使用"画面中、摄像头、监控"等词
- 用自然语言描述
- 可以带一点主观感受（但不要编造）

示例：
- "他正专注地看着屏幕，手指在键盘上敲打"
- "他靠在椅背上，看起来有点疲惫"
- "他一边看手机一边微笑"

如果图像模糊或看不清，就说"看不清楚"`,
        images:[image],stream:false,options:{temperature:0.35,num_predict:60}
      })
    });
    const d = await ollamaRes.json();
    const raw = (d.response||'').replace(/<think>[\s\S]*?(<\/think>|$)/gi,'').trim();

    if (raw && !/看不清楚|unclear/i.test(raw)) {
      lastVision = { description:raw, timestamp:Date.now(), available:true, source:visionModel };

      // ★ 视觉→内部状态处理
      _processVisionForPAD(raw);
      digitalLife.onVision(raw);
      savePAD(padPath, currentPAD);
    }
    res.json(lastVision);
  } catch (e) {
    console.error('[vision]',e.message);
    res.status(503).json({ ...lastVision, available:false, error:'视觉模型当前不可用' });
  }
});

app.get('/get-vision-status', (req, res) => res.json(lastVision));

// ★ 轻量视觉反馈：只更新内存PAD，不写磁盘
app.post('/vision-feedback', (req, res) => {
  try {
    const { text } = req.body;
    if (text) _processVisionForPAD_light(text);
    res.json({ ok: true });
  } catch (e) { res.json({ ok: false }); }
});

/**
 * 轻量摄像头感知：只接收本地帧差事件，不接收图片、不生成文本，
 * 因此不会把每一帧视觉内容注入主对话。
 */
let lastVisionWriteTs = 0;
let lastVisionKind = '';
app.post('/vision-event', (req, res) => {
  try {
    const kind = String(req.body?.kind || '').trim();
    const score = Math.max(0, Math.min(1, Number(req.body?.score) || 0));
    if (!['camera_motion', 'camera_still', 'face_present', 'face_absent'].includes(kind)) {
      return res.status(400).json({ ok: false, error: 'unsupported vision event' });
    }
    // 服务端去重：人脸在/不在反复翻转不刷屏；运动事件也加最小间隔。
    // 摄像头状态是"环境"，不是值得逐条入忆的对话事件。
    const now = Date.now();
    const isFace = kind === 'face_present' || kind === 'face_absent';
    let shouldWrite = false;
    if (isFace) {
      shouldWrite = kind !== lastVisionKind && now - lastVisionWriteTs >= 60000;
    } else if (kind === 'camera_motion') {
      shouldWrite = score >= 0.08 && now - lastVisionWriteTs >= 20000;
    }
    if (shouldWrite) {
      lastVisionWriteTs = now;
      lastVisionKind = kind;
      // 摄像头存在/运动是连续环境状态，不是记忆事件。
      // 只更新去重状态，绝不写入 event_log，否则 RAG 会被无内容事件淹没。
      return res.json({ ok: true, activeChat: false, ignored: true, persisted: false, injectedToChat: false });
    }
    res.json({ ok: true, activeChat: false, ignored: true, injectedToChat: false });
  } catch (e) {
    res.status(500).json({ ok: false, error: e.message });
  }
});

// ★ Design Skill — 免费优先：本地 A1111/SD WebUI 出图；不可用时返回可执行设计方案
app.post('/design', async (req, res) => {
  try {
    const request = String(req.body.request || req.body.prompt || '').trim();
    if (!request) return res.status(400).json({ error: 'empty design request' });

    const record = await createDesignTask({
      rootPath,
      dataDir,
      request,
      options: {
        style: req.body.style || '',
        ratio: req.body.ratio || '1:1',
        render: req.body.render !== false,
        model: req.body.model,
      },
    });

    memorySystem.addEvent(
      'design',
      `设计任务：${request.substring(0, 40)}${record.render ? '（已出图）' : '（方案）'}`,
      0.35,
      { A: 0.04, D: 0.03 }
    );
    res.json({ ok: true, ...record });
  } catch (e) {
    console.error('[design]', e.message);
    res.status(500).json({ error: e.message });
  }
});

// ★ TTS 代理 — 绕过 CORS OPTIONS 405
const SOVITS_URL = process.env.AMADEUS_SOVITS_URL || 'http://localhost:9880';
/** SoVITS 固定随机种子（与 api_v2 一致）；可用 AMADEUS_SOVITS_SEED 覆盖 */
const SOVITS_TTS_SEED = (() => {
  const raw = process.env.AMADEUS_SOVITS_SEED;
  if (raw != null && String(raw).trim() !== '') {
    const n = Number(raw);
    if (Number.isFinite(n)) return Math.trunc(n);
  }
  return 3557467070;
})();

function isRagIndexed() {
  if (fs.existsSync(hnswIndexPath)) return true;
  if (!fs.existsSync(vectorFallbackPath)) return false;
  try {
    const raw = JSON.parse(fs.readFileSync(vectorFallbackPath, 'utf8'));
    return Array.isArray(raw) && raw.length > 0;
  } catch {
    return false;
  }
}

/** 启动自检：Ollama / 模型 / RAG / TTS */
app.get('/health', async (_req, res) => {
  try {
    const report = await runStartupChecks({
      ollamaBase: OLLAMA_BASE,
      chatModel: process.env.AMADEUS_CHAT_MODEL || 'amadeus-kurisu-swallow:8b',
      embedModel: process.env.AMADEUS_EMBED_MODEL || 'nomic-embed-text',
      ragIndexed: isRagIndexed(),
      sovitsUrl: process.env.AMADEUS_SOVITS_URL || 'http://localhost:9880',
      visionModels: String(process.env.AMADEUS_VISION_MODELS || 'llama3.2-vision:latest,llama3.2-vision,qwen2.5vl:7b')
        .split(',').map((s) => s.trim()).filter(Boolean),
    });
    res.status(report.ready ? 200 : 503).json({
      ...report,
      translateModel: resolveTranslateModel(),
      replyLanguage: getReplyLanguageMode(),
      proactiveModel: String(process.env.AMADEUS_PROACTIVE_MODEL || process.env.AMADEUS_CHAT_MODEL || '').trim(),
    });
  } catch (e) {
    res.status(500).json({
      ready: false,
      ok: false,
      hints: ['系统自检失败，请重启后端'],
      error: e.message,
    });
  }
});

/** TTS 专用中→日：固定 AMADEUS_TRANSLATE_MODEL，不走 kurisu */
app.post('/translate/cn-jp', async (req, res) => {
  try {
    const text = String(req.body?.text || req.body?.chinese || '').trim();
    if (!text) return res.status(400).json({ ok: false, error: 'empty text' });
    const result = await _translateChineseToJapaneseForTts(text);
    if (!result.ok) {
      return res.status(502).json({
        ok: false,
        japanese: result.japanese || '',
        model: result.model,
        error: result.error || 'translate failed',
      });
    }
    return res.json({
      ok: true,
      japanese: result.japanese,
      text: result.japanese,
      model: result.model,
    });
  } catch (e) {
    return res.status(500).json({ ok: false, error: String(e.message || e) });
  }
});

/** 仅探测 SoVITS 进程是否监听，不触发语音合成 */
app.get('/tts-health', async (_req, res) => {
  try {
    const base = SOVITS_URL.replace(/\/$/, '');
    const r = await fetch(`${base}/`, { method: 'GET', signal: AbortSignal.timeout(4000) });
    if (r.status >= 200 && r.status < 500) return res.json({ ok: true });
    return res.status(502).json({ ok: false, error: `SoVITS HTTP ${r.status}` });
  } catch (e) {
    const msg = String((e && e.message) || e);
    return res.status(502).json({ ok: false, error: msg });
  }
});
/** GPT-SoVITS 安装目录（参考音 ref.wav 通常在此，而非 Amadeus_Project 根目录） */
const SOVITS_ROOT = (() => {
  const root = process.env.AMADEUS_SOVITS_ROOT;
  if (root && String(root).trim()) return path.normalize(String(root).trim());
  const ref = process.env.AMADEUS_SOVITS_REF;
  if (ref && String(ref).trim()) return path.dirname(path.normalize(String(ref).trim()));
  return '';
})();

const SOVITS_REF_DEFAULT =
  process.env.AMADEUS_SOVITS_REF ||
  (SOVITS_ROOT ? path.join(SOVITS_ROOT, 'ref.wav') : '') ||
  path.join(rootPath, 'ref.wav');

function resolveAsrPython() {
  const configured = String(process.env.AMADEUS_ASR_PYTHON || '').trim();
  if (configured && fs.existsSync(configured)) return configured;
  if (SOVITS_ROOT) {
    const candidate = path.join(SOVITS_ROOT, 'runtime', 'python.exe');
    if (fs.existsSync(candidate)) return candidate;
  }
  const venvPy = path.join(rootPath, '.venv-voice', 'Scripts', 'python.exe');
  if (fs.existsSync(venvPy)) return venvPy;
  return '';
}

function pcm16ToWavBuffer(pcmBuf, sampleRate = 16000) {
  const dataSize = pcmBuf.length;
  const buffer = Buffer.alloc(44 + dataSize);
  buffer.write('RIFF', 0);
  buffer.writeUInt32LE(36 + dataSize, 4);
  buffer.write('WAVE', 8);
  buffer.write('fmt ', 12);
  buffer.writeUInt32LE(16, 16);
  buffer.writeUInt16LE(1, 20);
  buffer.writeUInt16LE(1, 22);
  buffer.writeUInt32LE(sampleRate, 24);
  buffer.writeUInt32LE(sampleRate * 2, 28);
  buffer.writeUInt16LE(2, 32);
  buffer.writeUInt16LE(16, 34);
  buffer.write('data', 36);
  buffer.writeUInt32LE(dataSize, 40);
  pcmBuf.copy(buffer, 44);
  return buffer;
}

function runProcessCapture(command, args, timeoutMs = 45000) {
  return new Promise((resolve) => {
    const child = spawn(command, args, {
      windowsHide: true,
      env: {
        ...process.env,
        PYTHONIOENCODING: 'utf-8',
      },
    });
    const stdoutChunks = [];
    const stderrChunks = [];
    const timer = setTimeout(() => {
      try { child.kill(); } catch (_) {}
      resolve({
        code: -1,
        stdout: Buffer.concat(stdoutChunks).toString('utf8'),
        stderr: Buffer.concat(stderrChunks).toString('utf8') || 'asr timeout',
        stdoutBuf: Buffer.concat(stdoutChunks),
      });
    }, timeoutMs);
    child.stdout.on('data', (chunk) => { stdoutChunks.push(Buffer.from(chunk)); });
    child.stderr.on('data', (chunk) => { stderrChunks.push(Buffer.from(chunk)); });
    child.on('error', (err) => {
      clearTimeout(timer);
      resolve({
        code: -1,
        stdout: Buffer.concat(stdoutChunks).toString('utf8'),
        stderr: String(err.message || err),
        stdoutBuf: Buffer.concat(stdoutChunks),
      });
    });
    child.on('close', (code) => {
      clearTimeout(timer);
      const stdoutBuf = Buffer.concat(stdoutChunks);
      resolve({
        code: Number(code) || 0,
        stdout: stdoutBuf.toString('utf8'),
        stderr: Buffer.concat(stderrChunks).toString('utf8'),
        stdoutBuf,
      });
    });
  });
}

function decodeAsrPayload(parsed) {
  if (!parsed || typeof parsed !== 'object') return '';
  if (parsed.text_b64) {
    try {
      return Buffer.from(String(parsed.text_b64), 'base64').toString('utf8').trim();
    } catch {
      return '';
    }
  }
  return String(parsed.text || '').trim();
}

/** 拒收明显编码损坏/无意义符号，避免通话把乱码送进对话 */
function sanitizeAsrText(text) {
  const t = String(text || '').trim();
  if (!t) return '';
  const replacement = (t.match(/\uFFFD/g) || []).length;
  if (replacement >= 2) return '';
  // 常见 GBK 被当 UTF-8 读时的「Ã/Â/å/æ」乱码簇
  const mojibake = (t.match(/[ÃÂåæçèé]/g) || []).length;
  const han = (t.match(/[\u4e00-\u9fff]/g) || []).length;
  const kana = (t.match(/[\u3040-\u30ff]/g) || []).length;
  if (mojibake >= 3 && han + kana < 2) return '';
  // 几乎全是标点/符号
  const meaningful = t.replace(/[\s\u3000-\u303f\uff00-\uffef.,!?;:'"「」『』（）【】\[\]{}()…·\-—_/\\|+*=<>@#$%^&~`]/g, '');
  if (!meaningful) return '';
  return t;
}

async function transcribeWavFile(wavPath) {
  const ps1 = path.join(rootPath, 'scripts', 'windows-asr-wav.ps1');
  if (fs.existsSync(ps1)) {
    const result = await runProcessCapture('powershell', [
      '-NoProfile', '-ExecutionPolicy', 'Bypass',
      '-File', ps1,
      '-WavPath', wavPath,
      '-Culture', 'zh-CN',
    ], 30000);
    try {
      const line = String(result.stdout || '').trim().split(/\r?\n/).filter(Boolean).pop() || '';
      const parsed = JSON.parse(line);
      if (parsed && parsed.ok) {
        const text = sanitizeAsrText(decodeAsrPayload(parsed));
        return { ok: true, text, engine: parsed.engine || 'windows-speech' };
      }
      if (parsed && parsed.error) console.warn('[asr] windows-speech:', parsed.error);
    } catch (e) {
      console.warn('[asr] windows-speech parse', e.message, String(result.stdout || '').slice(0, 200));
    }
  }

  const py = resolveAsrPython();
  const script = path.join(rootPath, 'scripts', 'asr_transcribe.py');
  if (py && fs.existsSync(script)) {
    const model = String(process.env.AMADEUS_ASR_MODEL || 'tiny').trim() || 'tiny';
    const result = await runProcessCapture(py, [script, '--wav', wavPath, '--language', 'zh', '--model', model], 90000);
    try {
      const line = String(result.stdout || '').trim().split(/\r?\n/).filter(Boolean).pop() || '';
      const parsed = JSON.parse(line);
      if (parsed && parsed.ok) {
        const text = sanitizeAsrText(decodeAsrPayload(parsed) || String(parsed.text || '').trim());
        return { ok: true, text, engine: 'faster-whisper' };
      }
      return { ok: false, error: parsed?.error || result.stderr || 'whisper failed', engine: 'faster-whisper' };
    } catch (e) {
      return { ok: false, error: e.message || result.stderr || 'whisper parse failed', engine: 'faster-whisper' };
    }
  }

  return { ok: false, error: 'no local ASR engine available', engine: 'none' };
}

/** 本地通话 ASR：接收 int16 PCM（base64） */
app.post('/asr', async (req, res) => {
  const tmpWav = path.join(os.tmpdir(), `amadeus-asr-${Date.now()}-${Math.random().toString(36).slice(2)}.wav`);
  try {
    const sampleRate = Math.max(8000, Math.min(48000, Number(req.body.sampleRate) || 16000));
    let wavBuf = null;
    if (req.body.wavBase64) {
      wavBuf = Buffer.from(String(req.body.wavBase64), 'base64');
    } else if (req.body.pcmBase64) {
      const pcm = Buffer.from(String(req.body.pcmBase64), 'base64');
      if (pcm.length < 3200) {
        return res.json({ ok: true, text: '', engine: 'skip', reason: 'too_short' });
      }
      wavBuf = pcm16ToWavBuffer(pcm, sampleRate);
    } else {
      return res.status(400).json({ ok: false, error: 'pcmBase64 or wavBase64 required' });
    }
    fs.writeFileSync(tmpWav, wavBuf);
    const out = await transcribeWavFile(tmpWav);
    if (!out.ok) return res.status(502).json(out);
    return res.json({ ok: true, text: out.text || '', engine: out.engine || 'local' });
  } catch (e) {
    return res.status(500).json({ ok: false, error: e.message });
  } finally {
    try { fs.unlinkSync(tmpWav); } catch (_) {}
  }
});

app.get('/asr/health', (_req, res) => {
  const ps1 = path.join(rootPath, 'scripts', 'windows-asr-wav.ps1');
  const py = resolveAsrPython();
  res.json({
    ok: fs.existsSync(ps1) || !!py,
    windowsSpeech: fs.existsSync(ps1),
    whisperPython: py || '',
  });
});

function _absRefPath(p) {
  const t = String(p || '').trim();
  if (!t) return '';
  return path.isAbsolute(t) ? path.normalize(t) : path.normalize(path.join(rootPath, t));
}

/** 解析参考 wav：相对路径优先在 AMADEUS_SOVITS_ROOT 下查找，避免误用项目内不存在的 ref.wav */
function resolveSoVitsRef(reqBody) {
  const reqNames = [
    reqBody && reqBody.refer_wav_path,
    reqBody && reqBody.ref_audio_path,
  ].filter((p) => typeof p === 'string' && p.trim());

  const tryList = [];
  for (const name of reqNames) {
    const t = name.trim();
    if (path.isAbsolute(t)) {
      tryList.push(path.normalize(t));
    } else if (SOVITS_ROOT) {
      tryList.push(path.join(SOVITS_ROOT, path.basename(t)));
    }
    tryList.push(path.join(rootPath, t));
  }
  if (process.env.AMADEUS_SOVITS_REF) tryList.push(_absRefPath(process.env.AMADEUS_SOVITS_REF));
  if (SOVITS_ROOT) tryList.push(path.join(SOVITS_ROOT, 'ref.wav'));
  tryList.push(
    SOVITS_REF_DEFAULT,
    path.join(rootPath, 'ref.wav'),
    path.join(rootPath, 'assets', 'ref.wav'),
  );

  const seen = new Set();
  for (const p of tryList) {
    const abs = _absRefPath(p) || path.normalize(String(p || '').trim());
    if (!abs || seen.has(abs)) continue;
    seen.add(abs);
    try {
      if (fs.existsSync(abs)) return abs;
    } catch (_) { /* ignore */ }
  }
  return null;
}

app.post('/tts', async (req, res) => {
  // #region agent log
  try {
    const tl = String((req.body && req.body.text) || '').length;
    agentDebugLog({ hypothesisId: 'TTS-C', location: 'server.js:tts.enter', message: 'POST /tts', data: { textLen: tl } });
  } catch (_) {}
  // #endregion
  try {
    const refWav = resolveSoVitsRef(req.body);
    if (!refWav) {
      const hint = SOVITS_ROOT
        ? `未找到参考音频。请确认 ${SOVITS_ROOT} 下有 ref.wav，或设置 AMADEUS_SOVITS_REF。`
        : '未找到参考音频。请设置 AMADEUS_SOVITS_ROOT 或 AMADEUS_SOVITS_REF（GPT-SoVITS 安装目录下的 ref.wav）。';
      return res.status(400).json({ error: hint });
    }
    // SoVITS：speed_factor≠1 时常与并行/分桶冲突并 400；情绪语速改由前端 audio.playbackRate 承担
    const wantParallel = req.body.parallel_infer !== false;
    const body = {
      ...req.body,
      text: String(req.body.text || '')
        .replace(/[、，,]+/g, '、')
        .replace(/[…]+/g, '。')
        .replace(/[、]\s*([。！？!?])/g, '$1')
        .replace(/([。！？!?]){2,}/g, '$1')
        .trim(),
      text_lang: 'ja',
      text_language: 'ja',
      prompt_lang: 'ja',
      prompt_language: 'ja',
      ref_audio_path: refWav,
      refer_wav_path: refWav,
      prompt_text: req.body.prompt_text || '',
      text_split_method: 'cut0',
      speed_factor: 1,
      parallel_infer: wantParallel,
      split_bucket: false,
      batch_size: 1,
      return_fragment: false,
      streaming_mode: false,
      seed: Number.isFinite(Number(req.body.seed)) ? Math.trunc(Number(req.body.seed)) : SOVITS_TTS_SEED,
    };
    if (!body.text) return res.status(400).json({ error: 'empty tts text' });
    const kanaCount = (body.text.match(/[\u3040-\u309f\u30a0-\u30ff\uff65-\uff9f]/g) || []).length;
    const unsupportedSpeechChars = body.text.match(/[\u0080-\u024f\u1e00-\u1eff\u0400-\u04ff]/g) || [];
    if (kanaCount < 2 || unsupportedSpeechChars.length > 0) {
      return res.status(422).json({
        error: 'tts accepts Japanese speech only (clean Japanese required); translate or retry before synthesis',
      });
    }
    const sovitsTtsUrl = `${SOVITS_URL.replace(/\/$/, '')}/tts`;
    const synthTimeoutMs = Math.min(90000, 28000 + body.text.length * 140);
    let r = null;
    let lastSynthErr = null;
    for (let attempt = 0; attempt < 3; attempt++) {
      if (attempt > 0) await new Promise((resolve) => setTimeout(resolve, 450 + attempt * 650));
      try {
        r = await fetch(sovitsTtsUrl, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
          signal: AbortSignal.timeout(synthTimeoutMs),
        });
        if (r.ok) break;
        const errBody = await r.text().catch(() => '');
        lastSynthErr = { status: r.status, errBody };
        if (r.status < 502 || attempt >= 2) break;
      } catch (e) {
        lastSynthErr = { error: e };
        const code = e && (e.code || (e.cause && e.cause.code));
        const msg = String((e && e.message) || e);
        const refused = code === 'ECONNREFUSED' || /ECONNREFUSED|fetch failed|connect/i.test(msg);
        if (!refused || attempt >= 2) throw e;
      }
    }
    if (!r || !r.ok) {
      const errBody = lastSynthErr && lastSynthErr.errBody
        ? lastSynthErr.errBody
        : String((lastSynthErr && lastSynthErr.error && lastSynthErr.error.message) || '');
      const status = (r && r.status) || 502;
      console.error('[tts-proxy] SoVITS error:', status, errBody.substring(0, 200));
      // #region agent log
      agentDebugLog({ hypothesisId: 'TTS-B', location: 'server.js:tts.sovitsErr', message: 'SoVITS non-ok', data: { status, sovitsUrl: SOVITS_URL, errSlice: errBody.substring(0, 160) } });
      // #endregion
      return res.status(status).json({ error: `SoVITS: ${errBody.substring(0, 100)}` });
    }
    res.setHeader('Content-Type', r.headers.get('content-type') || 'audio/wav');
    // 使用管道流式传输，减少内存占用并降低首包延迟
    const reader = r.body.getReader();
    // 由于 Node.js fetch 返回的是 web stream，我们手动读取并写入 Express 的 res (Node writable stream)
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      res.write(value);
    }
    res.end();
    // #region agent log
    agentDebugLog({ hypothesisId: 'TTS-B', location: 'server.js:tts.ok', message: 'SoVITS streaming started', data: { sovitsUrl: SOVITS_URL } });
    // #endregion
  } catch (e) {
    const code = e && (e.code || (e.cause && e.cause.code));
    const msg = String((e && e.message) || e);
    const refused = code === 'ECONNREFUSED' || /ECONNREFUSED|fetch failed|connect/i.test(msg);
    const hint = refused
      ? '无法连接 GPT-SoVITS（9880）。若 api_v2 在 TTSPipeline 初始化就崩溃，需先修复 SoVITS 环境再启动服务。'
      : msg;
    console.error('[tts-proxy]', msg, code || '');
    // #region agent log
    agentDebugLog({ hypothesisId: 'TTS-B', location: 'server.js:tts.catch', message: 'tts proxy error', data: { errMsg: msg.slice(0, 200), code: code || null, sovitsUrl: SOVITS_URL, refused } });
    // #endregion
    res.status(502).json({ error: hint });
  }
});

// ★ 用户档案 — 让她知道在和谁对话
app.get('/whoami', (req, res) => {
  try {
    res.json(ensureWhoamiOnDisk(whoamiPath));
  } catch (e) {
    res.json(ensureWhoamiOnDisk(whoamiPath));
  }
});

app.post('/whoami', (req, res) => {
  try {
    const current = ensureWhoamiOnDisk(whoamiPath);
    const { name, traits, preference, basic_key, basic_value, relationship_note } = req.body;
    if (name) current.name = name;
    if (traits && Array.isArray(traits)) {
      for (const t of traits) {
        if (t && !current.traits.includes(t)) current.traits.push(t);
      }
      if (current.traits.length > 20) current.traits = current.traits.slice(-20);
    }
    if (preference && !current.preferences.includes(preference)) {
      current.preferences.push(preference);
      if (current.preferences.length > 20) current.preferences = current.preferences.slice(-20);
    }
    if (basic_key && basic_value) current.basics[basic_key] = basic_value;
    if (relationship_note) current.relationship_note = relationship_note;
    current.last_updated = Date.now();
    fs.writeFileSync(whoamiPath, JSON.stringify(current, null, 2));
    res.json({ ok: true, profile: current });
  } catch (e) { res.status(500).json({ error: e.message }); }
});

// ★ Ollama 代理，解决前端直连 CORS 问题或模型加载超时
app.post('/ollama/:api', async (req, res) => {
  const api = req.params.api; // e.g. chat, generate, tags
  const url = `${OLLAMA_BASE}/api/${api}`;
  try {
    const r = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req.body),
      signal: AbortSignal.timeout(120000) // 2分钟超时，允许模型加载
    });
    if (!r.ok) {
      const errTxt = await r.text().catch(() => '');
      return res.status(r.status).send(errTxt);
    }
    // 如果是流式，则流式转发
    if (req.body.stream) {
      const reader = r.body.getReader();
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        res.write(value);
      }
      res.end();
    } else {
      const json = await r.json();
      res.json(json);
    }
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

app.listen(PORT, () => {
  console.log(`Amadeus 后端 v7.0 已就绪: http://localhost:${PORT}`);
  console.log(`系统：记忆(${memorySystem.events.length}条) / 动机 / 行为决策 / 好奇心 / 用户理解 / 学习引擎 / 元认知`);
  const _defCtx = Number(process.env.AMADEUS_OLLAMA_NUM_CTX);
  const _ctx = Number.isFinite(_defCtx) && _defCtx > 0 ? _defCtx : 2048;
  const _maxP = Number(process.env.AMADEUS_MAX_PROMPT_CHARS);
  const _cap = Number.isFinite(_maxP) && _maxP > 0 ? _maxP : 6000;
  const _ka = String(process.env.AMADEUS_OLLAMA_KEEP_ALIVE || '2m').trim() || '2m';
  console.log(`[ollama] 默认 num_ctx=${_ctx} maxPromptChars=${_cap} keep_alive=${_ka}（8GB 友好；覆盖请设环境变量）`);
  console.log(`[ollama] 若 GPU 空闲：请在运行 ollama serve 的环境设置 OLLAMA_NUM_GPU=999、OLLAMA_FLASH_ATTENTION=1，见 docs/GPU_OLLAMA_SOVITS.md`);
  if (process.env.AMADEUS_PREWARM !== '0') {
    const warmModel = process.env.AMADEUS_CHAT_MODEL || 'amadeus-kurisu-swallow:8b';
    // Most first interactions are short social turns. Warm the same context
    // size they use; warming 2048 and then serving 1024 forces Ollama to
    // unload/reload the 6GB model and makes the first visible token slow.
    const _fastCtx = Number(process.env.AMADEUS_FAST_CHAT_NUM_CTX);
    const _warmCtxEnv = Number(process.env.AMADEUS_PREWARM_NUM_CTX);
    const warmCtx = Number.isFinite(_warmCtxEnv) && _warmCtxEnv >= 768
      ? _warmCtxEnv
      : (Number.isFinite(_fastCtx) && _fastCtx >= 768 ? _fastCtx : 1024);
    setTimeout(() => {
      const started = Date.now();
      fetch(`${OLLAMA_BASE}/api/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: warmModel,
          prompt: '',
          stream: false,
          keep_alive: _ka,
          options: { num_predict: 1, num_ctx: warmCtx },
        }),
        signal: AbortSignal.timeout(45000),
      }).then((r) => {
        if (r.ok) console.log(`[ollama] ${warmModel} 预热完成 ${Date.now() - started}ms (ctx=${warmCtx})`);
      }).catch((e) => console.warn(`[ollama] 预热跳过: ${e.message}`));
    }, 300);
  }

  // GPT-SoVITS 首次合成会加载模型；在后台预热，避免用户第一句回复
  // 额外承担一次十秒级等待。预热只在本机进行，不进入对话、记忆或日志。
  if (process.env.AMADEUS_TTS_PREWARM !== '0') {
    const ttsWarmText = 'テスト。';
    const warmTts = async (attempt = 0) => {
      try {
        const started = Date.now();
        const r = await fetch(`http://127.0.0.1:${PORT}/tts`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: ttsWarmText }),
          signal: AbortSignal.timeout(45000),
        });
        // Consume the body so the proxy can release the upstream connection.
        if (r.ok) {
          await r.arrayBuffer();
          console.log(`[tts] background prewarm completed ${Date.now() - started}ms`);
          return;
        }
        throw new Error(`HTTP ${r.status}`);
      } catch (e) {
        // Electron starts the backend before the one-click launcher finishes
        // bringing SoVITS up. Retry a few times without blocking the UI.
        if (attempt < 4) {
          setTimeout(() => { void warmTts(attempt + 1); }, 5000);
        } else {
          console.warn(`[tts] background prewarm skipped: ${e.message}`);
        }
      }
    };
    setTimeout(() => { void warmTts(); }, 1200);
  }
});
