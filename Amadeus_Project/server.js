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
const { BehaviorIngest } = require('./lib/behaviorIngest');
const { normalizeClientContext, buildClientContextBlock } = require('./lib/clientContext');
const { InnerStateSix } = require('./cognitive/innerStateSix');
const { ConversationInitiativeEngine } = require('./cognitive/conversationInitiative');
const { readSocialField, nextSenseMs } = require('./cognitive/socialRead');
const { EmotionalBandwidthEngine } = require('./cognitive/emotionalBandwidth');
const { ButlerKernel } = require('./lib/butler/kernel');
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
    if (process.env[key] == null || process.env[key] === '') process.env[key] = val;
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
const conversationInitiative = new ConversationInitiativeEngine({
  statePath: path.join(dataDir, 'conversation_initiative.json'),
});
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
  updates: butlerKernel.updatesSince(req.query.since, req.query.limit),
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
app.post('/butler/reminders/:id/delivered', butlerRoute((req) => ({
  reminder: butlerKernel.reminders.markDelivered(req.params.id),
})));
app.post('/butler/reminders/:id/acknowledge', butlerRoute((req) => ({
  reminder: butlerKernel.reminders.acknowledge(req.params.id),
})));
app.get('/butler/trash', butlerRoute((req) => ({
  entries: butlerKernel.fileUndo.list(String(req.query.status || '')),
})));

const soulPath    = path.join(rootPath, "kurisu_soul.txt");
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
let cachedVoiceContent = '';
let cachedCharacterRules = '';
function loadSoulCache() {
  const parts = [];
  try { parts.push(fs.readFileSync(corePromptPath, 'utf8')); } catch {}
  try { parts.push(fs.readFileSync(soulPath, 'utf8')); } catch {}
  cachedSoulContent = parts.filter(Boolean).join('\n\n---\n\n');
  try { cachedVoiceContent = fs.readFileSync(voicePath, 'utf8'); } catch {}
  try { cachedCharacterRules = fs.readFileSync(characterRulesPath, 'utf8'); } catch {}
}
loadSoulCache();

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
let ragStore = null, hnswVectorStore = null;
let HNSWLib = null, OllamaEmbeddings = null;
try {
  ({ HNSWLib } = require("@langchain/community/vectorstores/hnswlib"));
  ({ OllamaEmbeddings } = require("@langchain/ollama"));
} catch (e) { console.warn("[rag] LangChain HNSW unavailable:", e.message); }

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
  if(HNSWLib&&OllamaEmbeddings&&fs.existsSync(hnswIndexPath)) {
    try {
      if(!hnswVectorStore) {
        const emb=new OllamaEmbeddings({model:"nomic-embed-text",baseUrl:OLLAMA_BASE});
        hnswVectorStore=await HNSWLib.load(vectorDir,emb);
      }
      const docs=await hnswVectorStore.similaritySearchWithScore(query.trim(),topK);
      return docs.map(([doc,score])=>({text:doc.pageContent,score,source:doc.metadata?.source||"unknown"}));
    } catch(e) { console.warn("[rag] HNSW fallback:",e.message); }
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
function loadMotivationState() {
  try {
    if (fs.existsSync(motivePath)) {
      return JSON.parse(fs.readFileSync(motivePath, 'utf8'));
    }
  } catch {}
  return { desire_closeness: 0.25, fear_rejection: 0.55, curiosity: 0.5 };
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
  saveMotivationState(motivationState);
}

// ══════════════════════════════════════════════════════════════════
//  全局实例
// ══════════════════════════════════════════════════════════════════
const memorySystem  = new MemorySystem(memoryDir, eventLogPath);
const memoryAdmission = new MemoryAdmissionPolicy(memoryDir);
const unifiedDialogueLog = new UnifiedDialogueLog(memoryDir);
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

// ── 启动：记忆衰减 ───────────────────────────────────────────────
memorySystem.decay();
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
    });
    if (item) {
      observeMemoryEvidence(
        role === 'user' ? 'user' : req.body.proactive === true ? 'proactive' : 'assistant',
        item.text,
      );
    }
    res.json({ ok: true, item });
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
    const phase = String(req.body.phase || 'floor_release');
    const lastUserText = String(req.body.lastUserText || req.body.userText || '');
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
    };
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
    const body = req.body || {};
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

    const behavior = digitalLife.evaluateAutonomy({
      pad,
      memorySystem,
      relScore,
      idleMs,
      quietMs,
      dtMs,
      facePresent,
      faceMs,
      social,
      lastUserText,
      lastReplyText,
      alreadyTalking,
      awaitingProactiveReply: body.awaitingProactiveReply === true,
      isThinking: body.isThinking === true,
      ttsPlaying: body.ttsPlaying === true,
      userPresenceActive,
      dnd,
      proactiveQuotaOk: body.proactiveQuotaOk !== false,
      sheSpokeRecently,
    });

    const urgeIntensity = Number(behavior?.primaryUrge?.intensity) || 0;
    const senseMs = nextSenseMs(social, urgeIntensity);

    const decision = conversationInitiative.decideBeside({
      autonomyShouldAct: behavior?.shouldAct === true && behavior?.suppressProactive !== true,
      autonomyReason: behavior?.shouldAct ? '' : (behavior?.speakHint || 'waiting'),
      speakHint: behavior?.speakHint || '',
      urgeIntentKey: behavior?.primaryUrge?.intentKey || behavior?.primaryUrge?.intent || '',
      urgeIntent: behavior?.primaryUrge?.intent || '',
      social,
      alreadyTalking,
      dialogueStarted: alreadyTalking,
      lastUserText,
      idleMs,
      facePresent,
      dnd,
      proactiveQuotaOk: body.proactiveQuotaOk !== false,
      nextCheckMs: senseMs,
    });

    res.json({
      ok: true,
      ...decision,
      nextCheckMs: decision.nextCheckMs || senseMs,
      social,
      urge: behavior?.primaryUrge || null,
      speakHint: behavior?.speakHint || decision.reason || '',
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
  res.json({ ok: true, ...conversationInitiative.snapshot() });
});

app.post('/initiative/feedback', (req, res) => {
  try {
    const state = conversationInitiative.registerFeedback({ type: req.body.type, text: req.body.text });
    res.json({ ok: true, state });
  } catch (e) {
    res.status(500).json({ ok: false, error: e.message });
  }
});

app.post('/initiative/sent', (req, res) => {
  try {
    const state = conversationInitiative.registerSent({ text: req.body.text, action: req.body.action });
    res.json({ ok: true, state });
  } catch (e) {
    res.status(500).json({ ok: false, error: e.message });
  }
});

// ── 主动开口 lite 通道（低 token、快响应）────────────────────────
app.post('/chat/lite', async (req, res) => {
  try {
    const userLine = String(req.body.userText || req.body.message || '（想说话）').trim();
    let lastUserAnchor = String(req.body.lastUserAnchor || '').slice(0, 200);
    const lastKurisuAnchor = String(req.body.lastKurisuAnchor || '').slice(0, 200);
    const legacyAnchor = String(req.body.proactiveAnchor || '').slice(0, 200);
    const isAutonomyInitiative = req.body.isAutonomyInitiative === true || /^（想说话）/.test(userLine);
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
      bubbleCount: Math.max(1, Math.min(3, Number(requestedDelivery.bubbleCount) || 1)),
      maxCharsPerBubble: Math.max(8, Math.min(32, Number(requestedDelivery.maxCharsPerBubble) || 20)),
      allowNonSemantic: requestedDelivery.allowNonSemantic === true,
    };
    const clientOwnsLog = req.body.clientOwnsLog === true;
    const idleMs = Math.max(0, Number(req.body.idleMsSinceUser) || 0);
    const model = req.body.model
      || process.env.AMADEUS_LITE_MODEL
      || process.env.AMADEUS_CHAT_MODEL
      || 'kurisu:latest';
    const temp = Number(req.body.temperature) || 0.82;
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
        : unifiedDialogueLog.toOllamaDialogue({ maxMsgs: 8 });
    if (!isConversationInitiative && /^（想说话）/.test(userLine)) dialogue.push({ role: 'user', content: userLine });
    // 主动开口也要看见近况，否则会说出让人觉得「没感觉到人」的怪话
    const conversationCtx = isConversationInitiative
      ? ''
      : isAutonomyInitiative
        ? (recentDialogue.length
          ? recentDialogue.map((m) => `${m.role === 'user' ? '他' : 'Kurisu'}: ${m.content}`).join('\n').slice(0, 900)
          : unifiedDialogueLog.toPromptBlock({ maxChars: 700, userText: '' }))
        : unifiedDialogueLog.toPromptBlock({ maxChars: 900, userText: userLine });
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
      };
      continuityLine = [
        '【对话内自主意图】这不是固定补话、定时问候或第二次回答。',
        `他刚才说：「${lastUserAnchor}」`,
        `你刚回答：「${lastKurisuAnchor}」`,
        `你此刻想开口的原因：${initiativeReason || '产生了一个具体内在动机'}`,
        `行动意图（只说明动机，不规定台词）：${actionContracts[initiativeAction] || '顺着真实动机自然说一句。'}`,
        `发送方式：${delivery.style}；约 ${delivery.bubbleCount} 个气泡；每个尽量短。气泡之间用空行分隔。`,
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
          `约 ${delivery.bubbleCount} 个短气泡，气泡间空一行。`,
        ]
        : [
          '【察觉到人 · 先开口】两个人相见，总有人先说话。这里是文字窗口刚碰上，不是打电话。',
          initiativeReason ? `此刻动机：${initiativeReason}` : '',
          '意图层：可以搞笑、轻微骚扰、开玩笑、或只让他感觉到你也感觉到了他——只要是你想。',
          '禁止编造通话状态（打电话/还没打来/接通）。',
          '表达层：牧濑红莉栖本人；自由发挥，不要套固定开场模板。',
          `约 ${delivery.bubbleCount} 个短气泡，气泡间空一行；长短随你。`,
          '不要查岗说教，不要解释“我为什么主动”。',
        ]
      ).filter(Boolean).join('\n');
    } else if (isAutonomyInitiative && initiativePhase === 'coldness') {
      continuityLine = [
        '【冷感回拉】气氛有点冷/敷衍或他嫌无聊。你想把场子拉回来——因为你在意这场对话，不是讨好。',
        lastUserAnchor ? `他上一句：「${lastUserAnchor}」——优先接这句里的词和情绪。` : '',
        initiativeReason ? `此刻动机：${initiativeReason}` : '',
        '意图层：顺着他上一句给一点具体动静（吐槽、半句正经、或接他刚说的处境）；不要另起电话/见面戏。',
        '禁止元评论收束：「又是这个话题吗」「又是那个话题」「好无聊」这种把话题判死刑的固定腔——他喊无聊时尤其禁止复读无聊。',
        '禁止编造：打电话/还没打来/刚刚才注意到/你们怎么还不来——这些都不是当前事实。',
        '表达层：你是独立的灵魂；毒舌、傲、软都可以，只要是你此刻想说的。',
        `约 ${delivery.bubbleCount} 个短气泡，气泡间空一行。`,
      ].filter(Boolean).join('\n');
    } else if (isAutonomyInitiative && lastUserAnchor && contextFresh) {
      continuityLine = [
        '【主动开口 · 接他上一句】',
        `他上一句：「${lastUserAnchor}」`,
        '你要接着他的话开口；这句话是他说的，不是你自己刚说的。',
        '用你自己的声音说，不要通用陪伴腔。',
        `按 ${delivery.style} 方式发约 ${delivery.bubbleCount} 个短气泡，气泡间空一行。`,
      ].join('\n');
    } else if (isAutonomyInitiative) {
      continuityLine = [
        '【空闲时随手开口】这不是任务或提醒——是你自己想说。',
        '意图可以很轻：吐槽、玩笑、轻微骚扰、一个小念头、一点动静。',
        '你是独立的灵魂；措辞按牧濑红莉栖本人自由发挥。',
        `约 ${delivery.bubbleCount} 个短气泡，气泡间空一行。`,
        '不要从旧聊天里硬捞名词续讲，不要说教，不要汇报自己在做实验。',
      ].join('\n');
    } else if (lastKurisuAnchor) {
      continuityLine = `【主动延续】你上一轮主动说的是：「${lastKurisuAnchor}」——可顺着补半句，别丢题。`;
    } else if (!isAutonomyInitiative && legacyAnchor) {
      continuityLine = `【承接话题】他上一句：「${legacyAnchor}」——顺着聊，别把用户话当成你自己说的。`;
    }

    let consciousnessBlock = '';
    let consciousnessMeta = null;
    try {
      const proactive = getBrain().evaluateProactiveSpeech({
        idleMs,
        pad: currentPAD,
        refreshWorld: true,
      });
      consciousnessMeta = {
        shouldSpeak: proactive.shouldSpeak,
        intention: proactive.intention,
        narrative: proactive.narrative,
      };
      if (proactive.workspaceBlock) {
        consciousnessBlock = proactive.workspaceBlock;
      }
      // 严格模式：意识未达开口阈值则拒绝主动 lite（前端可忽略）
      if (
        isAutonomyInitiative
        && !isConversationInitiative
        && String(process.env.AMADEUS_BRAIN_CONSCIOUS_PROACTIVE || '0').trim() === '1'
        && !proactive.shouldSpeak
      ) {
        return res.json({
          ok: true,
          skipped: true,
          reason: 'consciousness_threshold',
          consciousness: consciousnessMeta,
          response: '',
          choices: [{ message: { role: 'assistant', content: '' } }],
        });
      }
    } catch (e) {
      console.warn('[chat/lite] consciousness', e.message);
    }

    const affect = emotionalBandwidth.resolve({
      userText: String(lastUserAnchor || userLine || '').replace(/^（想说话）\s*/, ''),
      pad: currentPAD,
      relScore,
      relHigh: relScore >= 0.55,
    });

    const litePrompt = [
      cachedVoiceContent ? `【口吻】\n${_clipInnerPrompt(cachedVoiceContent, 900)}` : '',
      conversationCtx,
      consciousnessBlock || digitalLife.buildPromptContext({
        pad: currentPAD,
        memory: memorySystem,
        relScore,
        includeDream: idleMs > 20 * 60 * 1000,
      }),
      continuityLine,
      affect.block || '',
      `亲近 ${relScore.toFixed(2)} · 内在 ${padTelemetry(currentPAD)}`,
      '只写聊天气泡正文。允许自然语气词和偶尔的颜文字；不要每次都完整、正式、有结论。',
      '禁止旁白、Markdown、【意识广播】清单，以及 [打算]/[注意到]/[感受] 等内部标签。',
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
      return stripConsciousnessEcho(stripRoleplayActions(text));
    };
    let data = await requestLiteOnce(ollamaMessages);
    let reply = extractLiteText(data);
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
    }
    if (!reply && REPLY_FALLBACK_ENABLED && !isAutonomyInitiative) {
      reply = companionFallbackForLowInfo(realUserLine);
    }

    if (reply && !clientOwnsLog) {
      unifiedDialogueLog.append('assistant', reply, { proactive: true, autonomy: true, lite: true, source: 'lite' });
      observeMemoryEvidence('proactive', reply);
      conversationInitiative.registerSent({ text: reply, action: initiativeAction || 'lite' });
    }
    if (reply && affect?.band) {
      emotionalBandwidth.registerSpoken(affect.band);
    }

    res.json({
      ok: true,
      response: reply,
      affectBand: affect?.band || null,
      consciousness: consciousnessMeta,
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
    cachedVoiceContent,
    cachedCharacterRules,
    memorySystem,
    memoryAdmission,
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
  const autoWorkBrain = options.autoWorkBrain === true || process.env.AMADEUS_WORK_AUTO === '1';
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

function _polishResult(chinese, japanese = '') {
  return {
    chinese: String(chinese || '').trim(),
    japanese: String(japanese || '').trim(),
  };
}

async function _translateJapaneseToChinese(jp) {
  const translateModel = resolveTranslateModel();
  const body = String(jp || '').trim();
  if (!body) return '';
  // 用户要求：界面中文必须是硬翻，禁止通顺化脑补；再按日语原文纠偏常见错译
  const raw = await _ollamaChatOnce(
    translateModel,
    buildLiteralJpToCnMessages(body),
    { temperature: 0.0, num_predict: 280, num_ctx: 1536 },
  );
  return alignLiteralCnToJapanese(body, raw);
}

async function _polishJapaneseModelReply(jpRaw, userContent = '', extra = {}) {
  const logBlock = unifiedDialogueLog.toPromptBlock({ maxChars: 2000 });
  let whoamiName = '';
  try {
    whoamiName = resolvePartnerDisplayName(ensureWhoamiOnDisk(whoamiPath)) || '';
  } catch { /* ignore */ }

  // TTS 用可提取的日语正文；界面硬翻用去装饰后的完整原话（含中日夹杂）
  const rawForDisplay = stripConsciousnessEcho(stripModelDecorations(jpRaw));
  let jp = extractJapaneseBody(jpRaw) || rawForDisplay;
  if (!jp && !rawForDisplay) return _polishResult('');

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
  }

  let cn = '';
  try {
    cn = await _translateJapaneseToChinese(rawForDisplay || jp);
  } catch (e) {
    console.warn('[jp-pipeline] 日译中', e.message);
  }
  cn = String(cn || '').trim();
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
  const logBlock = unifiedDialogueLog.toPromptBlock({ maxChars: 2000 });
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

/** 回复后的PAD反馈更新（分析AI回复的情感倾向） */
async function _postReplyPadUpdate(reply, userContent = '', extra = {}) {
  let finalReply = String(reply || '').trim();
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
    finalReply = polished.chinese || finalReply;
    modelJp = polished.japanese || '';
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

  if (finalReply) {
    const loggedReply = unifiedDialogueLog.append('assistant', finalReply, {
      conversationId: extra.conversationId,
      turnId: extra.autonomy ? '' : extra.turnId,
      proactive: extra.autonomy === true,
      autonomy: extra.autonomy === true,
      source: extra.autonomy ? 'autonomy' : (extra.source || 'chat'),
    });
    observeMemoryEvidence(extra.autonomy ? 'proactive' : 'assistant', finalReply);
    if (extra.autonomy && loggedReply) conversationInitiative.registerSent({ text: finalReply, action: 'formal' });
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
  return _polishResult(finalReply, modelJp);
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
app.post('/vision-event', (req, res) => {
  try {
    const kind = String(req.body?.kind || '').trim();
    const score = Math.max(0, Math.min(1, Number(req.body?.score) || 0));
    if (!['camera_motion', 'camera_still', 'face_present', 'face_absent'].includes(kind)) {
      return res.status(400).json({ ok: false, error: 'unsupported vision event' });
    }
    if ((kind === 'camera_motion' && score >= 0.08) || kind === 'face_present' || kind === 'face_absent') {
      const event = memorySystem.addEvent(
        kind === 'camera_motion' ? 'vision_motion' : 'vision_presence',
        kind === 'camera_motion'
          ? `摄像头检测到画面变化（${score.toFixed(2)}），仅作为环境事件记录`
          : `摄像头检测到${kind === 'face_present' ? '有人在场' : '暂未检测到人脸'}，仅作为环境事件记录`,
        kind === 'camera_motion' ? 0.025 : 0.04,
        kind === 'camera_motion' ? { A: 0.01 } : {},
      );
      return res.json({ ok: true, activeChat: false, eventId: event.id, injectedToChat: false });
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
      chatModel: process.env.AMADEUS_CHAT_MODEL || 'kurisu:latest',
      embedModel: process.env.AMADEUS_EMBED_MODEL || 'nomic-embed-text',
      ragIndexed: isRagIndexed(),
      sovitsUrl: process.env.AMADEUS_SOVITS_URL || 'http://localhost:9880',
      visionModels: String(process.env.AMADEUS_VISION_MODELS || 'llama3.2-vision:latest,llama3.2-vision,qwen2.5vl:7b')
        .split(',').map((s) => s.trim()).filter(Boolean),
    });
    res.status(report.ready ? 200 : 503).json({
      ...report,
      translateModel: resolveTranslateModel(),
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
    if (kanaCount < 2) {
      return res.status(422).json({
        error: 'tts accepts Japanese speech only; Chinese text must be translated before synthesis',
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
    const warmModel = process.env.AMADEUS_CHAT_MODEL || 'kurisu:latest';
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
