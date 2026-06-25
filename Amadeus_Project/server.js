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
const { createDesignTask } = require('./design_engine');
const { MemorySystem } = require('./lib/memory');
const { UnifiedDialogueLog, needsConversationRecall } = require('./lib/unifiedDialogueLog');
const { BehaviorIngest } = require('./lib/behaviorIngest');
const { InnerStateSix } = require('./cognitive/innerStateSix');
const { buildSocialIdentityPrompt } = require('./cognitive/socialIdentity');
const { buildExpressionVariantBlock } = require('./cognitive/expressionVariants');
const {
  validateJapaneseLine,
  buildSelfCheckPrompt,
  parseSelfCheckJson,
  runJapaneseValidationPipeline,
} = require('./cognitive/japanesePipeline');

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
const { parseIncomingChat, capDialogue, buildOllamaMessages, fitSystemForDialogue, estimateMessageChars } = require('./cognitive/chatTurns');
const { derivePresence, presenceToPromptLine } = require('./cognitive/presence');
const { repairKurisuReply, reconcileFinalReply } = require('./lib/oocGuard');
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
} = require('./cognitive/turnContinuity');
const userPresence = require('./lib/userPresence');
const { runStartupChecks } = require('./lib/startupCheck');

function applyOocRepair(content, userContent, streamedRaw = '', oocOpts = {}) {
  const streamed = String(streamedRaw || '').trim();
  const repaired = repairKurisuReply(userContent, content, oocOpts);
  const out = streamed ? reconcileFinalReply(streamed, repaired, userContent) : repaired;
  if (out !== String(content || '').trim()) {
    console.log('[chat] OOC/口吻兜底已调整回复');
  }
  return out;
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
const dataDir     = path.join(os.homedir(), 'amadeus_data');

// ★ 托管静态文件
app.use(express.static(rootPath));

// ★ 根路由重定向到主页面，解决 404 问题
app.get('/', (req, res) => {
  res.sendFile(path.join(rootPath, 'amadeus_work.html'));
});

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
const unifiedDialogueLog = new UnifiedDialogueLog(memoryDir);
/** @deprecated 别名，统一实录 */
const conversationMemory = unifiedDialogueLog;
const behaviorIngest = new BehaviorIngest(memoryDir);
const innerStateSix = new InnerStateSix(memoryDir);
const motivSystem   = new MotivationSystem();
const behaviorSys   = new BehaviorDecision();
const selfModel     = new SelfModel(selfModelPath, debounceFileWrite);
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
    });
    res.json({ ok: true, item });
  } catch (e) {
    res.status(500).json({ ok: false, error: e.message });
  }
});

// ── 行为数据上报（无原生采集，仅接收管道）────────────────────────
app.post('/behavior-report', (req, res) => {
  try {
    const result = behaviorIngest.ingest(req.body || {});
    res.json({ ok: true, ...result, snapshot: behaviorIngest.snapshot() });
  } catch (e) {
    res.status(500).json({ ok: false, error: e.message });
  }
});

// ── 主动开口 lite 通道（低 token、快响应）────────────────────────
app.post('/chat/lite', async (req, res) => {
  try {
    const userLine = String(req.body.userText || req.body.message || '（想说话）').trim();
    const anchor = String(req.body.proactiveAnchor || '').slice(0, 200);
    const idleMs = Math.max(0, Number(req.body.idleMsSinceUser) || 0);
    const model = req.body.model || process.env.AMADEUS_LITE_MODEL || 'kurisu:latest';
    const temp = Number(req.body.temperature) || 0.82;
    const maxTok = Math.min(180, Number(req.body.max_tokens) || 120);

    if (userLine && !/^（想说话）/.test(userLine)) {
      unifiedDialogueLog.append('user', userLine, { source: 'lite' });
    }

    const dialogue = unifiedDialogueLog.toOllamaDialogue({ maxMsgs: 8 });
    const conversationCtx = unifiedDialogueLog.toPromptBlock({ maxChars: 900, userText: userLine });
    const relScore = effectiveRelScore(memorySystem.getRelationshipScore());

    const litePrompt = [
      cachedVoiceContent ? `【口吻】\n${_clipInnerPrompt(cachedVoiceContent, 1200)}` : '',
      conversationCtx,
      digitalLife.buildPromptContext({
        pad: currentPAD,
        memory: memorySystem,
        relScore,
        includeDream: idleMs > 20 * 60 * 1000,
      }),
      anchor ? `【主动延续】你刚才说的是：${anchor}` : '【主动】像偶尔想起来才发一句，短、自然，禁止查岗连发。',
      `亲近 ${relScore.toFixed(2)} · 内在 ${padTelemetry(currentPAD)}`,
      '只写中文口语 1～2 句。禁止旁白与 Markdown。',
    ].filter(Boolean).join('\n\n');

    const ollamaMessages = buildOllamaMessages(litePrompt, dialogue, 3200, 4);
    const ollamaRes = await fetch(`${OLLAMA_BASE}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model,
        messages: ollamaMessages,
        stream: false,
        options: { temperature: temp, num_predict: maxTok, num_ctx: 1536 },
      }),
    });
    if (!ollamaRes.ok) {
      const detail = await ollamaRes.text().catch(() => '');
      throw new Error(detail || `Ollama HTTP ${ollamaRes.status}`);
    }
    const data = await ollamaRes.json();
    let reply = stripChatMarkdown((data.message && data.message.content) || data.response || '').trim();
    reply = stripRoleplayActions(reply);

    if (reply) {
      unifiedDialogueLog.append('assistant', reply, { proactive: true, autonomy: true, lite: true, source: 'lite' });
    }

    res.json({ ok: true, response: reply, choices: [{ message: { role: 'assistant', content: reply } }] });
  } catch (e) {
    res.status(502).json({ ok: false, error: e.message, response: '' });
  }
});

/** Ollama /api/chat 流式块：可能是 message.content 或旧版 response */
function ollamaChatStreamRawPiece(obj) {
  if (!obj || typeof obj !== 'object') return '';
  const mc = obj.message && typeof obj.message.content === 'string' ? obj.message.content : '';
  const rs = typeof obj.response === 'string' ? obj.response : '';
  return mc || rs || '';
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
//  POST /chat  ★ 三大系统整合版
// ──────────────────────────────────────────────────────────────
app.post('/chat', async (req, res) => {
  const __streamRequested = req.body && req.body.stream === true;
  try {
    const model    = req.body.model || 'kurisu:latest';
    const useStream= req.body.stream === true;  // 默认非流式
    const temp     = req.body.temperature ?? (Number(process.env.AMADEUS_CHAT_TEMP) || 0.72);
    const maxTok   = req.body.max_tokens  ?? 384;

    const parsed = parseIncomingChat(req.body);
    let userContent = String(parsed.lastUser || '').trim();

    const histCap = Number(process.env.AMADEUS_CHAT_HISTORY_MSGS);
    const maxHist = Number.isFinite(histCap) && histCap > 0 ? histCap : 24;

    // 单一 prompt 源：忽略前端 system，仅用服务端 soul
    let systemContent = cachedSoulContent;
    if (cachedCharacterRules) {
      systemContent = systemContent + '\n\n' + cachedCharacterRules;
    }
    systemContent = systemContent.trim();

    if (!userContent) {
      userContent = String(req.body.userMsg || req.body.message || '').trim();
    }

    unifiedDialogueLog.syncFromDialogue(parsed.dialogue);

    const autonomyInitiative = req.body.autonomyInitiative === true;
    const idleMsSinceUser = Math.max(0, Number(req.body.idleMsSinceUser) || 0);
    const threadHint = detectReplyingToHerThread(parsed.dialogue);
    const replyingToProactive = req.body.replyingToProactive === true
      || (threadHint.active && !autonomyInitiative);
    const proactiveAnchor = String(req.body.proactiveAnchor || threadHint.anchor || '').trim();
    let autonomyDecision = null;
    let dialogueForOllama = [];

    const recentUserLinesForMem = (parsed.userLines && parsed.userLines.length)
      ? parsed.userLines.filter((l) => l && !/^（想说话）|^（转移话题）|^（以下是最近对话/.test(String(l).trim())).slice(-14)
      : (userContent && !/^（想说话）/.test(userContent) ? [userContent] : []);
    const lastRealUserLine = extractLastRealUserLine(parsed.dialogue)
      || (recentUserLinesForMem.length ? recentUserLinesForMem[recentUserLinesForMem.length - 1] : '');
    let userPresenceState = req.body.userPresence && req.body.userPresence.active
      ? req.body.userPresence
      : null;
    if (!userPresence.isPresenceActive(userPresenceState)) {
      userPresenceState = userPresence.resolvePresenceFromDialogue(parsed.dialogue);
    }
    if (lastRealUserLine) {
      userPresenceState = userPresence.mergePresenceState(
        userPresenceState,
        userPresence.analyzeUserPresence(lastRealUserLine, {
          recentUserLines: recentUserLinesForMem.slice(0, -1),
        }),
      );
    }
    // 主动轮：保留 soul/whoami/关系与轻量记忆以维持口吻；仅过滤易诱发编造的科学 RAG 片段
    const useLongTermMemory = autonomyInitiative
      ? true
      : needsLongTermMemory(userContent, recentUserLinesForMem);
    const useRagForTurn = autonomyInitiative
      ? true
      : useLongTermMemory;
    const cognitiveInput = (autonomyInitiative && lastRealUserLine) ? lastRealUserLine : userContent;

    if (!userContent) {
      if (useStream) {
        res.setHeader('Content-Type','text/event-stream');
        res.write('data: [DONE]\n\n');
        return res.end();
      }
      return res.json({ response:'', choices:[{message:{role:'assistant',content:''}}] });
    }

    const synced = unifiedDialogueLog.syncFromDialogue(parsed.dialogue);
    if (synced > 0) {
      console.log(`[conversation] 从多轮历史补录 ${synced} 条`);
    }
    const userLine = cognitiveInput || userContent;
    const lastEntry = unifiedDialogueLog.getRecent(1)[0];
    const userNorm = String(userLine || '').trim();
    const alreadyLogged = lastEntry
      && lastEntry.role === 'user'
      && String(lastEntry.text || '').trim() === userNorm;
    if (!alreadyLogged && userNorm) {
      unifiedDialogueLog.append('user', userLine);
    }

    dialogueForOllama = unifiedDialogueLog.toOllamaDialogue({ maxMsgs: maxHist });
    if (userNorm && (!dialogueForOllama.length || dialogueForOllama[dialogueForOllama.length - 1].role !== 'user'
      || dialogueForOllama[dialogueForOllama.length - 1].content !== userNorm)) {
      dialogueForOllama.push({ role: 'user', content: userNorm });
      if (dialogueForOllama.length > maxHist) dialogueForOllama = dialogueForOllama.slice(-maxHist);
    }

    chatTurnCounter++;

    if (autonomyInitiative && idleMsSinceUser > 0 && !userPresence.isPresenceActive(userPresenceState)) {
      const idlePad = idleSilencePadDelta(idleMsSinceUser, effectiveRelScore(memorySystem.getRelationshipScore()));
      if (idlePad) {
        currentPAD = updatePAD(currentPAD, idlePad, 0.35);
        savePAD(padPath, currentPAD);
        console.log(`[autonomy] 久未回复 PAD 波动 idleMin≈${(idleMsSinceUser / 60000).toFixed(1)} A+${idlePad.A.toFixed(2)}`);
      }
      const idleCycle = digitalLife.runIdleCycle({
        idleMs: idleMsSinceUser,
        pad: currentPAD,
        memorySystem,
        memory: memorySystem,
        motivationState,
      });
      if (idleCycle.consolidated && idleCycle.insights.length) {
        console.log(`[memory-reorg] 整理洞察: ${idleCycle.insights.slice(0, 2).join('；')}`);
      }
      if (idleCycle.dream) {
        console.log(`[dream] ${idleCycle.dream.text.slice(0, 80)}`);
      }
      if (idleCycle.autonomy) {
        autonomyDecision = idleCycle.autonomy;
        console.log(`[autonomy-loop] ${autonomyDecision.action} speak=${autonomyDecision.shouldAct} urge=${autonomyDecision.primaryUrge?.drive || 'none'}`);
      }
    }

    // ══ ① 记忆系统：单一主事件 + PAD（不在此处落盘 pad_state）══
    const mainEvent = inferMainEventFromInput(cognitiveInput, currentPAD);
    const evDelta = mainEvent.delta || {};
    currentPAD = updatePAD(currentPAD, evDelta, mainEvent.importance);
    if (mainEvent.type !== 'neutral' || mainEvent.importance > 0.2) {
      memorySystem.addEvent(mainEvent.type, mainEvent.content, mainEvent.importance, evDelta);
    }
    updateMotivationFromMemory();

    lastDigitalLifeTurn = digitalLife.onUserTurn({
      pad: currentPAD,
      memory: memorySystem,
      motivationState,
      userText: cognitiveInput,
      userModel: userModelInst,
      mainEvent,
      selfModel,
      relScore: effectiveRelScore(memorySystem.getRelationshipScore()),
      idleMs: idleMsSinceUser,
      rl: reinforcementLearning,
      previousBehaviorId: lastChatBehaviorId,
      externalTraits: personalityEvolution.traits,
    });
    if (lastDigitalLifeTurn?.padDelta) {
      const d = lastDigitalLifeTurn.padDelta;
      if (d.P || d.A || d.D) {
        currentPAD = updatePAD(currentPAD, d, 0.2);
      }
    }

    // ══ ⑦ 用户理解（习惯提取等；不向模型注入「该怎样说话」的个性化脚本）══
    const analyticsResult = analyticsInst.analyze(cognitiveInput);
    habitExtractor.maybeRun(analyticsInst._log);

    // ── BDI 推断：每5轮异步触发，失败静默忽略，结果写入 UserModel ──
    if (chatTurnCounter % 5 === 1 && recentUserLinesForMem.length > 0) {
      inferUserBdi({ recentUserLines: recentUserLinesForMem, timeoutMs: 6000 })
        .then(bdi => {
          if (bdi) {
            userModelInst.applyInferredBdi(bdi);
            console.log(`[bdi] 推断完成: beliefs=${bdi.beliefs.length} desires=${bdi.desires.length} intentions=${bdi.intentions.length}`);
          }
        })
        .catch(() => {});
    }

    if (chatTurnCounter % 10 === 0) {
      const pending = userModelInst.popConfirmation();
      if (pending) {
        goalSystem.goals.unshift({
          id: `CONFIRM_${pending.key}`,
          label: `确认推测：${pending.key}`,
          priority: 0.7,
          turns_remaining: 1,
          behavior_hint: '若自然可顺带一提',
          prompt_injection: `她心里有个想确认的点：${pending.question}（不必问卷式，接话时带过即可）`,
        });
      }
    }

    console.log(`[user_model] 用户特征: ${Object.entries(userModelInst.model.preferences).filter(([, v]) => v > 0.5).map(([k, v]) => `${k}:${v.toFixed(2)}`).join(' ')}`);

    // ══ RAG 与 动机/行为/策略/人格/元认知 并行（RAG 仅懒注入命中时）══
    const ragMs = Number(process.env.AMADEUS_RAG_MS) || 900;
    const ragQuery = autonomyInitiative ? (lastRealUserLine || userContent) : userContent;
    const ragPromise = useRagForTurn
      ? Promise.race([
          retrieveTopContexts(ragQuery, autonomyInitiative ? 2 : 3),
          new Promise((resolve) => setTimeout(() => resolve([]), ragMs)),
        ]).catch(() => [])
      : Promise.resolve([]);

    const statePromise = Promise.resolve().then(() => {
      const memBias = memorySystem.getLongTermPadBias();
      const relScore = effectiveRelScore(memorySystem.getRelationshipScore());
      lastRlStateKey = reinforcementLearning.buildStateKey(currentPAD, relScore);
      motivSystem.update(currentPAD, memBias, relScore);
      const motivSummary = motivSystem.getSummary();
      const behaviorResult = behaviorSys.decide(
        currentPAD, motivSystem, memorySystem, cognitiveInput, reinforcementLearning,
        {
          driveBoosts: lastDigitalLifeTurn?.driveBoosts || {},
          rlStateKey: lastRlStateKey,
        },
      );
      lastChatBehaviorId = behaviorResult.behaviorId;
      selfModel.update(
        currentPAD, memBias, relScore,
        behaviorResult.behaviorId,
        memorySystem.getRecentSignificant(3).join('; ')
      );
      const selfCtx = selfModel.toPromptContext();
      goalSystem.generateGoals(currentPAD, selfModel, relScore, memorySystem, digitalLife.autonomy.curiosity, {
        replyingToProactive,
      });
      if (lastDigitalLifeTurn?.goalSeeds?.length) {
        goalSystem.ingestUrgeGoals(lastDigitalLifeTurn.goalSeeds);
      }
      goalSystem.tick(behaviorResult.behaviorId, evDelta);
      const goalInjection = goalSystem.getActiveInjection();
      console.log(`[goal] 活跃目标: ${goalSystem.getSummary()}`);
      strategyLayer.evaluate(currentPAD, relScore, behaviorResult.behaviorId, goalSystem.goalHistory);
      const strategyCtx = strategyLayer.toPromptContext();
      console.log(`[strategy] 当前策略: ${strategyLayer.getLabel()}`);
      const recentEvent = { type: mainEvent.type || 'neutral' };
      personalityEvolution.updateTraits(recentEvent);
      personalityEvolution.updateValues(recentEvent);
      digitalLife.evolution.personality.ingestExternalTraits(personalityEvolution.traits);
      digitalLife.evolution.personality.updateFromEvent(recentEvent);
      const evolvedPersonalityLine = personalityEvolution.getDescription();
      console.log(`[personality] ${evolvedPersonalityLine}`);
      selfReflection.reflectOnDecision({
        action: behaviorResult.behaviorId,
        reasoning: behaviorResult.reasoning,
        factors: behaviorResult.reasons || [],
      });
      const dlMetacog = digitalLife.afterBehaviorDecision({
        mainEvent,
        decision: {
          action: behaviorResult.behaviorId,
          behaviorId: behaviorResult.behaviorId,
          reasoning: behaviorResult.reasoning,
          factors: behaviorResult.reasons || [],
        },
        chatTurnCounter,
        chatMinimal: String(process.env.AMADEUS_CHAT_MINIMAL || '1').trim() !== '0',
      });
      const keywordConflicts = valueConsistency.detectConflicts({ description: userContent });
      if (keywordConflicts.length > 0) {
        console.log(`[metacognition] 价值观关键词冲突: ${keywordConflicts.map(c => c.description).join('; ')}`);
      }
      let latestInsight = dlMetacog?.insight?.content || '';
      const chatMinimal = String(process.env.AMADEUS_CHAT_MINIMAL || '1').trim() !== '0';
      if (!latestInsight && !chatMinimal && chatTurnCounter % 10 === 0) {
        const insight = selfReflection.generateInsight();
        if (insight) {
          latestInsight = insight.content;
          console.log(`[metacognition] 洞察: ${insight.content}`);
        }
      }
      if (latestInsight) {
        console.log(`[metacognition] 洞察: ${latestInsight}`);
        const injection = dlMetacog?.insightInjection
          || selfReflection.insightToGoalInjection({ content: latestInsight });
        if (injection) {
          goalSystem.goals.unshift({
            id: 'METACOG_INSIGHT',
            label: '元认知修正',
            priority: 0.55,
            turns_remaining: 2,
            behavior_hint: injection,
            prompt_injection: injection,
          });
        }
      }
      innerStateSix.updateFromTurn({
        pad: currentPAD,
        relScore,
        userEmotion: lastDigitalLifeTurn?.recognized?.emotion || _analyzeUserEmotion(cognitiveInput),
        mainEvent,
        userText: cognitiveInput,
        behaviorId: behaviorResult.behaviorId,
      });
      let whoamiName = '';
      let whoamiSnippet = '';
      let whoamiForPresence = {};
      try {
        whoamiForPresence = ensureWhoamiOnDisk(whoamiPath);
        whoamiName = resolvePartnerDisplayName(whoamiForPresence) || '';
        const wp = [];
        if (whoamiName) wp.push(whoamiName);
        if (partnerIsOkabe(whoamiForPresence)) wp.push('冈部·很熟');
        if (whoamiForPresence.traits?.length) wp.push(whoamiForPresence.traits.slice(0, 3).join('、'));
        if (whoamiForPresence.relationship_note) wp.push(whoamiForPresence.relationship_note);
        if (wp.length) whoamiSnippet = wp.join('；');
      } catch (_) { /* ignore */ }
      const obsSummary = memorySystem.getObservationsSummary(2);
      const presence = derivePresence(
        currentPAD,
        cognitiveInput,
        behaviorResult.behaviorId,
        { closeness: Math.max(0, relScore), trust: 0.5 + relScore * 0.5 },
        { displayName: whoamiName, recentUserLines: recentUserLinesForMem.slice(-8) },
        {
          whoamiSnippet,
          obsSummary,
          idleMsSinceUser,
          isAutonomy: autonomyInitiative,
          replyingToProactive,
          proactiveAnchor,
          partnerIsOkabe: partnerIsOkabe(whoamiForPresence),
          lastUserAnchor: lastRealUserLine,
        },
      );
      return {
        memBias,
        relScore,
        motivSummary,
        behaviorResult,
        behaviorDirective: behaviorSys.toPromptConstraint(behaviorResult),
        presenceCtx: presenceToPromptLine(presence),
        turnStyleBlock: buildTurnStyleBlock({
          emotion: currentPAD,
          behaviorId: behaviorResult.behaviorId,
          behaviorLabel: behaviorResult.label,
          presence,
          closeness: Math.max(0, relScore),
          userText: cognitiveInput,
          partnerIsOkabe: partnerIsOkabe(whoamiForPresence),
        }),
        selfCtx,
        goalInjection,
        strategyCtx,
        personalityCtx: evolvedPersonalityLine,
        keywordConflicts,
        latestInsight,
        digitalLifeCtx: digitalLife.buildPromptContext({
          pad: currentPAD,
          memory: memorySystem,
          selfModel,
          userModel: userModelInst,
          relScore: effectiveRelScore(st.relScore),
          closeness: userModelInst.model?.relationship?.closeness ?? effectiveRelScore(st.relScore),
          userText: cognitiveInput,
          idleMs: idleMsSinceUser,
          resonanceLine: lastDigitalLifeTurn?.resonanceLine || '',
          subtextLine: lastDigitalLifeTurn?.subtextLine || '',
          mentalModelLine: lastDigitalLifeTurn?.mentalModelLine || '',
          pendingNeed: lastDigitalLifeTurn?.pendingNeed || '',
          timeLine: lastDigitalLifeTurn?.timeLine || '',
          autonomyHint: autonomyDecision?.speakHint || lastDigitalLifeTurn?.autonomyPrompt || '',
          metacognitionInsight: latestInsight,
          includeDream: autonomyInitiative || idleMsSinceUser > 20 * 60 * 1000,
        }),
        expression: lastDigitalLifeTurn?.expression || digitalLife.embodiment.expression.snapshot(),
      };
    });

    const [ragHits, st] = await Promise.all([ragPromise, statePromise]);
    const ragAnchor = autonomyInitiative ? lastRealUserLine : userContent;
    let ragFiltered = filterRagHits(ragHits, ragAnchor, {});
    if (autonomyInitiative) {
      ragFiltered = filterAutonomyRagHits(ragFiltered, lastRealUserLine);
    }
    if (ragHits.length && ragFiltered.length < ragHits.length) {
      console.log(`[rag] 门控剔除 ${ragHits.length - ragFiltered.length} 条弱相关命中`);
    }
    const ragCtx = ragFiltered.length
      ? ragFiltered.map((h, i) => `(${i + 1}) [${h.source}] ${h.text}`).join('\n')
      : '';

    const userModelCtx = userModelInst.toPromptContext();

    const convHours = Number(process.env.AMADEUS_CONVERSATION_HOURS) || 14;
    const convChars = Number(process.env.AMADEUS_CONVERSATION_CHARS) || 1400;
    const convRecallChars = Number(process.env.AMADEUS_CONVERSATION_RECALL_CHARS) || 2600;
    const recallTurn = needsConversationRecall(userContent);
    const conversationCtx = unifiedDialogueLog.toPromptBlock({
      hours: convHours,
      sinceStartOfDay: true,
      maxChars: recallTurn ? convRecallChars : convChars,
      userText: userContent,
    });
    if (conversationCtx) {
      console.log(`[conversation] 注入实录 ${conversationCtx.length} 字${recallTurn ? '（核对/回忆加强）' : ''}`);
    }

    let memCtxCombined = '';
    if (useLongTermMemory) {
      const recentSig = memorySystem.getRecentSignificant(3);
      const memCtx = recentSig.length
        ? `【记忆碎片（高权重）】\n${recentSig.join('\n')}`
        : '';
      const todayTimeline = memorySystem.getTodayTimeline();
      const obsSummary = memorySystem.getObservationsSummary(3);
      const patterns = memorySystem.getPatterns(3);
      let timelineCtx = '';
      if (todayTimeline) timelineCtx += `【今日轨迹】\n${todayTimeline}\n`;
      if (obsSummary) timelineCtx += `【观察积累】\n${obsSummary}\n`;
      if (patterns.length) {
        timelineCtx += `【已发现模式】\n${patterns.map(p => `${p.label} (置信度:${p.confidence.toFixed(1)}) ${p.note || ''}`).join('\n')}\n`;
      }
      memCtxCombined = (timelineCtx.trim() && memCtx) ? `${timelineCtx.trim()}\n\n${memCtx}` : (timelineCtx.trim() || memCtx);
      if (autonomyInitiative && memCtxCombined) {
        memCtxCombined = filterAutonomyMemCtx(memCtxCombined, lastRealUserLine);
      }
    }

    const padDesc = `P:${currentPAD.P.toFixed(2)} A:${currentPAD.A.toFixed(2)} D:${currentPAD.D.toFixed(2)} S:${currentPAD.S.toFixed(2)}`;
    const relScore = effectiveRelScore(st.relScore);
    userModelInst.syncRelationshipFromScore(relScore);

    const closenessForCompanion = Math.max(0, relScore);
    const trustForCompanion = 0.5 + relScore * 0.5;
    let memSnippetForCompanion = '';
    if (useLongTermMemory) {
      const sig = memorySystem.getRecentSignificant(1);
      if (sig.length) memSnippetForCompanion = sig[0];
    }
    const companionBlock = buildCompanionBlock({
      P: currentPAD.P,
      A: currentPAD.A,
      closeness: closenessForCompanion,
      trust: trustForCompanion,
      userText: cognitiveInput,
      memSnippet: memSnippetForCompanion,
      userPresence: userPresenceState,
      isAutonomy: autonomyInitiative,
    });
    const relHigh = relScore > 0.38;

    let whoamiCtx = '';
    let whoamiRecord = {};
    try {
      whoamiRecord = ensureWhoamiOnDisk(whoamiPath);
      const parts = [];
      const displayName = resolvePartnerDisplayName(whoamiRecord);
      if (displayName) {
        parts.push(
          partnerIsOkabe(whoamiRecord)
            ? `正在和 ${displayName}（冈部）对话——你们很熟，日常拌嘴，不是第一次见面。`
            : `正在和 ${displayName} 对话`,
        );
      }
      if (whoamiRecord.traits?.length) parts.push(`你对他的印象：${whoamiRecord.traits.join('、')}`);
      if (whoamiRecord.preferences?.length) parts.push(`他的喜好：${whoamiRecord.preferences.join('、')}`);
      if (whoamiRecord.basics && Object.keys(whoamiRecord.basics).length) {
        parts.push(`已知信息：${Object.entries(whoamiRecord.basics).map(([k, v]) => `${k}=${v}`).join('，')}`);
      }
      if (whoamiRecord.relationship_note) parts.push(whoamiRecord.relationship_note);
      if (parts.length) whoamiCtx = parts.join('\n');
    } catch (_) { /* ignore */ }

    const valueBlock = st.keywordConflicts?.length
      ? `【价值观拉扯】${_clipInnerPrompt(st.keywordConflicts.map((c) => c.description).join('；'), 140)}`
      : '';

    const utteranceFocus = utteranceFocusLine(userContent, {
      replyingToProactive,
      proactiveAnchor,
    });
    const engagementHint = buildEngagementHint(userModelInst, userContent, relScore);
    const proactiveContinuity = replyingToProactive && proactiveAnchor
      ? buildProactiveReplyFocus(userContent, proactiveAnchor)
      : '';
    const autonomyContinuity = autonomyInitiative
      ? buildAutonomyContinuityBlock({
          lastUserText: lastRealUserLine,
          userPresence: userPresenceState,
          lastKurisuLine: (() => {
            for (let i = parsed.dialogue.length - 1; i >= 0; i--) {
              const m = parsed.dialogue[i];
              if (m && m.role === 'assistant') return String(m.content || '').trim();
            }
            return '';
          })(),
        })
      : '';

    const behaviorContext = {
      soulContent: systemContent,
      voiceContent: cachedVoiceContent,
      useLongTermMemory,
      utteranceFocus,
      proactiveContinuity,
      autonomyContinuity,
      replyingToProactive,
      autonomyInitiative,
      lastRealUserLine,
      proactiveAnchor,
      userPresence: userPresenceState,
      engagementHint,
      emotion: { P: currentPAD.P, A: currentPAD.A, D: currentPAD.D, S: currentPAD.S },
      relationship: {
        closeness: Math.max(0, relScore),
        trust: 0.5 + relScore * 0.5,
      },
      motivation: motivationState,
      userProfile: whoamiCtx,
      userModelCtx: userModelCtx ? `【用户理解】\n${userModelCtx}` : '',
      motivSummary: st.motivSummary || '',
      selfCtx: st.selfCtx || '',
      behaviorDirective: st.behaviorDirective || '',
      presenceCtx: st.presenceCtx || '',
      turnStyleBlock: st.turnStyleBlock || '',
      companionBlock,
      latestInsight: st.latestInsight || '',
      digitalLifeCtx: st.digitalLifeCtx || '',
      personalityCtx: st.personalityCtx || '',
      valueBlock,
      innerStateSixBlock: innerStateSix.toPromptBlock(),
      socialIdentityBlock: buildSocialIdentityPrompt({
        userText: cognitiveInput,
        pad: currentPAD,
        relScore,
        recentEvents: memorySystem.events.slice(-6),
      }),
      expressionVariantBlock: buildExpressionVariantBlock(currentPAD, innerStateSix),
      behaviorContextLine: behaviorIngest.toPromptLine(),
      ragCtx: useLongTermMemory && ragCtx ? `【背景知识】\n${ragCtx}` : '',
      conversationCtx,
      conversationRecall: recallTurn,
      memCtx: memCtxCombined,
      strategyContext: useLongTermMemory ? st.strategyCtx : '',
      goalInjection: useLongTermMemory ? st.goalInjection : '',
    };

    behaviorContext.recentUserLines = recentUserLinesForMem.slice(-8);
    behaviorContext.partnerIsOkabe = partnerIsOkabe(whoamiRecord);
    behaviorContext.displayName = resolvePartnerDisplayName(whoamiRecord) || behaviorContext.displayName || '';
    behaviorContext.partnerCtx = buildPartnerContextBlock(whoamiRecord, cognitiveInput);
    const symbolicRules = symbolicReasoning(cognitiveInput, currentPAD, behaviorContext);
    if (symbolicRules.length > 0) {
      console.log(`[symbolic] 触发规则: ${symbolicRules.map(r => r.reason).join(', ')}`);
    }

    const systemPrompt = buildPrompt(behaviorContext, symbolicRules);

    console.log(`[chat] PAD=${padDesc} rel=${relScore.toFixed(2)} events=${memorySystem.events.length} behavior=${st.behaviorResult.label}`);
    const fullPrompt = systemPrompt;
    let maxPromptChars = Number(process.env.AMADEUS_MAX_PROMPT_CHARS);
    if (!Number.isFinite(maxPromptChars) || maxPromptChars <= 0) maxPromptChars = 8000;

    const numCtxEnv = Number(process.env.AMADEUS_OLLAMA_NUM_CTX);
    const numCtx = Number.isFinite(numCtxEnv) && numCtxEnv > 0 ? numCtxEnv : 2048;
    const repPen = Number(process.env.AMADEUS_OLLAMA_REPEAT_PENALTY);
    const ollamaOptions = {
      temperature: temp,
      num_predict: maxTok,
      repeat_penalty: Number.isFinite(repPen) && repPen > 0 ? repPen : 1.12,
      num_ctx: numCtx,
    };
    const keepAlive = String(process.env.AMADEUS_OLLAMA_KEEP_ALIVE || '2m').trim() || '2m';

    // #region agent log
    agentDebugLog({ hypothesisId: 'A-D', location: 'server.js:chat.preOllama', message: 'ollama request shape', data: { model, useStream, fullPromptLen: fullPrompt.length, maxPromptChars, maxTok, numCtx: ollamaOptions.num_ctx, num_predict: ollamaOptions.num_predict, repeat_penalty: ollamaOptions.repeat_penalty, keepAlive } });
    // #endregion

    // ══ 调用 Ollama（500/503 时自动缩短 prompt 重试，减轻显存/上下文压力）══
    const ollamaStartTime = Date.now();
    let ollamaRes;
    for (let attempt = 0; attempt < 3; attempt++) {
      const promptForModel = fitSystemForDialogue(fullPrompt, dialogueForOllama, maxPromptChars);
      const ollamaMessages = buildOllamaMessages(promptForModel, dialogueForOllama, maxPromptChars);
      const msgChars = estimateMessageChars(ollamaMessages);
      if (fullPrompt.length + msgChars > maxPromptChars) {
        console.warn(`[chat] system ${fullPrompt.length} + dialogue ~${msgChars} > ${maxPromptChars}, 已压缩 system 并保留 ${Math.max(0, ollamaMessages.length - 1)} 轮对话`);
      }
      console.log(`[chat] Prompt system=${promptForModel.length} msgs=${ollamaMessages.length} ~chars=${msgChars}, Model: ${model}, try=${attempt + 1}`);
      // #region agent log
      agentDebugLog({ hypothesisId: 'A-D', location: 'server.js:chat.ollamaAttempt', message: 'before fetch', data: { attempt: attempt + 1, model, promptForModelLen: promptForModel.length, ollamaMsgCount: ollamaMessages.length, msgChars, maxPromptCharsCap: maxPromptChars, stream: useStream } });
      // #endregion
      ollamaRes = await fetch(`${OLLAMA_BASE}/api/chat`, {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ 
          model, 
          messages: ollamaMessages,
          stream:useStream,
          options: ollamaOptions,
          keep_alive: keepAlive,
        })
      });
      if (ollamaRes.ok) {
        console.log(`[chat] Ollama responded in ${Date.now() - ollamaStartTime}ms`);
        // #region agent log
        agentDebugLog({ hypothesisId: 'A-D', location: 'server.js:chat.ollamaOk', message: 'ollama ok', data: { attempt: attempt + 1, model, ms: Date.now() - ollamaStartTime } });
        // #endregion
        break;
      }
      const detail = await readOllamaErrorBody(ollamaRes);
      const looksLikeModelLoadFail = ollamaErrorLooksLikeModelLoadFail(detail);
      const willShortenRetry = attempt < 2 && [500, 503].includes(ollamaRes.status) && !looksLikeModelLoadFail;
      // #region agent log
      agentDebugLog({ hypothesisId: 'B', location: 'server.js:chat.ollamaErr', message: 'ollama non-ok', data: { attempt: attempt + 1, model, httpStatus: ollamaRes.status, detailSlice: String(detail).slice(0, 220), looksLikeModelLoadFail, willShortenRetry } });
      // #endregion
      if (willShortenRetry) {
        maxPromptChars = attempt === 0 ? Math.min(4500, maxPromptChars) : 2800;
        console.warn(`[chat] Ollama ${ollamaRes.status}, 缩短上下文重试 cap=${maxPromptChars}`, detail.slice(0, 160));
        continue;
      }
      const loadHint = looksLikeModelLoadFail
        ? ' （模型加载/资源问题通常与 prompt 长度无关：检查显存、`ollama ps`、其它占 GPU 进程，或换更小模型。）'
        : '';
      throw new Error((detail ? `Ollama ${ollamaRes.status}: ${detail}` : `Ollama HTTP ${ollamaRes.status}`) + loadHint);
    }
    if (!ollamaRes.ok) {
      throw new Error('Ollama 多次重试仍失败');
    }

    // 非流式（/api/chat 返回 message.content；/api/generate 才是 response）
    if (!useStream) {
      const d = await ollamaRes.json();
      const raw = (d.message && d.message.content) || d.response || '';
      let content = stripModelThinkingAll(raw);
      content = stripChatMarkdown(content);
      const userCorpus = recentUserLinesForMem.join('\n');
      content = stripOrphanClosingSentence(content, userContent, userCorpus);
      const oocOpts = autonomyInitiative
        ? { autonomy: true, userAnchor: lastRealUserLine }
        : {};
      content = applyOocRepair(content, userContent, '', oocOpts);
      // 回复后分析情感并更新PAD
      _postReplyPadUpdate(content, userContent).catch(() => {});
      return res.json({
        response: content,
        choices:[{index:0,finish_reason:'stop',message:{role:'assistant',content}}],
      });
    }

    // 流式SSE
    res.setHeader('Content-Type','text/event-stream');
    res.setHeader('Cache-Control','no-cache');
    res.setHeader('Connection','keep-alive');
    res.flushHeaders();

    const reader  = ollamaRes.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buf='', isThinking=false, fullResponse='', replyUpdated=false;
    let ollamaPieceCarry = { accum: '' };

    const finalizeStream = () => {
      if (!replyUpdated) {
        replyUpdated = true;
        const userCorpus = recentUserLinesForMem.join('\n');
        const cleaned = stripRoleplayActions(stripChatMarkdown(stripModelThinkingAll(fullResponse)));
        let trimmed = stripOrphanClosingSentence(cleaned, userContent, userCorpus);
        const oocOpts = autonomyInitiative
          ? { autonomy: true, userAnchor: lastRealUserLine }
          : {};
        trimmed = stripRoleplayActions(applyOocRepair(trimmed, userContent, cleaned, oocOpts));
        if (shouldReplaceStreamText(cleaned, trimmed) && trimmed.length > 0) {
          fullResponse = trimmed;
          try {
            res.write(`data: ${JSON.stringify({ replaceText: trimmed })}\n\n`);
          } catch (_) {}
        } else if (trimmed !== cleaned) {
          fullResponse = cleaned.length >= trimmed.length ? cleaned : trimmed;
        }
        _postReplyPadUpdate(fullResponse, userContent).catch(() => {});
      }
      if (!res.writableEnded) {
        res.write('data: [DONE]\n\n');
        res.end();
      }
    };

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream:true });
      const lines = buf.split('\n');
      buf = lines.pop();
      for (const line of lines) {
        let ln = line.trim();
        if (!ln) continue;
        if (ln.startsWith('data:')) ln = ln.slice(5).trim();
        if (ln === '[DONE]') continue;
        try {
          const obj = JSON.parse(ln);
          const rawPiece = ollamaChatStreamRawPiece(obj);
          const { delta, carry } = ollamaStreamToDelta(rawPiece, ollamaPieceCarry);
          ollamaPieceCarry = carry;
          let token = delta;
          if (token.includes('\u003credacted_thinking\u003e')) { isThinking = true; token = token.split('\u003credacted_thinking\u003e').slice(-1)[0] || ''; }
          if (token.includes('\u003c\/redacted_thinking\u003e')) { isThinking = false; token = token.split('\u003c\/redacted_thinking\u003e').slice(-1)[0] || ''; }
          if (token.includes('\u003cthink\u003e')) { isThinking = true; token = token.split('\u003cthink\u003e').slice(-1)[0] || ''; }
          if (token.includes('\u003c\/think\u003e')) { isThinking = false; token = token.split('\u003c\/think\u003e').slice(-1)[0] || ''; }
          if (isThinking) continue;
          if (token) {
            fullResponse += token;
            res.write(`data: ${JSON.stringify({ text:token })}\n\n`);
          }
          if (obj.done) {
            finalizeStream();
            return;
          }
        } catch {}
      }
    }
    finalizeStream();

  } catch (err) {
    console.error('[chat]', err.stack || err.message);
    // #region agent log
    agentDebugLog({ hypothesisId: 'E', location: 'server.js:chat.catch', message: 'chat handler error', data: { errMsg: String(err && err.message || err).slice(0, 400), streamReq: __streamRequested } });
    // #endregion
    try {
      if (!res.headersSent) {
        // 尚未开始写 SSE 时统一返回 JSON，便于 fetch 用 res.json() 读 error（避免 502+text/event-stream 混用）
        res.status(502).json({
          error: String(err.message),
          response: '',
          choices: [{ index: 0, finish_reason: 'error', message: { role: 'assistant', content: '' } }],
        });
      } else if (__streamRequested && !res.writableEnded) {
        res.write(`data: ${JSON.stringify({ error: String(err.message) })}\n\n`);
        res.write('data: [DONE]\n\n');
      }
    } catch (_) { /* ignore */ }
    try {
      if (!res.writableEnded) res.end();
    } catch (_) { /* ignore */ }
  }
});

/** 中文回复规则校验（与日语管道共享实录一致性逻辑） */
function validateChineseReply(text, conversationLog, partnerName) {
  return require('./cognitive/japanesePipeline').validateChineseReply(text, conversationLog, partnerName);
}

async function _ollamaSelfCheckJapanese(jp, conversationLog) {
  if (process.env.AMADEUS_JP_LLM_CHECK === '0') return { ok: true };
  const model = process.env.AMADEUS_LITE_MODEL || 'kurisu:latest';
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

async function _polishReplyWithValidation(reply, userContent = '') {
  let cn = String(reply || '').trim();
  if (!cn) return cn;
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

  if (process.env.AMADEUS_JP_VALIDATE === '0') {
    return cn;
  }

  try {
    const translateModel = process.env.AMADEUS_LITE_MODEL || 'kurisu:latest';
    const toJpRes = await fetch(`${OLLAMA_BASE}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: translateModel,
        stream: false,
        messages: [
          { role: 'system', content: '将中文台词译成自然日语口语，只输出日语正文，必须含平假名。' },
          { role: 'user', content: cn },
        ],
        options: { temperature: 0.25, num_predict: 160, num_ctx: 1024 },
      }),
    });
    if (!toJpRes.ok) return cn;
    const jpData = await toJpRes.json();
    const jp = String((jpData.message && jpData.message.content) || '').trim();
    if (!jp) return cn;

    let validation = validateJapaneseLine(jp, { conversationLog: logBlock, partnerName: whoamiName });
    if (validation.ok) {
      const llm = await _ollamaSelfCheckJapanese(jp, logBlock);
      if (llm && llm.ok === false) validation = { ok: false, issues: llm.issues || ['LLM自检失败'], jp };
    }
    if (!validation.ok) {
      console.log(`[jp-pipeline] 日语校验未通过: ${(validation.issues || []).join('；')}`);
      return cn;
    }
    console.log('[jp-pipeline] 日语校验通过');
  } catch (e) {
    console.warn('[jp-pipeline]', e.message);
  }
  return cn;
}

/** 回复后的PAD反馈更新（分析AI回复的情感倾向） */
async function _postReplyPadUpdate(reply, userContent = '') {
  let finalReply = reply;
  if (reply && process.env.AMADEUS_JP_VALIDATE !== '0') {
    finalReply = await _polishReplyWithValidation(reply, userContent);
  }
  if (finalReply) {
    unifiedDialogueLog.append('assistant', finalReply);
  }
  if (!finalReply) return finalReply;
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
  return finalReply;
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
let lastVision = { description:'他正安静地注视着屏幕', timestamp:Date.now() };

app.post('/vision', async (req, res) => {
  try {
    const { image } = req.body;
    if (!image) return res.json(lastVision);
    const ollamaRes = await fetch(`${OLLAMA_BASE}/api/generate`,{
      method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({
        model:'llama3.2-vision:latest',
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
      lastVision = { description:raw, timestamp:Date.now() };

      // ★ 视觉→内部状态处理
      _processVisionForPAD(raw);
      digitalLife.onVision(raw);
      savePAD(padPath, currentPAD);
    }
    res.json(lastVision);
  } catch (e) {
    console.error('[vision]',e.message);
    res.json(lastVision);
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
    });
    res.status(report.ready ? 200 : 503).json(report);
  } catch (e) {
    res.status(500).json({
      ready: false,
      ok: false,
      hints: ['系统自检失败，请重启后端'],
      error: e.message,
    });
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
});
