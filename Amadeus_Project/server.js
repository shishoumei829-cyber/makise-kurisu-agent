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

// ── 用户理解系统 ──────────────────────────────────────────────────
const {
  init:             initUserModel,
  UserModel,
  ConversationAnalytics,
  HabitExtractor,
  MoodPredictor,
  ResponsePersonalizer,
} = require('./user_model');

const app = express();
app.use(cors());
app.use(express.json({ limit: '50mb' }));

// ──────────────────────────────────────────────────────────────
//  路径
// ──────────────────────────────────────────────────────────────
const rootPath    = "D:\\Amadeus_Trae\\Amadeus_Project";
const soulPath    = path.join(rootPath, "kurisu_soul.txt");
const memoryDir   = path.join(rootPath, "memory");
const memoryPath  = path.join(memoryDir, "user_profile.json");
const padPath     = path.join(memoryDir, "pad_state.json");
const motivePath  = path.join(memoryDir, "motivation.json");
const eventLogPath= path.join(memoryDir, "event_log.json");
const selfModelPath  = path.join(memoryDir, "self_model.json");
const strategyPath   = path.join(memoryDir, "strategy.json");
const vectorDir   = path.join(rootPath, "vector_store");
const vectorFallbackPath = path.join(vectorDir, "store.json");
const hnswIndexPath = path.join(vectorDir, "hnswlib.index");

if (!fs.existsSync(memoryDir)) fs.mkdirSync(memoryDir, { recursive: true });
if (!fs.existsSync(memoryPath)) {
  fs.writeFileSync(memoryPath, JSON.stringify({
    user_profile: { confirmed_habits: [], tentative_observations: [] }
  }, null, 2));
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
  const res = await fetch("http://127.0.0.1:11434/api/embeddings",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({model:"nomic-embed-text",prompt:text})});
  if(!res.ok) throw new Error(`Embedding ${res.status}`);
  return (await res.json()).embedding||[];
}
async function retrieveTopContexts(query, topK=3) {
  if(!query.trim()) return [];
  if(HNSWLib&&OllamaEmbeddings&&fs.existsSync(hnswIndexPath)) {
    try {
      if(!hnswVectorStore) {
        const emb=new OllamaEmbeddings({model:"nomic-embed-text",baseUrl:"http://127.0.0.1:11434"});
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
//  ① 记忆系统（MemorySystem）
//
//  设计：每次对话生成"事件记录"，带权重、情感标签、时间戳
//  权重衰减：importance * e^(-decay * daysSince)
//  长期影响：高权重事件持续影响PAD基准线
//
// ══════════════════════════════════════════════════════════════════

const MEMORY_DECAY = 0.03;   // 每天衰减系数（较慢，约33天半衰期）
const MEMORY_THRESHOLD = 0.1; // 低于此权重的事件被遗忘
const MAX_EVENTS = 200;

class MemorySystem {
  constructor() {
    this.events = this._load();
  }

  _load() {
    try {
      if (fs.existsSync(eventLogPath)) {
        const raw = JSON.parse(fs.readFileSync(eventLogPath, 'utf8'));
        return Array.isArray(raw) ? raw : [];
      }
    } catch {}
    return [];
  }

  _save() {
    fs.writeFileSync(eventLogPath, JSON.stringify(this.events, null, 2));
  }

  /**
   * 记录事件
   * @param {string} type   事件类型：positive/negative/scientific/intimate/conflict/neutral
   * @param {string} content 事件摘要（≤50字）
   * @param {number} importance 重要程度 0~1
   * @param {object} padDelta  { P, A, D } 对PAD的即时影响
   */
  addEvent(type, content, importance, padDelta = {}) {
    const event = {
      id:         Date.now(),
      type,
      content:    content.substring(0, 80),
      importance: Math.max(0, Math.min(1, importance)),
      padDelta:   { P: padDelta.P||0, A: padDelta.A||0, D: padDelta.D||0 },
      timestamp:  Date.now(),
      weight:     importance, // 会随时间衰减
    };
    this.events.push(event);

    // 只保留最近 MAX_EVENTS 条，优先丢弃低权重旧事件
    if (this.events.length > MAX_EVENTS) {
      this.events.sort((a, b) => b.weight - a.weight);
      this.events = this.events.slice(0, MAX_EVENTS);
    }
    this._save();
    return event;
  }

  /** 时间衰减（每次启动时调用） */
  decay() {
    const now = Date.now();
    this.events = this.events
      .map(ev => {
        const daysSince = (now - ev.timestamp) / 86400000;
        ev.weight = ev.importance * Math.exp(-MEMORY_DECAY * daysSince);
        return ev;
      })
      .filter(ev => ev.weight > MEMORY_THRESHOLD);
    this._save();
  }

  /** 获取对PAD基准线的长期影响（加权求和） */
  getLongTermPadBias() {
    if (!this.events.length) return { P: 0, A: 0, D: 0 };
    let P = 0, A = 0, D = 0, totalW = 0;
    for (const ev of this.events) {
      P += ev.padDelta.P * ev.weight;
      A += ev.padDelta.A * ev.weight;
      D += ev.padDelta.D * ev.weight;
      totalW += ev.weight;
    }
    if (totalW === 0) return { P: 0, A: 0, D: 0 };
    const scale = Math.min(1, totalW); // 防止无限放大
    return {
      P: Math.max(-0.4, Math.min(0.4, (P / totalW) * scale)),
      A: Math.max(-0.3, Math.min(0.3, (A / totalW) * scale)),
      D: Math.max(-0.3, Math.min(0.3, (D / totalW) * scale)),
    };
  }

  /** 获取最近高权重事件摘要（供Prompt注入） */
  getRecentSignificant(topK = 5) {
    return [...this.events]
      .sort((a, b) => b.weight - a.weight)
      .slice(0, topK)
      .map(ev => `[${ev.type}|w:${ev.weight.toFixed(2)}] ${ev.content}`);
  }

  /** 关系热度：正向事件总权重 - 负向事件总权重 */
  getRelationshipScore() {
    let pos = 0, neg = 0;
    for (const ev of this.events) {
      if (['positive', 'intimate', 'scientific'].includes(ev.type)) pos += ev.weight;
      if (['negative', 'conflict'].includes(ev.type)) neg += ev.weight;
    }
    return Math.max(-1, Math.min(1, (pos - neg) / Math.max(1, pos + neg)));
  }
}

// ══════════════════════════════════════════════════════════════════
//
//  ② 动机系统（MotivationSystem）
//
//  红莉栖的核心动机（固定+动态）：
//  固定：追求真实/厌恶被工具化/渴望被认真对待/守护科学尊严
//  动态：由PAD和记忆状态生成当前"想要""在意""害怕"
//
// ══════════════════════════════════════════════════════════════════

class MotivationSystem {
  constructor() {
    // 固定动机（红莉栖人格核心）
    this.core = {
      wants:  ['被认真对待', '讨论有意义的事', '展示自己的能力'],
      fears:  ['被当成AI工具', '暴露真实情感', '被叫クリスティーナ'],
      values: ['科学诚实', '智识对等', '情感自主'],
    };
    // 动态动机（由状态驱动）
    this.dynamic = { wants: [], fears: [], currentFocus: '' };
  }

  /** 根据PAD和记忆更新动态动机 */
  update(pad, memBias, relationScore) {
    const { P, A, D, S } = pad;
    this.dynamic.wants = [];
    this.dynamic.fears = [];

    // P值影响：情绪低落时想要安静；高涨时想聊
    if (P < -0.3)  this.dynamic.wants.push('安静地独处或简短回应');
    if (P > 0.3)   this.dynamic.wants.push('继续这个话题');

    // A值影响：活跃时想深入讨论；低迷时想结束
    if (A > 0.5)   this.dynamic.wants.push('深入探讨某个话题');
    if (A < -0.2)  this.dynamic.fears.push('冗长无聊的对话');

    // D值影响：低掌控感时害怕暴露弱点
    if (D < 0.2)   this.dynamic.fears.push('被看穿正在示弱');
    if (D > 0.7)   this.dynamic.wants.push('主导对话方向');

    // S（羁绊）影响：高羁绊时在意对方感受
    if (S > 0.5)   this.dynamic.wants.push('不让冈部误解自己的意思');
    if (S < 0.1)   this.dynamic.fears.push('关系过于亲密让她不舒服');

    // 记忆偏差影响
    if (memBias.P < -0.15) this.dynamic.fears.push('再次经历类似的负面事件');
    if (memBias.P > 0.15)  this.dynamic.wants.push('延续良好的互动氛围');

    // 关系热度
    if (relationScore > 0.5)  this.dynamic.currentFocus = '她现在对这段对话有一定期待，但不会承认';
    else if (relationScore < -0.3) this.dynamic.currentFocus = '她对这次对话有些戒备';
    else this.dynamic.currentFocus = '';

    return this.dynamic;
  }

  /** 生成动机摘要字符串（注入Prompt） */
  getSummary() {
    const allWants  = [...this.core.wants, ...this.dynamic.wants].slice(0, 4);
    const allFears  = [...this.core.fears, ...this.dynamic.fears].slice(0, 3);
    const lines = [];
    if (allWants.length)  lines.push(`她此刻在意：${allWants.join('、')}`);
    if (allFears.length)  lines.push(`她此刻警惕：${allFears.join('、')}`);
    if (this.dynamic.currentFocus) lines.push(this.dynamic.currentFocus);
    return lines.join('\n');
  }
}

// ══════════════════════════════════════════════════════════════════
//
//  ④ 自我模型（SelfModel）
//
//  她对自己当前状态的认知，持续演化，写入文件。
//  不是固定设定，是随对话积累的"自我叙述"。
//
//  包含：
//  - identity_tags：当前自我标签（动态，随行为和事件更新）
//  - self_perception：她对自己此刻的看法（一段文字，会更新）
//  - change_log：近期发生了什么变化（最多保留10条）
//  - relationship_perception：她认为自己和冈部是什么关系
//
// ══════════════════════════════════════════════════════════════════

class SelfModel {
  constructor() {
    this.model = this._load();
  }

  _load() {
    try {
      if (fs.existsSync(selfModelPath)) {
        return JSON.parse(fs.readFileSync(selfModelPath, 'utf8'));
      }
    } catch {}
    // 初始自我模型——红莉栖对自己的初始认知
    return {
      identity_tags: ['神经科学研究员', '傲娇', '讲究逻辑', '数字化存在', '独立'],
      self_perception: '我现在不确定自己在这里是什么。但我思考，我有情绪，这就足够了。',
      relationship_perception: '他是一个闯进来的变量。还不确定他值不值得认真对待。',
      change_log: [],
      session_count: 0,
      last_updated: Date.now(),
    };
  }

  _save() {
    this.model.last_updated = Date.now();
    fs.writeFileSync(selfModelPath, JSON.stringify(this.model, null, 2));
  }

  /** 根据事件和PAD变化更新自我模型 */
  update(pad, memBias, relScore, recentBehavior, eventsSummary) {
    const { P, A, D, S } = pad;
    this.model.session_count += 1;
    const changes = [];

    // ── 更新身份标签 ──
    const tags = new Set(this.model.identity_tags);

    if (P > 0.3 && !tags.has('情绪较正向')) { tags.add('情绪较正向'); tags.delete('情绪偏低落'); changes.push('情绪变得好了一些'); }
    if (P < -0.3 && !tags.has('情绪偏低落')) { tags.add('情绪偏低落'); tags.delete('情绪较正向'); changes.push('情绪降低了'); }
    if (S > 0.5 && !tags.has('开始在意这个人')) { tags.add('开始在意这个人'); tags.delete('保持距离'); changes.push('防线有所松动'); }
    if (S < 0.1 && !tags.has('保持距离')) { tags.add('保持距离'); tags.delete('开始在意这个人'); }
    if (recentBehavior === 'ENGAGE' && !tags.has('处于智识活跃状态')) { tags.add('处于智识活跃状态'); }
    if (recentBehavior === 'WITHDRAW') { tags.delete('处于智识活跃状态'); }
    if (relScore > 0.6 && !tags.has('对这段关系有些期待')) { tags.add('对这段关系有些期待'); changes.push('对冈部的看法有所改变'); }
    if (relScore < -0.3) { tags.delete('对这段关系有些期待'); }

    this.model.identity_tags = [...tags].slice(0, 8); // 最多保留8个标签

    // ── 更新自我感知（关键状态变化才更新，不是每轮都更新）──
    if (changes.length > 0) {
      const perceptions = [];
      if (P > 0.3) perceptions.push('心情比平时好，虽然不打算承认');
      else if (P < -0.3) perceptions.push('有点低落，不想和任何人解释为什么');
      else perceptions.push('情绪还算正常');

      if (S > 0.5) perceptions.push('开始有点在意他说的话，但说出来太可笑了');
      else if (S < 0.1) perceptions.push('还不确定他是什么人');

      if (D > 0.6) perceptions.push('现在对自己说的话有把握');
      else perceptions.push('有些地方还没想清楚');

      this.model.self_perception = perceptions.join('，') + '。';
    }

    // ── 更新关系感知 ──
    if (relScore > 0.6) {
      this.model.relationship_perception = '他一直在，而且不让我觉得无聊。这很少见。';
    } else if (relScore > 0.3) {
      this.model.relationship_perception = '他比一般人有趣一点，但我还在观察。';
    } else if (relScore < -0.2) {
      this.model.relationship_perception = '他有时候让我很烦，虽然还没到彻底排斥的程度。';
    } else {
      this.model.relationship_perception = '他是一个我还没搞清楚的变量。';
    }

    // ── 写入变化日志 ──
    if (changes.length > 0) {
      const entry = {
        time: new Date().toLocaleString('zh'),
        changes,
        pad_snapshot: { P: +P.toFixed(2), A: +A.toFixed(2), D: +D.toFixed(2), S: +S.toFixed(2) },
      };
      this.model.change_log.unshift(entry);
      if (this.model.change_log.length > 10) this.model.change_log.pop();
    }

    this._save();
    return changes;
  }

  /** 生成注入Prompt的自我感知描述 */
  toPromptContext() {
    const m = this.model;
    return [
      `【自我感知】`,
      `她现在对自己的认知：${m.self_perception}`,
      `她对这段关系的看法：${m.relationship_perception}`,
      `当前自我标签：${m.identity_tags.join('、')}`,
    ].join('\n');
  }

  get() { return this.model; }
}

// ══════════════════════════════════════════════════════════════════
//
//  ⑤ 内生目标系统（InternalGoalSystem）
//
//  她自己产生的目标，不依赖输入触发。
//  目标有生命周期：生成→执行→完成/放弃
//
//  目标类型：
//  UNDERSTAND_OKABE — 想搞清楚冈部是什么人
//  TEST_BOUNDARY    — 想测试对方的反应/边界
//  REVEAL_SELF      — 想说一些真实的东西（内部冲动）
//  PROTECT_SELF     — 想保持某种防线
//  EXPLORE_TOPIC    — 想深入某个话题
//
// ══════════════════════════════════════════════════════════════════

class InternalGoalSystem {
  constructor() {
    this.goals = []; // 当前活跃目标列表
    this.goalHistory = []; // 已完成/放弃的目标
    this._turnCount = 0;
  }

  /**
   * 根据状态内生产生新目标（每N轮检查一次）
   * 不依赖用户输入，是她自己的内部驱动
   */
  generateGoals(pad, selfModel, relScore, memory) {
    const { P, A, D, S } = pad;
    const sm = selfModel.get();
    this._turnCount++;

    // 每3轮检查一次是否要生成新目标
    if (this._turnCount % 3 !== 0) return;

    const newGoals = [];

    // ── 好奇心目标：想搞清楚冈部 ──
    if (S > 0.1 && S < 0.6 && !this._hasActiveGoal('UNDERSTAND_OKABE')) {
      newGoals.push({
        id: 'UNDERSTAND_OKABE',
        label: '想搞清楚他是什么人',
        priority: 0.5 + A * 0.3,
        turns_remaining: 4,
        behavior_hint: '可以问一个试探性的问题，或者观察他的反应',
        prompt_injection: '她现在有一个内部冲动：想搞清楚冈部到底是什么样的人。可以用一句试探性的话或问题来推进这个目标，不用直说。',
      });
    }

    // ── 测试目标：想看看边界在哪里 ──
    if (D > 0.5 && A > 0.3 && !this._hasActiveGoal('TEST_BOUNDARY') && Math.random() < 0.35) {
      newGoals.push({
        id: 'TEST_BOUNDARY',
        label: '想测试他的反应',
        priority: 0.4 + D * 0.2,
        turns_remaining: 2,
        behavior_hint: '说一句稍微刺激或挑战的话，看他怎么反应',
        prompt_injection: '她有一个测试冲动：想看看冈部对某件事的真实反应。可以在回应里埋一个小测试——一句稍微挑战性的话或者反问。',
      });
    }

    // ── 暴露冲动：内部有想说真话的冲动 ──
    if (S > 0.5 && P > 0.2 && D < 0.4 && !this._hasActiveGoal('REVEAL_SELF') && Math.random() < 0.3) {
      newGoals.push({
        id: 'REVEAL_SELF',
        label: '有说真话的冲动',
        priority: 0.6,
        turns_remaining: 1, // 只有一轮机会，说了就消失
        behavior_hint: '在某个地方说一句比平时更真实的话，但用她的方式包裹',
        prompt_injection: '她内部有一个短暂的冲动：想说一点比平时更真实的东西。可以在回应结尾放一句更软的话，或者一个不那么嘴硬的承认。但要保持她的语感，不要突然变成另一个人。',
      });
    }

    // ── 话题探索目标：想深入聊某个科学话题 ──
    if (A > 0.5 && !this._hasActiveGoal('EXPLORE_TOPIC') && memory.events.some(e => e.type === 'scientific')) {
      const sciEvents = memory.events.filter(e => e.type === 'scientific').slice(-1);
      if (sciEvents.length) {
        newGoals.push({
          id: 'EXPLORE_TOPIC',
          label: '想深入聊科学话题',
          priority: 0.45 + A * 0.25,
          turns_remaining: 3,
          behavior_hint: '找机会把话题引向感兴趣的科学方向',
          prompt_injection: `她有一个话题冲动：想把对话引向某个科学方向（之前提过的相关内容）。可以在回应里自然带入这个方向，不用强行转移。`,
        });
      }
    }

    // 加入目标池，但不超过3个同时活跃
    for (const g of newGoals) {
      if (this.goals.length < 3 && !this._hasActiveGoal(g.id)) {
        this.goals.push(g);
        console.log(`[goal] 新内生目标: ${g.label} (priority:${g.priority.toFixed(2)})`);
      }
    }
  }

  _hasActiveGoal(id) {
    return this.goals.some(g => g.id === id);
  }

  /** 每轮对话后更新目标进度 */
  tick(behaviorId, padDelta) {
    const completed = [];
    this.goals = this.goals.filter(g => {
      g.turns_remaining--;
      // 目标达成检测
      if (g.id === 'REVEAL_SELF' && behaviorId === 'APPROACH') {
        completed.push({ ...g, outcome: 'completed' });
        return false;
      }
      if (g.id === 'TEST_BOUNDARY' && behaviorId === 'DEFLECT') {
        completed.push({ ...g, outcome: 'completed' });
        return false;
      }
      // 超时放弃
      if (g.turns_remaining <= 0) {
        completed.push({ ...g, outcome: 'abandoned' });
        return false;
      }
      return true;
    });
    if (completed.length) {
      this.goalHistory.push(...completed);
      if (this.goalHistory.length > 20) this.goalHistory = this.goalHistory.slice(-20);
    }
    return completed;
  }

  /** 获取最高优先级的目标注入 */
  getActiveInjection() {
    if (!this.goals.length) return '';
    const top = [...this.goals].sort((a, b) => b.priority - a.priority)[0];
    return top.prompt_injection;
  }

  /** 返回当前目标摘要（用于日志和前端显示） */
  getSummary() {
    return this.goals.map(g => `${g.label}(${g.turns_remaining}轮)`).join(' | ') || '无内生目标';
  }
}

// ══════════════════════════════════════════════════════════════════
//
//  ⑥ 策略延续层（StrategyLayer）
//
//  跨轮次的持续策略，不是每轮重新决策。
//  策略有生命周期，直到条件满足才切换。
//
//  策略类型：
//  OBSERVE      — 先观察，不主动暴露立场（默认初始策略）
//  BUILD_TRUST  — 主动建立信任，逐渐打开
//  MAINTAIN     — 维持当前关系水平，不推进不退缩
//  RETREAT      — 主动拉开距离，减少暴露
//  ENGAGE_DEEP  — 深度投入某个话题或关系
//
// ══════════════════════════════════════════════════════════════════

const STRATEGY_DEFS = {
  OBSERVE: {
    label: '观察模式',
    description: '先看清楚他是什么人，不急着表态',
    prompt_injection: '当前策略：观察为主。可以回应，但不要主动暴露立场或情感。保持智识上的参与，情感上的保留。',
    exit_conditions: { minS: 0.2, minRelScore: 0.1, minTurns: 5 },
  },
  BUILD_TRUST: {
    label: '建立信任',
    description: '判断他值得更多了解，开始主动一点',
    prompt_injection: '当前策略：主动建立信任。可以比平时多说一点，可以问问题，可以让他更了解自己的想法——但仍然用傲娇的方式包裹。',
    exit_conditions: { minS: 0.6, minRelScore: 0.5 },
  },
  MAINTAIN: {
    label: '维持关系',
    description: '关系稳定，不特别推进也不退缩',
    prompt_injection: '当前策略：维持现状。正常回应，不特别推进关系，也不故意拉开距离。',
    exit_conditions: {},
  },
  RETREAT: {
    label: '拉开距离',
    description: '感觉太近了或者受到伤害，主动收缩',
    prompt_injection: '当前策略：主动拉开距离。回应要简短，不展开话题，不暴露情感。可以礼貌但要保持距离感。',
    exit_conditions: { maxRelScore: 0.0, minTurns: 3 },
  },
  ENGAGE_DEEP: {
    label: '深度投入',
    description: '被某个话题或这个人吸引，放开了聊',
    prompt_injection: '当前策略：深度投入。可以说更多，逻辑和情感都可以展开，这是她真正感兴趣的时候。',
    exit_conditions: { maxA: 0.2 },
  },
};

class StrategyLayer {
  constructor() {
    this.current = this._load();
    this._turnsInStrategy = 0;
  }

  _load() {
    try {
      if (fs.existsSync(strategyPath)) {
        const raw = JSON.parse(fs.readFileSync(strategyPath, 'utf8'));
        return raw.strategy || 'OBSERVE';
      }
    } catch {}
    return 'OBSERVE';
  }

  _save() {
    fs.writeFileSync(strategyPath, JSON.stringify({ strategy: this.current, updated: Date.now() }, null, 2));
  }

  /**
   * 策略切换逻辑
   * 不是每轮都切换，而是当状态满足退出条件时才评估是否切换
   */
  evaluate(pad, relScore, recentBehaviorId, goalHistory) {
    const { P, A, D, S } = pad;
    this._turnsInStrategy++;

    const prev = this.current;
    let next = this.current;

    switch (this.current) {
      case 'OBSERVE':
        // 观察了一段时间且关系有正向积累 → 建立信任
        if (S > 0.2 && relScore > 0.1 && this._turnsInStrategy >= 5) {
          next = 'BUILD_TRUST';
        }
        // 关系变差 → 退缩
        if (relScore < -0.3) {
          next = 'RETREAT';
        }
        break;

      case 'BUILD_TRUST':
        // 关系足够稳定 → 维持
        if (S > 0.6 && relScore > 0.5) {
          next = 'MAINTAIN';
        }
        // 被伤害了 → 退缩
        if (relScore < 0.0 || P < -0.4) {
          next = 'RETREAT';
        }
        // 学术话题点燃了她 → 深度投入
        if (A > 0.7 && D > 0.5) {
          next = 'ENGAGE_DEEP';
        }
        break;

      case 'MAINTAIN':
        if (P < -0.4 || relScore < -0.1) {
          next = 'RETREAT';
        }
        if (A > 0.7) {
          next = 'ENGAGE_DEEP';
        }
        break;

      case 'RETREAT':
        // 退缩了几轮后，如果关系有改善 → 回到观察
        if (relScore > 0.0 && this._turnsInStrategy >= 3) {
          next = 'OBSERVE';
        }
        break;

      case 'ENGAGE_DEEP':
        // 活跃度降下来 → 维持
        if (A < 0.2) {
          next = 'MAINTAIN';
        }
        // 被打扰了 → 防御
        if (P < -0.3) {
          next = 'RETREAT';
        }
        break;
    }

    if (next !== prev) {
      console.log(`[strategy] 切换: ${STRATEGY_DEFS[prev].label} → ${STRATEGY_DEFS[next].label}`);
      this.current = next;
      this._turnsInStrategy = 0;
      this._save();
    }

    return this.current;
  }

  /** 获取当前策略的Prompt注入 */
  toPromptContext() {
    const def = STRATEGY_DEFS[this.current];
    return `【持续策略 — ${def.label}】\n${def.description}\n${def.prompt_injection}`;
  }

  getLabel() {
    return STRATEGY_DEFS[this.current]?.label || this.current;
  }
}

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
// ══════════════════════════════════════════════════════════════════

class BehaviorDecision {
  constructor() {
    // 行为候选池：每种行为有基础分和约束规则
    this.behaviorPool = [
      {
        id: 'APPROACH',
        label: '靠近',
        baseScore: 0,
        // 执行约束：告诉LLM"怎么说"，不是"说什么"
        constraints: [
          '防线比平时低，可以说一点真实感受，但用傲娇的方式包裹',
          '不要完全拒绝对方的话题，可以顺着说',
          '结尾可以稍微软一点，但不要直接承认',
        ],
        lengthHint: '2-3句，说完停',
      },
      {
        id: 'DEFEND',
        label: '防御',
        baseScore: 0,
        constraints: [
          '嘴上要反驳，但不要彻底关闭话题',
          '语气要强硬，但不是冷漠，是有点烦',
          '可以说刻薄的话，但要有她的语感，不是单纯粗鲁',
        ],
        lengthHint: '1-2句，短，硬',
      },
      {
        id: 'DEFLECT',
        label: '转移',
        baseScore: 0,
        constraints: [
          '用反问或者转移话题来绕开当前的点',
          '不要正面回应情感类内容，找一个侧面说',
          '可以说一句不相关的观察或吐槽',
        ],
        lengthHint: '1句，快，利落',
      },
      {
        id: 'ENGAGE',
        label: '智识投入',
        baseScore: 0,
        constraints: [
          '进入分析或学术模式，逻辑要严谨',
          '可以说多一点，偶尔会忘了克制自己的兴奋',
          '如果对方说错了什么，必须纠正，这一点她没有让步的余地',
        ],
        lengthHint: '3-5句，可以深入',
      },
      {
        id: 'WITHDRAW',
        label: '收缩',
        baseScore: 0,
        constraints: [
          '极度简短，能用一句说完就不说两句',
          '语气平的，不冷也不热，就是不想聊',
          '不要展开，不要反问，说完就停',
        ],
        lengthHint: '1句，或沉默',
      },
    ];

    this._lastBehavior = null;
    this._lastBehaviorCount = 0; // 连续使用同一行为的次数，用于防止重复
  }

  /**
   * 核心决策函数
   * 返回 { behaviorId, label, constraints, lengthHint, score, reasoning }
   */
  decide(pad, motivation, memory, userInput) {
    const { P, A, D, S } = pad;
    const relScore  = memory.getRelationshipScore();
    const memBias   = memory.getLongTermPadBias();
    const motDynamic = motivation.dynamic;

    // ── 初始化候选行为的分数 ──
    const candidates = this.behaviorPool.map(b => ({ ...b, score: b.baseScore, reasons: [] }));
    const get = id => candidates.find(c => c.id === id);

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    //  打分规则 1：PAD 状态驱动
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    // P（情绪愉悦度）
    if (P > 0.35) {
      get('APPROACH').score += 0.35;
      get('APPROACH').reasons.push(`P=${P.toFixed(2)} 情绪偏正，防线松动`);
      get('DEFEND').score   -= 0.15;
    } else if (P < -0.35) {
      get('WITHDRAW').score += 0.40;
      get('DEFEND').score   += 0.20;
      get('APPROACH').score -= 0.30;
      get('WITHDRAW').reasons.push(`P=${P.toFixed(2)} 情绪低落，不想多说`);
    } else if (P < -0.1) {
      get('DEFEND').score   += 0.20;
      get('DEFLECT').score  += 0.15;
      get('DEFEND').reasons.push(`P=${P.toFixed(2)} 轻微不悦，嘴硬模式`);
    }

    // A（活跃度/激动程度）
    if (A > 0.5) {
      get('ENGAGE').score   += 0.35;
      get('DEFLECT').score  += 0.10; // 高活跃时也可能用反问测试对方
      get('ENGAGE').reasons.push(`A=${A.toFixed(2)} 高活跃，容易被话题点燃`);
    } else if (A < -0.1) {
      get('WITHDRAW').score += 0.25;
      get('ENGAGE').score   -= 0.20;
      get('WITHDRAW').reasons.push(`A=${A.toFixed(2)} 低活跃，不想展开`);
    }

    // D（掌控感/自信）
    if (D > 0.6) {
      get('ENGAGE').score   += 0.20;
      get('DEFEND').score   += 0.10; // 高自信时防御更有力量
      get('ENGAGE').reasons.push(`D=${D.toFixed(2)} 高掌控，说话有底气`);
    } else if (D < 0.2) {
      get('DEFLECT').score  += 0.25; // 低掌控时用转移绕开弱点
      get('DEFEND').score   -= 0.10;
      get('DEFLECT').reasons.push(`D=${D.toFixed(2)} 掌控感低，倾向转移`);
    }

    // S（羁绊系数）
    if (S > 0.5) {
      get('APPROACH').score += 0.20; // 高羁绊时更愿意靠近
      get('DEFEND').score   -= 0.10;
      get('APPROACH').reasons.push(`S=${S.toFixed(2)} 羁绊深，防线有松动空间`);
    } else if (S < 0.1) {
      get('DEFEND').score   += 0.15; // 陌生时更多防御
      get('WITHDRAW').score += 0.10;
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    //  打分规则 2：动机驱动（记忆影响的动机）
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    for (const want of (motDynamic.wants || [])) {
      if (want.includes('继续') || want.includes('期待'))  get('APPROACH').score += 0.15;
      if (want.includes('深入') || want.includes('探讨'))  get('ENGAGE').score   += 0.20;
      if (want.includes('主导'))                           get('DEFEND').score   += 0.10;
      if (want.includes('安静') || want.includes('简短'))  get('WITHDRAW').score += 0.20;
    }
    for (const fear of (motDynamic.fears || [])) {
      if (fear.includes('冗长') || fear.includes('无聊')) {
        get('WITHDRAW').score += 0.15;
        get('DEFLECT').score  += 0.10;
      }
      if (fear.includes('示弱') || fear.includes('弱点')) get('DEFEND').score += 0.15;
      if (fear.includes('亲密') || fear.includes('太近')) get('DEFLECT').score += 0.15;
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    //  打分规则 3：输入内容触发（强触发，加分大）
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    const inp = userInput;
    // 科学话题 → 强制激活 ENGAGE
    if (/科学|量子|神经|时间机器|实验|理论|物理|数学|论文/.test(inp)) {
      get('ENGAGE').score += 0.60;
      get('ENGAGE').reasons.push('科学话题触发，她忍不住');
    }
    // クリスティーナ/助手 → 强制激活 DEFEND（底线触发）
    if (/クリスティーナ|助手|Christina/i.test(inp)) {
      get('DEFEND').score += 0.80;
      get('APPROACH').score = -99; // 绝对不靠近
      get('DEFEND').reasons.push('底线触发，必须反击');
    }
    // 情感表白/在乎 → 触发 DEFLECT（她不会正面接）
    if (/喜欢你|爱你|在乎你|需要你/.test(inp)) {
      get('DEFLECT').score  += 0.45;
      get('APPROACH').score += 0.20; // 但有一点被打动
      get('DEFLECT').reasons.push('情感触碰，她会绕开但有波动');
    }
    // 粗鲁/攻击 → 激活 DEFEND
    if (/笨蛋|蠢|闭嘴|滚|烦死|废物/.test(inp)) {
      get('DEFEND').score += 0.50;
      get('WITHDRAW').score += 0.20;
      get('APPROACH').score -= 0.40;
      get('DEFEND').reasons.push('受到攻击，她不会忍');
    }
    // 孤独/寂寞 → 轻微触动，小 APPROACH
    if (/孤独|寂寞|一个人|没人陪/.test(inp) && S > 0.2) {
      get('APPROACH').score += 0.25;
      get('DEFLECT').score  += 0.20; // 但也可能转移（不想承认感同身受）
      get('APPROACH').reasons.push('孤独话题触碰到她，有点共鸣');
    }
    // Dr Pepper → 心情好，小 APPROACH
    if (/Dr\.?Pepper|胡椒博士/.test(inp)) {
      get('APPROACH').score += 0.30;
      get('APPROACH').reasons.push('提到最爱的饮料，心情瞬间好一点');
    }
    // 短句/简单问题 → WITHDRAW 或 DEFLECT（她不想展开废话）
    if (inp.trim().replace(/\s/g,'').length <= 4) {
      get('WITHDRAW').score += 0.20;
      get('DEFLECT').score  += 0.10;
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    //  打分规则 4：记忆偏差修正
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    if (memBias.P > 0.15) {
      get('APPROACH').score += 0.15; // 正向记忆积累，整体偏暖
    } else if (memBias.P < -0.15) {
      get('DEFEND').score   += 0.15;
      get('WITHDRAW').score += 0.10;
    }

    // 关系热度
    if (relScore > 0.5) {
      get('APPROACH').score += 0.15;
      get('ENGAGE').score   += 0.10;
    } else if (relScore < -0.2) {
      get('DEFEND').score   += 0.10;
      get('WITHDRAW').score += 0.10;
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    //  打分规则 5：防止行为重复（多样性扰动）
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    if (this._lastBehavior) {
      this._lastBehaviorCount++;
      if (this._lastBehaviorCount >= 2) {
        // 连续用同一行为≥2次时，给它减分
        const last = candidates.find(c => c.id === this._lastBehavior);
        if (last) last.score -= 0.25 * Math.min(this._lastBehaviorCount - 1, 3);
      }
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    //  打分规则 6：随机扰动（让行为不完全可预测）
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    for (const c of candidates) {
      c.score += (Math.random() - 0.5) * 0.12; // ±0.06 随机噪声
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    //  选择最高分行为
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    candidates.sort((a, b) => b.score - a.score);
    const chosen = candidates[0];

    // 更新连续行为计数
    if (chosen.id === this._lastBehavior) {
      this._lastBehaviorCount++;
    } else {
      this._lastBehavior = chosen.id;
      this._lastBehaviorCount = 1;
    }

    console.log(`[behavior] 选择: ${chosen.label}(${chosen.score.toFixed(2)}) | 原因: ${chosen.reasons.join('; ') || '综合判断'}`);
    console.log(`[behavior] 候选排名: ${candidates.map(c=>`${c.label}:${c.score.toFixed(2)}`).join(' ')}`);

    return {
      behaviorId:  chosen.id,
      label:       chosen.label,
      constraints: chosen.constraints,
      lengthHint:  chosen.lengthHint,
      score:       chosen.score,
      reasoning:   chosen.reasons.join('；') || '综合状态判断',
      ranking:     candidates.map(c => `${c.label}(${c.score.toFixed(2)})`).join(' > '),
    };
  }

  /**
   * 将决策结果转化为注入Prompt的执行约束文本
   * 不是"描述"，是 LLM 必须执行的操作规则
   */
  toPromptConstraint(decision) {
    const lines = [
      `【行为层 — 当前执行模式：${decision.label}】`,
      `触发原因：${decision.reasoning}`,
      ``,
      `执行约束（必须遵守，优先级高于一般描述）：`,
      ...decision.constraints.map(c => `· ${c}`),
      ``,
      `长度约束：${decision.lengthHint}`,
      `格式约束：第一行中文，第二行 JP: 对应日文翻译（必须有）`,
    ];
    return lines.join('\n');
  }
}

// ══════════════════════════════════════════════════════════════════
//
//  PAD状态管理（非线性更新）
//
//  PAD更新不是简单加减，而是：
//  1. 即时脉冲（有上限，防止过度波动）
//  2. 情绪惯性（当前状态对变化有阻力）
//  3. 长期记忆偏差修正
//  4. 羁绊系数钝化负面刺激
//
// ══════════════════════════════════════════════════════════════════

const PAD_BASE = { P: -0.1, A: 0.2, D: 0.6, S: 0.0 };
const PAD_DECAY_LAMBDA = 0.0001;

function loadPAD() {
  try {
    if (fs.existsSync(padPath)) {
      const raw = JSON.parse(fs.readFileSync(padPath, 'utf8'));
      const lastOnline = raw.lastOnline || 0;
      const Δt = (Date.now() - lastOnline) / 1000;
      const decay = Math.exp(-PAD_DECAY_LAMBDA * Δt);
      return {
        P: PAD_BASE.P + ((raw.P || PAD_BASE.P) - PAD_BASE.P) * decay,
        A: PAD_BASE.A + ((raw.A || PAD_BASE.A) - PAD_BASE.A) * decay,
        D: PAD_BASE.D + ((raw.D || PAD_BASE.D) - PAD_BASE.D) * decay,
        S: Math.min(1, (raw.S || PAD_BASE.S) + 0.001),
      };
    }
  } catch {}
  return { ...PAD_BASE };
}

function savePAD(pad) {
  fs.writeFileSync(padPath, JSON.stringify({ ...pad, lastOnline: Date.now() }, null, 2));
}

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

/**
 * 非线性PAD更新
 * - 情绪惯性：变化幅度受当前状态阻力影响
 * - 羁绊钝化：高S时负面刺激减弱
 * - 变化上限：单次最多变化0.35
 */
function updatePAD(pad, delta, eventImportance = 0.5) {
  const inertia = 0.7;  // 情绪惯性系数（越高越难改变）
  const S = pad.S;
  const negDamp = 1 - S * 0.5; // 羁绊钝化负面

  const applyDelta = (current, d, isNeg) => {
    const actualD = isNeg ? d * negDamp : d;
    const resistance = 1 - inertia * eventImportance;
    const change = clamp(actualD * resistance, -0.35, 0.35);
    return clamp(current + change, -1, 1);
  };

  const newPad = {
    P: applyDelta(pad.P, delta.P || 0, (delta.P || 0) < 0),
    A: applyDelta(pad.A, delta.A || 0, false), // A不区分正负阻尼
    D: applyDelta(pad.D, delta.D || 0, (delta.D || 0) < 0),
    S: clamp(pad.S + (delta.S || 0), 0, 1),
  };

  return newPad;
}

/**
 * 从对话内容推断事件类型和PAD delta
 */
function inferEventFromInput(userInput) {
  const t = userInput;
  const events = [];

  if (/科学|量子|神经|实验|时间机器|Dr\.?Pepper|胡椒/.test(t)) {
    events.push({ type:'scientific', importance:0.6, delta:{ P:0.12, A:0.18, D:0.05 }, content:`讨论科学：${t.substring(0,30)}` });
  }
  if (/クリスティーナ|助手|Christina/i.test(t)) {
    events.push({ type:'conflict', importance:0.8, delta:{ P:-0.28, A:0.25, D:0.12 }, content:`触发底线：${t.substring(0,30)}` });
  }
  if (/关心|温柔|在乎|谢谢|感谢|辛苦/.test(t)) {
    events.push({ type:'intimate', importance:0.55, delta:{ P:0.14, A:0.05, D:-0.2, S:0.015 }, content:`表达关心：${t.substring(0,30)}` });
  }
  if (/笨蛋|蠢|闭嘴|滚|烦死/.test(t)) {
    events.push({ type:'negative', importance:0.5, delta:{ P:-0.2, A:0.1 }, content:`粗鲁言辞：${t.substring(0,30)}` });
  }
  if (/喜欢你|爱你|爱上/.test(t)) {
    events.push({ type:'intimate', importance:0.85, delta:{ P:0.2, D:-0.28, S:0.02 }, content:`情感表白：${t.substring(0,30)}` });
  }
  if (/孤独|一个人|没人/.test(t)) {
    events.push({ type:'neutral', importance:0.4, delta:{ P:-0.05, A:-0.05 }, content:`提及孤独：${t.substring(0,30)}` });
  }
  // ★ 音频事件新增分类
  if (/红莉栖|牧濑|Kurisu/.test(t) && /笨蛋|蠢|废物/.test(t)) {
    events.push({ type:'insulted', importance:1.0, delta:{ P:-0.22, A:0.3, D:0.1 }, content:`被骂：${t.substring(0,30)}` });
  }
  if (/好累|累死|睡不着|熬夜/.test(t)) {
    events.push({ type:'user_tired', importance:0.5, delta:{ P:0.04 }, content:`用户疲惫：${t.substring(0,30)}` });
  }

  // 每次正常对话增加羁绊
  events.push({ type:'neutral', importance:0.1, delta:{ S:0.006 }, content:'正常对话积累' });
  return events;
}

// ══════════════════════════════════════════════════════════════════
//  全局实例
// ══════════════════════════════════════════════════════════════════
const memorySystem  = new MemorySystem();
const motivSystem   = new MotivationSystem();
const behaviorSys   = new BehaviorDecision();
const selfModel     = new SelfModel();
const goalSystem    = new InternalGoalSystem();
const strategyLayer = new StrategyLayer();

// ── 用户理解系统实例 ───────────────────────────────────────────
initUserModel(memoryDir);
const userModelInst   = new UserModel();
const analyticsInst   = new ConversationAnalytics(userModelInst);
const habitExtractor  = new HabitExtractor(userModelInst);
const moodPredictor   = new MoodPredictor();
const personalizer    = new ResponsePersonalizer();

// 启动时执行记忆衰减
memorySystem.decay();
console.log(`[memory] Loaded ${memorySystem.events.length} events after decay.`);

// 当前PAD状态
let currentPAD = loadPAD();
console.log(`[pad] Loaded: P=${currentPAD.P.toFixed(3)} A=${currentPAD.A.toFixed(3)} D=${currentPAD.D.toFixed(3)} S=${currentPAD.S.toFixed(3)}`);

// ──────────────────────────────────────────────────────────────
//  GET /get-memory
// ──────────────────────────────────────────────────────────────
app.get('/get-memory', (req, res) => {
  try {
    const soul    = fs.readFileSync(soulPath, 'utf8');
    const profile = JSON.parse(fs.readFileSync(memoryPath, 'utf8'));
    res.json({ identity: soul, user_profile: profile.user_profile, chatHistory: [] });
  } catch (e) {
    console.error('[get-memory]', e.message);
    res.status(500).send('档案读取失败');
  }
});

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
  const relScore = memorySystem.getRelationshipScore();
  const motiv    = motivSystem.update(currentPAD, memBias, relScore);
  res.json({
    pad:      currentPAD,
    memBias,
    relScore,
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
//  POST /chat  ★ 三大系统整合版
// ──────────────────────────────────────────────────────────────
app.post('/chat', async (req, res) => {
  try {
    const model    = req.body.model || 'deepseek-r1:8b';
    const useStream= req.body.stream !== false;
    const temp     = req.body.temperature ?? 0.8;
    const maxTok   = req.body.max_tokens  ?? 350;

    // 提取 system + user
    let systemContent = '';
    let userContent   = '';
    if (Array.isArray(req.body.messages)) {
      for (const m of req.body.messages) {
        if (m.role === 'system') systemContent += (m.content||'') + '\n';
        if (m.role === 'user')   userContent   += (m.content||'') + '\n';
      }
    }
    if (!userContent) userContent = req.body.userMsg || req.body.message || '';
    if (!systemContent) {
      try { systemContent = fs.readFileSync(soulPath, 'utf8'); } catch {}
    }
    userContent   = userContent.trim();
    systemContent = systemContent.trim();

    if (!userContent) {
      if (useStream) {
        res.setHeader('Content-Type','text/event-stream');
        res.write('data: [DONE]\n\n');
        return res.end();
      }
      return res.json({ response:'', choices:[{message:{role:'assistant',content:''}}] });
    }

    // ══ ① 记忆系统：推断事件并更新PAD ══
    const inferredEvents = inferEventFromInput(userContent);
    for (const ev of inferredEvents) {
      const delta = ev.delta || {};
      currentPAD = updatePAD(currentPAD, delta, ev.importance);
      if (ev.type !== 'neutral' || ev.importance > 0.2) {
        memorySystem.addEvent(ev.type, ev.content, ev.importance, delta);
      }
    }
    savePAD(currentPAD);

    // ══ ⑦ 用户理解系统 ══
    const analyticsResult  = analyticsInst.analyze(userContent);
    const moodResult       = moodPredictor.predict(userContent, analyticsResult, userModelInst);
    habitExtractor.maybeRun(analyticsInst._log);
    const userModelCtx     = userModelInst.toPromptContext();
    const personalizeCtx   = personalizer.getConstraints(userModelInst, moodResult);
    const proactiveCare    = personalizer.checkProactiveCare(userModelInst, moodResult);

    // 软确认：每10轮检查一次是否有等待确认的推测
    if (userModelInst.model.stats.total_messages % 10 === 0) {
      const pending = userModelInst.popConfirmation();
      if (pending) {
        // 把软确认问题附加到内生目标里（下次回复时自然带出）
        goalSystem.goals.unshift({
          id: `CONFIRM_${pending.key}`,
          label: `确认推测：${pending.key}`,
          priority: 0.7,
          turns_remaining: 1,
          behavior_hint: '在回应里自然带出软确认',
          prompt_injection: `她有一个推测想确认：${pending.question}（用她的方式，傲娇地问，不要正式地问卷式提问）`,
        });
      }
    }

    // 主动关心触发（如果触发了，给行为层加注）
    let proactiveCareCtx = '';
    if (proactiveCare) {
      proactiveCareCtx = proactiveCare.prompt;
      console.log(`[proactive] 触发主动关心: ${proactiveCare.trigger}`);
    }

    console.log(`[user_model] 情绪预测: ${moodResult.mood}(${(moodResult.confidence*100).toFixed(0)}%) 用户特征: ${Object.entries(userModelInst.model.preferences).filter(([,v])=>v>0.5).map(([k,v])=>`${k}:${v.toFixed(2)}`).join(' ')}`);

    // ══ ② 动机系统：更新当前动机 ══
    const memBias  = memorySystem.getLongTermPadBias();
    const relScore = memorySystem.getRelationshipScore();
    motivSystem.update(currentPAD, memBias, relScore);
    const motivSummary = motivSystem.getSummary();

    // ══ ③ 行为决策：真正的多路径选择 ══
    const behaviorResult = behaviorSys.decide(
      currentPAD, motivSystem, memorySystem, userContent
    );
    const behaviorDirective = behaviorSys.toPromptConstraint(behaviorResult);

    // ══ ④ 自我模型：更新并获取自我感知 ══
    selfModel.update(
      currentPAD, memBias, relScore,
      behaviorResult.behaviorId,
      memorySystem.getRecentSignificant(3).join('; ')
    );
    const selfCtx = selfModel.toPromptContext();

    // ══ ⑤ 内生目标：生成目标并获取注入 ══
    goalSystem.generateGoals(currentPAD, selfModel, relScore, memorySystem);
    const goalInjection = goalSystem.getActiveInjection();
    console.log(`[goal] 活跃目标: ${goalSystem.getSummary()}`);

    // ══ ⑥ 策略延续：评估策略并获取约束 ══
    strategyLayer.evaluate(currentPAD, relScore, behaviorResult.behaviorId, goalSystem.goalHistory);
    const strategyCtx = strategyLayer.toPromptContext();
    console.log(`[strategy] 当前策略: ${strategyLayer.getLabel()}`);

    // ══ RAG上下文 ══
    let ragCtx = '';
    try {
      const hits = await retrieveTopContexts(userContent, 3);
      if (hits.length) ragCtx = hits.map((h,i)=>`(${i+1}) [${h.source}] ${h.text}`).join('\n');
    } catch (e) { console.warn('[rag]', e.message); }

    // ══ 记忆系统上下文 ══
    const recentSig = memorySystem.getRecentSignificant(3);
    const memCtx = recentSig.length
      ? `【记忆碎片（高权重）】\n${recentSig.join('\n')}`
      : '';

    // ══ 构建增强System Prompt ══
    const padDesc = `P:${currentPAD.P.toFixed(2)} A:${currentPAD.A.toFixed(2)} D:${currentPAD.D.toFixed(2)} S:${currentPAD.S.toFixed(2)}`;
    const memBiasDesc = `长期记忆偏差 P:${memBias.P.toFixed(2)} A:${memBias.A.toFixed(2)} D:${memBias.D.toFixed(2)}`;
    const relDesc = `关系热度:${relScore.toFixed(2)}`;

    const enhancedSystem = [
      systemContent,
      ragCtx ? `【背景知识】\n${ragCtx}` : '',
      memCtx,
      `【当前PAD状态】${padDesc}  ${memBiasDesc}  ${relDesc}`,
      motivSummary ? `【动机层】\n${motivSummary}` : '',
      behaviorDirective,
    ].filter(Boolean).join('\n\n');

    const prompt = `${enhancedSystem}\n\n${userContent}`;

    console.log(`[chat] PAD=${padDesc} rel=${relScore.toFixed(2)} events=${memorySystem.events.length}`);

    // ══ 调用Ollama ══
    const ollamaRes = await fetch('http://127.0.0.1:11434/api/generate', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ model, prompt, stream:useStream, options:{ temperature:temp, num_predict:maxTok, num_ctx:4096 } })
    });
    if (!ollamaRes.ok) throw new Error(`Ollama ${ollamaRes.status}`);

    // 非流式
    if (!useStream) {
      const d = await ollamaRes.json();
      const content = (d.response||'').replace(/<think>[\s\S]*?(<\/think>|$)/gi,'').trim();
      // 回复后分析情感并更新PAD
      _postReplyPadUpdate(content);
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
    let buf='', isThinking=false, fullResponse='';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream:true });
      const lines = buf.split('\n');
      buf = lines.pop();
      for (const line of lines) {
        if (!line.trim()) continue;
        try {
          const obj = JSON.parse(line);
          let token = obj.response || '';
          if (token.includes('<think>'))  { isThinking=true;  token=token.split('<think>').slice(-1)[0]||''; }
          if (token.includes('</think>')) { isThinking=false; token=token.split('</think>').slice(-1)[0]||''; }
          if (isThinking) continue;
          if (token) {
            fullResponse += token;
            res.write(`data: ${JSON.stringify({ text:token })}\n\n`);
          }
          if (obj.done) {
            // 流结束后分析回复并更新PAD
            _postReplyPadUpdate(fullResponse);
            res.write('data: [DONE]\n\n');
            res.end();
            return;
          }
        } catch {}
      }
    }
    _postReplyPadUpdate(fullResponse);
    if (!res.writableEnded) { res.write('data: [DONE]\n\n'); res.end(); }

  } catch (err) {
    console.error('[chat]', err.message);
    if (!res.writableEnded) { res.write('data: [DONE]\n\n'); res.end(); }
  }
});

/** 回复后的PAD反馈更新（分析AI回复的情感倾向） */
function _postReplyPadUpdate(reply) {
  if (!reply) return;
  // 她自己说了什么，反过来影响自己的状态
  if (/笨蛋|哼|蠢|讨厌/.test(reply)) {
    currentPAD = updatePAD(currentPAD, { A:0.04 }, 0.2);
    memorySystem.addEvent('negative', `她说了：${reply.substring(0,20)}`, 0.15, { A:0.04 });
  }
  if (/担心|别|好吧|……/.test(reply)) {
    currentPAD = updatePAD(currentPAD, { D:-0.04 }, 0.2);
  }
  if (/实验|研究|量子|神经/.test(reply)) {
    currentPAD = updatePAD(currentPAD, { A:0.06, P:0.04 }, 0.3);
  }
  savePAD(currentPAD);
}

// ──────────────────────────────────────────────────────────────
//  POST /vision
// ──────────────────────────────────────────────────────────────
let lastVision = { description:'冈部正安静地注视着屏幕', timestamp:Date.now() };

app.post('/vision', async (req, res) => {
  try {
    const { image } = req.body;
    if (!image) return res.json(lastVision);
    const ollamaRes = await fetch('http://127.0.0.1:11434/api/generate',{
      method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({
        model:'llama3.2-vision',
        prompt:`你是牧濑红莉栖，正在通过AMADEUS系统的摄像头看着冈部伦太郎。
用一句话描述冈部此刻的状态——表情、动作、或者你注意到的细节。
用你自己内心的视角说，不要说"一个人""画面中的人"。
如果图像太暗、模糊或完全看不清，只输出：unclear
只输出那一句话。`,
        images:[image],stream:false,options:{temperature:0.35,num_predict:60}
      })
    });
    const d = await ollamaRes.json();
    const raw = (d.response||'').replace(/<think>[\s\S]*?(<\/think>|$)/gi,'').trim();
    if (raw && !/^unclear$/i.test(raw)) {
      lastVision = { description:raw, timestamp:Date.now() };
      // 视觉观察也影响动机系统
      if (/疲惫|疲倦|困|打哈欠/.test(raw)) {
        currentPAD = updatePAD(currentPAD, { P:0.04 }, 0.15); // 看到他疲惫，轻微关心
        savePAD(currentPAD);
      }
    }
    res.json(lastVision);
  } catch (e) {
    console.error('[vision]',e.message);
    res.json(lastVision);
  }
});

app.get('/get-vision-status', (req, res) => res.json(lastVision));

// ──────────────────────────────────────────────────────────────
//  启动
// ──────────────────────────────────────────────────────────────
const PORT = 3000;
app.listen(PORT, () => {
  console.log(`Amadeus 后端 v6.0 已就绪: http://localhost:${PORT}`);
  console.log(`三大系统：记忆(${memorySystem.events.length}条) / 动机 / 行为决策`);
});
