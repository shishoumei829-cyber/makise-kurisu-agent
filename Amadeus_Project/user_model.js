'use strict';

// ══════════════════════════════════════════════════════════════════
//
//  用户理解系统 v2.0
//
//  功能：
//  1. 用户习惯提取（说话模式、话题偏好、行为规律）
//  2. 情绪预测（基于文本分析和历史模式）
//  3. 心智模型（信念、欲望、意图推断）
//  4. 情感共鸣（共情反应、情绪追踪）
//  5. 关系动态（关系阶段、信任度、亲密度）
//
// ══════════════════════════════════════════════════════════════════

const fs = require('fs');
const path = require('path');

let _log = [];
let _dataDir = '';
let _persistedModel = null;

// ── 情绪关键词库 ──────────────────────────────────────────────────
const EMOTION_KEYWORDS = {
  positive: ['开心', '高兴', '快乐', '喜欢', '爱', '感谢', '谢谢', '好的', '太棒了', '哈哈', '嘻嘻', '笑'],
  negative: ['难过', '伤心', '生气', '烦', '讨厌', '恨', '累', '疲倦', '无聊', '孤独', '寂寞'],
  curious: ['为什么', '怎么', '什么', '吗', '？', '好奇', '想知道'],
  intimate: ['想你', '喜欢你', '爱你', '在乎', '担心', '关心', '在乎你'],
  aggressive: ['笨蛋', '蠢', '闭嘴', '滚', '烦死', '废物', '讨厌你'],
};

// ── 话题关键词库 ──────────────────────────────────────────────────
const TOPIC_KEYWORDS = {
  science: ['科学', '物理', '数学', '神经', '实验', '理论', '研究', '论文', '量子'],
  daily: ['吃', '喝', '睡', '玩', '看', '听', '今天', '明天', '昨天'],
  emotional: ['感觉', '心情', '感情', '喜欢', '爱', '讨厌', '害怕', '担心'],
  technical: ['代码', '程序', '系统', 'bug', '错误', '问题', '解决'],
  personal: ['我', '你', '他', '她', '我们', '他们', '名字', '年龄', '工作'],
};

function init(dir) {
  _dataDir = dir;
  _log = [];
  _persistedModel = null;
  try {
    const p = path.join(dir, 'user_model.json');
    if (fs.existsSync(p)) {
      const d = JSON.parse(fs.readFileSync(p, 'utf8'));
      if (d._log) _log = d._log;
      _persistedModel = d && typeof d === 'object' ? d : null;
    }
  } catch {}
}

let _saveUserModelTimer = null;
function _save(data) {
  if (_saveUserModelTimer) clearTimeout(_saveUserModelTimer);
  _saveUserModelTimer = setTimeout(() => {
    _saveUserModelTimer = null;
    try {
      const p = path.join(_dataDir, 'user_model.json');
      fs.writeFileSync(p, JSON.stringify({ ...data, _log }, null, 2));
    } catch {}
  }, 200);
}

// ── 用户模型类 ──────────────────────────────────────────────────
class UserModel {
  constructor() {
    const defaults = {
      stats: { 
        total_messages: 0,
        first_interaction: null,
        last_interaction: null,
        interaction_frequency: 0,
      },
      preferences: {
        topics: {},        // 话题偏好：{ topic: count }
        styles: {},        // 语言风格偏好
        response_length: 'medium', // 偏好的回复长度
      },
      patterns: {
        active_hours: {},  // 活跃时间段
        message_length: [], // 消息长度历史
        emotion_history: [], // 情绪历史
      },
      /** 由 BDI 引擎（LLM）周期性写入 */
      inferred_bdi: {
        beliefs: [],
        desires: [],
        intentions: [],
        updated_at: 0,
      },
      relationship: {
        stage: 'stranger', // stranger, acquaintance, friend, close, intimate
        trust_level: 0.5,  // 0-1
        closeness: 0.5,    // 0-1
        history: [],       // 关系事件历史
      },
      pending_confirmations: [], // 待确认的推测
    };
    const { _log: _ignoredLog, ...saved } = _persistedModel || {};
    this.model = {
      ...defaults,
      ...saved,
      stats: { ...defaults.stats, ...(saved.stats || {}) },
      preferences: { ...defaults.preferences, ...(saved.preferences || {}), topics: { ...(saved.preferences?.topics || {}) }, styles: { ...(saved.preferences?.styles || {}) } },
      patterns: { ...defaults.patterns, ...(saved.patterns || {}) },
      inferred_bdi: { ...defaults.inferred_bdi, ...(saved.inferred_bdi || {}) },
      relationship: { ...defaults.relationship, ...(saved.relationship || {}) },
      pending_confirmations: Array.isArray(saved.pending_confirmations) ? saved.pending_confirmations : [],
    };
  }

  toPromptContext() {
    const m = this.model;
    const parts = [];
    
    if (m.stats.total_messages > 0) {
      parts.push(`你们已经聊了 ${m.stats.total_messages} 条消息`);
    }
    
    // 话题偏好
    const topTopics = Object.entries(m.preferences.topics)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3)
      .map(([topic]) => topic);
    if (topTopics.length > 0) {
      parts.push(`他感兴趣的话题：${topTopics.join('、')}`);
    }
    
    // 关系阶段
    const stageNames = {
      stranger: '陌生人',
      acquaintance: '认识的人',
      friend: '朋友',
      close: '亲密朋友',
      intimate: '特别亲密的人',
    };
    parts.push(`你们的关系：${stageNames[m.relationship.stage] || '未知'}`);
    
    // 情绪状态
    if (m.patterns.emotion_history.length > 0) {
      const recentEmotion = m.patterns.emotion_history[m.patterns.emotion_history.length - 1];
      parts.push(`他最近的情绪：${recentEmotion.emotion}`);
    }

    // BDI 推断（LLM 周期性更新，10分钟内有效）
    const bdi = m.inferred_bdi;
    if (bdi && bdi.updated_at && (Date.now() - bdi.updated_at) < 10 * 60 * 1000) {
      if (Array.isArray(bdi.beliefs) && bdi.beliefs.length)
        parts.push(`推测他的信念：${bdi.beliefs.join('、')}`);
      if (Array.isArray(bdi.desires) && bdi.desires.length)
        parts.push(`推测他的欲望：${bdi.desires.join('、')}`);
      if (Array.isArray(bdi.intentions) && bdi.intentions.length)
        parts.push(`推测他的意图：${bdi.intentions.join('、')}`);
    }

    if (m.relationship.closeness > 0.45) {
      parts.push(`亲密度约 ${(m.relationship.closeness * 100).toFixed(0)}%（你们已经比较近了）`);
    }

    return parts.join('\n');
  }

  /** 与 memory.getRelationshipScore 对齐，供 prompt 使用 */
  syncRelationshipFromScore(relScore) {
    const r = Number(relScore);
    if (!Number.isFinite(r)) return;
    const closeness = Math.max(0, Math.min(1, (r + 1) / 2));
    const trust = Math.max(0, Math.min(1, 0.5 + r * 0.5));
    const m = this.model.relationship;
    m.closeness = closeness;
    m.trust_level = trust;
    if (r > 0.55) m.stage = 'intimate';
    else if (r > 0.38) m.stage = 'close';
    else if (r > 0.15) m.stage = 'friend';
    else if (r > 0) m.stage = 'acquaintance';
    else m.stage = 'stranger';
    _save(this.model);
  }

  /** @param {{ beliefs?: string[], desires?: string[], intentions?: string[] } | null} bdi */
  applyInferredBdi(bdi) {
    if (!bdi) return;
    this.model.inferred_bdi = {
      beliefs: Array.isArray(bdi.beliefs) ? bdi.beliefs.slice(0, 5) : [],
      desires: Array.isArray(bdi.desires) ? bdi.desires.slice(0, 5) : [],
      intentions: Array.isArray(bdi.intentions) ? bdi.intentions.slice(0, 5) : [],
      updated_at: Date.now(),
    };
    _save(this.model);
  }

  popConfirmation() {
    if (this.model.pending_confirmations.length === 0) return null;
    return this.model.pending_confirmations.shift();
  }

  purgeContaminatedTopics(fragments = []) {
    const list = fragments.map(String).filter(Boolean);
    const hit = (value) => list.some((fragment) => String(value || '').includes(fragment));
    const bdi = this.model.inferred_bdi || {};
    bdi.beliefs = (bdi.beliefs || []).filter((item) => !hit(item));
    bdi.desires = (bdi.desires || []).filter((item) => !hit(item));
    bdi.intentions = (bdi.intentions || []).filter((item) => !hit(item));
    this.model.pending_confirmations = (this.model.pending_confirmations || [])
      .filter((item) => !hit(JSON.stringify(item)));
    _log = _log.filter((item) => !hit(item.text));
    _save(this.model);
  }
}

// ── 对话分析类 ──────────────────────────────────────────────────
class ConversationAnalytics {
  constructor(userModel) {
    this.userModel = userModel;
    this._log = _log;
  }

  analyze(text, options = {}) {
    const m = this.userModel.model;
    m.stats.total_messages++;
    m.stats.last_interaction = Date.now();
    if (!m.stats.first_interaction) {
      m.stats.first_interaction = Date.now();
    }
    
    // 分析情绪
    const sentiment = this._analyzeSentiment(text);
    
    // 分析话题
    const topics = this._analyzeTopics(text);
    
    // 分析意图
    const intent = this._analyzeIntent(text);
    
    // 更新话题偏好
    for (const topic of topics) {
      m.preferences.topics[topic] = (m.preferences.topics[topic] || 0) + 1;
    }
    
    // 记录消息长度
    m.patterns.message_length.push(text.length);
    if (m.patterns.message_length.length > 100) {
      m.patterns.message_length = m.patterns.message_length.slice(-100);
    }
    
    // 记录情绪历史
    m.patterns.emotion_history.push({
      emotion: sentiment.emotion,
      score: sentiment.score,
      timestamp: Date.now(),
    });
    if (m.patterns.emotion_history.length > 50) {
      m.patterns.emotion_history = m.patterns.emotion_history.slice(-50);
    }
    
    // 记录活跃时间
    const hour = new Date().getHours();
    m.patterns.active_hours[hour] = (m.patterns.active_hours[hour] || 0) + 1;
    
    // 记录日志
    _log.push({
      text: options.persistText === false ? '' : text.substring(0, 100),
      intent,
      sentiment: sentiment.score,
      topics,
      emotion: sentiment.emotion,
      timestamp: Date.now(),
    });
    if (_log.length > 200) {
      _log = _log.slice(-200);
    }
    
    // 保存
    _save(m);
    
    return {
      intent,
      sentiment: sentiment.score,
      topics,
      emotion: sentiment.emotion,
    };
  }

  _analyzeSentiment(text) {
    let score = 0;
    let emotion = 'neutral';
    
    // 检查正面情绪
    for (const word of EMOTION_KEYWORDS.positive) {
      if (text.includes(word)) {
        score += 0.2;
        emotion = 'positive';
      }
    }
    
    // 检查负面情绪
    for (const word of EMOTION_KEYWORDS.negative) {
      if (text.includes(word)) {
        score -= 0.2;
        emotion = 'negative';
      }
    }
    
    // 检查好奇
    for (const word of EMOTION_KEYWORDS.curious) {
      if (text.includes(word)) {
        emotion = 'curious';
      }
    }
    
    // 检查亲密
    for (const word of EMOTION_KEYWORDS.intimate) {
      if (text.includes(word)) {
        emotion = 'intimate';
        score += 0.1;
      }
    }
    
    // 检查攻击性
    for (const word of EMOTION_KEYWORDS.aggressive) {
      if (text.includes(word)) {
        emotion = 'aggressive';
        score -= 0.3;
      }
    }
    
    return { score: Math.max(-1, Math.min(1, score)), emotion };
  }

  _analyzeTopics(text) {
    const topics = [];
    
    for (const [topic, keywords] of Object.entries(TOPIC_KEYWORDS)) {
      for (const keyword of keywords) {
        if (text.includes(keyword)) {
          topics.push(topic);
          break;
        }
      }
    }
    
    return topics;
  }

  _analyzeIntent(text) {
    // 问题
    if (text.includes('？') || text.includes('?') || text.includes('吗') || text.includes('什么') || text.includes('怎么')) {
      return 'question';
    }
    
    // 陈述
    if (text.includes('是') || text.includes('有') || text.includes('在')) {
      return 'statement';
    }
    
    // 命令
    if (text.includes('请') || text.includes('帮') || text.includes('告诉我')) {
      return 'request';
    }
    
    // 情感表达
    if (text.includes('喜欢') || text.includes('爱') || text.includes('讨厌') || text.includes('恨')) {
      return 'emotional';
    }
    
    return 'neutral';
  }
}

// ── 习惯提取类 ──────────────────────────────────────────────────
class HabitExtractor {
  constructor(userModel) {
    this.userModel = userModel;
  }

  maybeRun(log) {
    if (log.length < 10) return; // 需要至少10条消息才能提取习惯
    
    const m = this.userModel.model;
    const recentLogs = log.slice(-50); // 分析最近50条消息
    
    // 提取活跃时间段
    const hourCounts = {};
    for (const entry of recentLogs) {
      if (entry.timestamp) {
        const hour = new Date(entry.timestamp).getHours();
        hourCounts[hour] = (hourCounts[hour] || 0) + 1;
      }
    }
    
    // 找出最活跃的时间段 - 存储为对象格式
    const activeHoursObj = {};
    for (const [h, count] of Object.entries(hourCounts)) {
      if (count > 0) activeHoursObj[h] = count;
    }
    m.patterns.active_hours = activeHoursObj;
    
    // 提取消息长度偏好
    const avgLength = m.patterns.message_length.reduce((a, b) => a + b, 0) / m.patterns.message_length.length;
    if (avgLength < 20) {
      m.preferences.response_length = 'short';
    } else if (avgLength > 100) {
      m.preferences.response_length = 'long';
    } else {
      m.preferences.response_length = 'medium';
    }
    
    // 保存
    _save(m);
  }
}

module.exports = {
  init,
  UserModel,
  ConversationAnalytics,
  HabitExtractor,
};
