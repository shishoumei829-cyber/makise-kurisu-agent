'use strict';

const { TOPIC_CLUSTERS } = require('./constants');

/**
 * 好奇心引擎 v2：话题图谱、信息增益、未解问题追踪、可注入 prompt 的追问
 */
class CuriosityEngine {
  constructor() {
    this.topicGraph = new Map();
    this.informationGain = new Map();
    this.openQuestions = [];
    this.answeredQuestions = [];
    this.clusterAffinity = {
      science: 0.85,
      emotion: 0.55,
      daily: 0.45,
      meta: 0.75,
      relationship: 0.60,
    };
    this._lastScanAt = 0;
  }

  _tokenize(text) {
    return (String(text || '').match(/[\u4e00-\u9fa5]{2,8}/g) || [])
      .filter((t) => !/^的$|^了$|^吗$|^呢$|^啊$/.test(t));
  }

  _clusterOf(token) {
    for (const [name, re] of Object.entries(TOPIC_CLUSTERS)) {
      if (re.test(token)) return name;
    }
    return 'general';
  }

  ingestText(text, weight = 1) {
    const tokens = this._tokenize(text);
    const seen = new Set();
    for (const tok of tokens) {
      const node = this.topicGraph.get(tok) || {
        count: 0,
        lastSeen: 0,
        cluster: this._clusterOf(tok),
        cooccur: {},
      };
      node.count += weight;
      node.lastSeen = Date.now();
      this.topicGraph.set(tok, node);
      seen.add(tok);
    }
    const list = [...seen];
    for (let i = 0; i < list.length; i++) {
      for (let j = i + 1; j < list.length; j++) {
        const a = this.topicGraph.get(list[i]);
        const b = this.topicGraph.get(list[j]);
        if (a) a.cooccur[list[j]] = (a.cooccur[list[j]] || 0) + weight;
        if (b) b.cooccur[list[i]] = (b.cooccur[list[i]] || 0) + weight;
      }
    }
  }

  calculateInformationGain(topic, memory) {
    const now = Date.now();
    const prev = this.informationGain.get(topic);
    const daysSince = prev?.lastExplored
      ? (now - prev.lastExplored) / (86400000)
      : 14;
    const novelty = Math.min(1, daysSince / 7);

    const events = memory?.events || [];
    const related = events.filter((e) => e.content && e.content.includes(topic)).length;
    const relevance = Math.min(1, related / 4);

    const node = this.topicGraph.get(topic);
    const cluster = node?.cluster || this._clusterOf(topic);
    const affinity = this.clusterAffinity[cluster] ?? 0.5;
    const frequencyPenalty = node ? Math.min(0.4, node.count * 0.03) : 0;

    const gain = clamp(
      novelty * 0.38 + relevance * 0.28 + affinity * 0.34 - frequencyPenalty,
      0, 1,
    );

    this.informationGain.set(topic, {
      gain,
      lastExplored: now,
      explorationCount: (prev?.explorationCount || 0) + 1,
      cluster,
    });
    return gain;
  }

  discoverKnowledgeGaps(memory, selfModel) {
    const gaps = [];
    const explored = new Set();
    for (const ev of memory?.events || []) {
      this._tokenize(ev.content).forEach((t) => explored.add(t));
    }

    const sm = selfModel?.get?.() || {};
    const seeds = [
      ...(sm.identity_tags || []),
      ...this._tokenize(sm.relationship_perception || ''),
    ];

    for (const topic of seeds) {
      if (!topic || explored.has(topic)) continue;
      const gain = this.calculateInformationGain(topic, memory);
      if (gain > 0.35) {
        gaps.push({
          topic,
          importance: gain,
          reason: '自我/关系认知里的空洞',
          cluster: this._clusterOf(topic),
        });
      }
    }

    const candidates = [...this.topicGraph.entries()]
      .filter(([tok, n]) => n.count >= 2 && n.count <= 6)
      .map(([tok, n]) => {
        const gain = this.calculateInformationGain(tok, memory);
        return { topic: tok, importance: gain, reason: '提过但没聊透', cluster: n.cluster };
      })
      .filter((g) => g.importance > 0.4)
      .sort((a, b) => b.importance - a.importance);

    for (const c of candidates.slice(0, 4)) {
      if (!gaps.find((g) => g.topic === c.topic)) gaps.push(c);
    }

    return gaps.sort((a, b) => b.importance - a.importance).slice(0, 8);
  }

  generateCuriousQuestions(context, memory) {
    const { pad = {}, selfModel, relScore = 0 } = context;
    const gaps = this.discoverKnowledgeGaps(memory, selfModel);
    const questions = [];

    const templates = {
      science: (t) => [
        `「${t}」——你具体指哪一层？定义先对齐。`,
        `关于${t}，你看到的是现象还是机制？`,
      ],
      emotion: (t) => [
        `你说${t}的时候，是突然这样还是攒了一段时间？`,
        `……${t}。你不想多说也行，但我在听。`,
      ],
      relationship: () => [
        relScore > 0.4 ? '你最近到底在想什么？' : '你平时都怎么打发时间？',
      ],
      meta: (t) => [
        `${t}这东西……你认真想过吗？`,
      ],
      general: (t) => [
        `「${t}」是怎么回事？`,
        `我对${t}还有疑问。`,
      ],
    };

    for (const gap of gaps.slice(0, 3)) {
      const pool = templates[gap.cluster] || templates.general;
      const variants = typeof pool === 'function' ? pool(gap.topic) : pool;
      const q = variants[Math.floor(Math.random() * variants.length)];
      questions.push({
        question: q,
        topic: gap.topic,
        priority: gap.importance,
        type: 'knowledge_gap',
        gain: gap.importance,
      });
      this._registerOpenQuestion(gap.topic, q, gap.importance);
    }

    if (pad.S > 0.28 && pad.S < 0.72 && relScore > 0.15) {
      const q = '他到底是什么样的人——表面和内核可能不是一回事。';
      questions.push({
        question: q,
        topic: 'user_personality',
        priority: 0.55 + relScore * 0.2,
        type: 'understanding',
      });
      this._registerOpenQuestion('user_personality', q, 0.55 + relScore * 0.2);
    }
    if (pad.A > 0.48 && gaps.length === 0) {
      const q = '现在有什么值得认真聊的？';
      questions.push({
        question: q,
        topic: 'open_exploration',
        priority: 0.48,
        type: 'exploration',
      });
      this._registerOpenQuestion('open_exploration', q, 0.48);
    }

    return questions.sort((a, b) => b.priority - a.priority).slice(0, 5);
  }

  _registerOpenQuestion(topic, question, priority) {
    const existing = this.openQuestions.find((q) => q.topic === topic && !q.answered);
    if (existing) {
      existing.priority = Math.max(existing.priority, priority);
      existing.lastRaised = Date.now();
      return;
    }
    this.openQuestions.push({
      topic,
      question,
      priority,
      askedAt: Date.now(),
      lastRaised: Date.now(),
      answered: false,
    });
    if (this.openQuestions.length > 30) {
      this.openQuestions = this.openQuestions
        .filter((q) => !q.answered)
        .concat(this.openQuestions.filter((q) => q.answered).slice(-10));
    }
  }

  markAnswered(topic, userReply) {
    let hit = false;
    for (const q of this.openQuestions) {
      if (q.topic === topic || (userReply && String(userReply).includes(q.topic))) {
        q.answered = true;
        q.answeredAt = Date.now();
        this.answeredQuestions.push({ ...q });
        hit = true;
      }
    }
    if (this.answeredQuestions.length > 50) this.answeredQuestions.shift();
    if (hit) {
      const gain = this.informationGain.get(topic);
      if (gain) gain.lastExplored = Date.now();
    }
    return hit;
  }

  getOpenQuestions(topK = 5) {
    return this.openQuestions
      .filter((q) => !q.answered)
      .sort((a, b) => b.priority - a.priority)
      .slice(0, topK);
  }

  toPromptBlock(memory, selfModel, context = {}) {
    const qs = this.generateCuriousQuestions(context, memory);
    const open = this.getOpenQuestions(2);
    const lines = [];
    if (qs[0]) {
      lines.push(`好奇：${qs[0].question.slice(0, 72)}`);
    }
    if (open[0] && open[0].topic !== qs[0]?.topic) {
      lines.push(`未解：关于「${open[0].topic}」还没弄清楚`);
    }
    const topGain = [...this.informationGain.entries()]
      .sort((a, b) => b[1].gain - a[1].gain)[0];
    if (topGain && topGain[1].gain > 0.55) {
      lines.push(`信息增益最高：${topGain[0]}(${topGain[1].gain.toFixed(2)})`);
    }
    return lines.join('\n');
  }

  load(data) {
    if (!data) return;
    if (data.topicGraph) this.topicGraph = new Map(Object.entries(data.topicGraph));
    if (data.informationGain) this.informationGain = new Map(Object.entries(data.informationGain));
    if (data.openQuestions) this.openQuestions = data.openQuestions;
    if (data.answeredQuestions) this.answeredQuestions = data.answeredQuestions;
    if (data.clusterAffinity) this.clusterAffinity = { ...this.clusterAffinity, ...data.clusterAffinity };
  }

  snapshot() {
    return {
      topicGraph: Object.fromEntries(this.topicGraph),
      informationGain: Object.fromEntries(this.informationGain),
      openQuestions: this.openQuestions.slice(-20),
      answeredCount: this.answeredQuestions.length,
      topOpen: this.getOpenQuestions(3),
    };
  }
}

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

module.exports = { CuriosityEngine };
