'use strict';

const { TOPIC_CLUSTERS } = require('./constants');

const CLUSTER_IMAGERY = {
  science: ['实验台', '未完成的推导', '噪声里的信号', '对照组'],
  emotion: ['潮汐', '未寄出的信', '薄雾'],
  daily: ['便利店灯光', '空杯子', '晚班电车'],
  meta: ['递归的镜子', '未证明的命题', '边界条件'],
  relationship: ['不同步的钟', '安全距离', '共振'],
  general: ['半成品', '侧信道', '余温'],
};

const KURISU_FILTERS = [/可爱|甜甜|宝贝|亲亲/, /总之呢|综上所述/];

/**
 * 创造性模块 v2：联想网络、跨簇隐喻、可控新想法、人格过滤
 */
class CreativityModule {
  constructor() {
    this.associations = new Map();
    this.metaphors = new Map([
      ['记忆', '未归档的实验日志'],
      ['时间', '不可逆过程里的熵'],
      ['情绪', '神经调质浓度在波动'],
      ['孤独', '真空腔里的回声'],
      ['关系', '弱耦合系统的共振'],
      ['思考', '尚未收敛的迭代'],
    ]);
    this.ideaHistory = [];
    this.ideaBank = [];
  }

  learnAssociation(a, b, weight = 1) {
    if (!a || !b || a === b) return;
    const key = [a, b].sort().join('|');
    this.associations.set(key, (this.associations.get(key) || 0) + weight);
    this._strengthenClusterBridge(a, b);
  }

  _strengthenClusterBridge(a, b) {
    const ca = this._cluster(a);
    const cb = this._cluster(b);
    if (ca !== cb && ca !== 'general' && cb !== 'general') {
      const bridge = `bridge:${ca}:${cb}`;
      this.associations.set(bridge, (this.associations.get(bridge) || 0) + 0.5);
    }
  }

  _cluster(term) {
    for (const [name, re] of Object.entries(TOPIC_CLUSTERS)) {
      if (re.test(term)) return name;
    }
    return 'general';
  }

  _tokens(text) {
    return (String(text || '').match(/[\u4e00-\u9fa5]{2,}/g) || []).slice(0, 10);
  }

  generateMetaphor(concept) {
    const c = String(concept || '').trim();
    if (!c) return '';
    if (this.metaphors.has(c)) return this.metaphors.get(c);

    const cluster = this._cluster(c);
    const imagery = CLUSTER_IMAGERY[cluster] || CLUSTER_IMAGERY.general;
    const img = imagery[Math.floor(Math.random() * imagery.length)];

    const related = [...this.associations.keys()]
      .filter((k) => k.includes(c) && !k.startsWith('bridge:'))
      .sort((a, b) => (this.associations.get(b) || 0) - (this.associations.get(a) || 0))
      .slice(0, 1);

    if (related.length) {
      const other = related[0].split('|').find((x) => x !== c);
      return `有点像${other}，但更像${img}——不严谨，只是直觉`;
    }
    return `像${img}，但我不确定这比喻是否成立`;
  }

  generateAnalogy(source, target) {
    const s = String(source || '').slice(0, 24);
    const t = String(target || '').slice(0, 24);
    if (!s || !t) return '';
    const cs = this._cluster(s);
    const ct = this._cluster(t);
    if (cs === ct) {
      return `${s}和${t}在同一类问题里，也许共享某个结构`;
    }
    return `如果把${s}的框架借到${t}上，可能会看见不同的失败模式`;
  }

  _noveltyScore(idea, recent = []) {
    let score = 1;
    for (const r of recent) {
      const overlap = [...idea].filter((ch) => r.includes(ch)).length;
      score -= Math.min(0.35, overlap / Math.max(idea.length, 1));
    }
    return Math.max(0, score);
  }

  _passesPersona(idea) {
    return !KURISU_FILTERS.some((re) => re.test(idea));
  }

  generateIdea(context = {}) {
    const tokens = this._tokens(context.userText || context.lastEvent || '');
    const pad = context.pad || {};
    const ideas = [];

    if (tokens.length >= 2) {
      this.learnAssociation(tokens[0], tokens[1], 1.2);
      ideas.push({
        text: `如果把「${tokens[0]}」和「${tokens[1]}」连起来，会不会有个新解释？`,
        type: 'combination',
        weight: 0.7,
      });
    }
    if (tokens[0]) {
      const meta = this.generateMetaphor(tokens[0]);
      ideas.push({ text: `${tokens[0]}……${meta}`, type: 'metaphor', weight: 0.65 });
    }
    if (pad.A > 0.42 && tokens[0] && tokens[1]) {
      ideas.push({
        text: this.generateAnalogy(tokens[0], tokens[1]),
        type: 'analogy',
        weight: 0.75 + pad.A * 0.1,
      });
    }
    if (context.driveIntent === 'CREATE_IDEA' || pad.A > 0.5) {
      ideas.push({
        text: '有个模糊念头还没组织成能说的话——可以先试探性抛一半',
        type: 'half_formed',
        weight: 0.6,
      });
    }

    const recent = this.ideaHistory.map((i) => i.text);
    const ranked = ideas
      .filter((i) => this._passesPersona(i.text))
      .map((i) => ({ ...i, novelty: this._noveltyScore(i.text, recent) }))
      .sort((a, b) => (b.weight * b.novelty) - (a.weight * a.novelty));

    const pick = ranked[0] || { text: '脑子里有东西在转，但还没成型', type: 'fallback' };
    const record = { ...pick, at: Date.now() };
    this.ideaHistory.push(record);
    if (this.ideaHistory.length > 40) this.ideaHistory.shift();
    this.ideaBank.push(record);
    if (this.ideaBank.length > 80) this.ideaBank = this.ideaBank.slice(-80);
    return pick.text;
  }

  toPromptBlock(context = {}) {
    const A = context.pad?.A || 0;
    if (A < 0.32 && Math.random() > 0.55) return '';
    const idea = this.generateIdea(context);
    return idea ? `创造：${idea.slice(0, 90)}` : '';
  }

  load(data) {
    if (!data) return;
    if (data.associations) this.associations = new Map(Object.entries(data.associations));
    if (data.metaphors) this.metaphors = new Map(Object.entries(data.metaphors));
    if (data.ideaHistory) this.ideaHistory = data.ideaHistory;
    if (data.ideaBank) this.ideaBank = data.ideaBank;
  }

  snapshot() {
    return {
      associations: Object.fromEntries(this.associations),
      metaphors: Object.fromEntries(this.metaphors),
      recentIdeas: this.ideaHistory.slice(-5),
      associationCount: this.associations.size,
    };
  }
}

module.exports = { CreativityModule };
