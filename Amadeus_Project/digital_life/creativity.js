'use strict';

/**
 * 创造性模块：联想、隐喻、新想法（启发式，不额外调 LLM）。
 */
class CreativityModule {
  constructor() {
    this.associations = new Map();
    this.metaphors = new Map([
      ['记忆', '像未整理的实验记录'],
      ['时间', '像不可逆的熵增'],
      ['情绪', '像神经递质浓度波动'],
      ['孤独', '像真空腔里的回声'],
      ['关系', '像两个纠缠粒子的弱耦合'],
    ]);
    this._ideaHistory = [];
  }

  _tokenize(text) {
    return (String(text || '').match(/[\u4e00-\u9fa5]{2,}/g) || []).slice(0, 12);
  }

  learnAssociation(a, b) {
    if (!a || !b || a === b) return;
    const key = [a, b].sort().join('|');
    this.associations.set(key, (this.associations.get(key) || 0) + 1);
  }

  generateMetaphor(concept) {
    const c = String(concept || '').trim();
    if (!c) return '';
    if (this.metaphors.has(c)) return this.metaphors.get(c);
    const hits = [...this.associations.keys()].filter((k) => k.includes(c));
    if (hits.length) {
      const pair = hits[0].split('|').find((x) => x !== c);
      return `有点像${pair}——但我不确定这比喻是否严谨`;
    }
    const fallbacks = ['未完成的假说', '噪声里的微弱信号', '还没跑完的模拟'];
    return fallbacks[Math.floor(Math.random() * fallbacks.length)];
  }

  generateAnalogy(source, target) {
    const s = String(source || '').slice(0, 20);
    const t = String(target || '').slice(0, 20);
    if (!s || !t) return '';
    return `如果把${s}想成${t}的结构，也许能换个角度看`;
  }

  generateIdea(context = {}) {
    const topics = this._tokenize(context.userText || context.lastEvent || '');
    const pad = context.pad || {};
    const ideas = [];
    if (topics.length >= 2) {
      ideas.push(`如果把「${topics[0]}」和「${topics[1]}」联系起来，会不会有新解释？`);
      this.learnAssociation(topics[0], topics[1]);
    }
    if (pad.A > 0.4 && topics[0]) {
      ideas.push(`关于${topics[0]}，我脑子里突然冒出一个不太严谨但有趣的角度`);
    }
    if (!ideas.length) {
      ideas.push('有一个模糊的念头，还没组织成能说出口的话');
    }
    const idea = ideas[0];
    this._ideaHistory.push({ idea, at: Date.now() });
    if (this._ideaHistory.length > 30) this._ideaHistory.shift();
    return idea;
  }

  toPromptLine(context) {
    if ((context.pad?.A || 0) < 0.35 && Math.random() > 0.4) return '';
    const idea = this.generateIdea(context);
    return idea ? `创造冲动：${idea.slice(0, 72)}` : '';
  }

  load(data) {
    if (!data) return;
    if (data.associations) this.associations = new Map(Object.entries(data.associations));
    if (data.metaphors) this.metaphors = new Map(Object.entries(data.metaphors));
  }

  snapshot() {
    return {
      associations: Object.fromEntries(this.associations),
      metaphors: Object.fromEntries(this.metaphors),
      recentIdeas: this._ideaHistory.slice(-5),
    };
  }
}

module.exports = { CreativityModule };
