'use strict';

const { REFLECTION_TRIGGERS } = require('./constants');

/**
 * 反思反馈环：决策回顾、偏见检测、洞察→目标注入。
 */
class ReflectionLoop {
  constructor() {
    this.reflectionHistory = [];
    this.insights = [];
    this.biases = [];
    this.behaviorCounts = {};
    this._lastInsightAt = 0;
  }

  reflectOnDecision(decision = {}) {
    const reflection = {
      action: decision.action || decision.behaviorId || 'unknown',
      reasoning: decision.reasoning || '',
      factors: decision.factors || decision.reasons || [],
      at: Date.now(),
      analysis: this._analyzeDecision(decision),
      possibleBiases: this._detectBiases(decision),
    };

    this.reflectionHistory.push(reflection);
    if (this.reflectionHistory.length > 60) this.reflectionHistory = this.reflectionHistory.slice(-60);

    const action = reflection.action;
    this.behaviorCounts[action] = (this.behaviorCounts[action] || 0) + 1;

    return reflection;
  }

  _analyzeDecision(decision) {
    const factors = decision.factors || decision.reasons || [];
    if (!factors.length) return '决策依据较少，可能偏直觉';
    const top = factors.slice(0, 2).join('、');
    return `主要受 ${top} 影响`;
  }

  _detectBiases(decision) {
    const biases = [];
    const action = decision.action || decision.behaviorId || '';
    const count = this.behaviorCounts[action] || 0;

    if (count >= 5 && count / Math.max(1, this.reflectionHistory.length) > REFLECTION_TRIGGERS.behavior_repeat) {
      biases.push({ type: 'habit', description: `近期频繁选择 ${action}` });
    }
    if (/回避|撤退|冷淡/.test(decision.reasoning || '')) {
      biases.push({ type: 'avoidance', description: '可能在回避亲近' });
    }
    if (/套话|模板/.test(decision.reasoning || '')) {
      biases.push({ type: 'scripted', description: '表达可能过于模板化' });
    }

    if (biases.length) {
      this.biases.push(...biases.map((b) => ({ ...b, at: Date.now() })));
      if (this.biases.length > 30) this.biases = this.biases.slice(-30);
    }
    return biases;
  }

  generateInsight(beliefRevision = null) {
    const now = Date.now();
    if (now - this._lastInsightAt < 5 * 60 * 1000) return null;

    let content = '';
    const recent = this.reflectionHistory.slice(-8);
    const dominant = Object.entries(this.behaviorCounts).sort((a, b) => b[1] - a[1])[0];

    if (dominant && dominant[1] >= 4) {
      content = `我好像总在用「${dominant[0]}」应对，要不要换种节奏`;
    } else if (this.biases.length) {
      content = this.biases[this.biases.length - 1].description;
    } else if (beliefRevision?.lowConfidenceBeliefs) {
      const low = beliefRevision.lowConfidenceBeliefs();
      if (low.length) {
        content = `对「${low[0].domain}」还不太确定：${low[0].text}`;
      }
    } else if (recent.length >= 3) {
      content = `最近三轮决策：${recent.map((r) => r.action).join('→')}`;
    }

    if (!content) return null;

    const insight = { content, at: now, source: 'reflection_loop' };
    this.insights.push(insight);
    if (this.insights.length > 40) this.insights = this.insights.slice(-40);
    this._lastInsightAt = now;
    return insight;
  }

  insightToGoalInjection(insight) {
    if (!insight?.content) return '';
    return `内心修正：${insight.content}（不必明说，调整语气与节奏即可）`;
  }

  shouldSurfaceInsight(chatTurnCounter, chatMinimal = true) {
    if (chatMinimal) return chatTurnCounter % 8 === 0;
    return chatTurnCounter % 5 === 0;
  }

  load(data) {
    if (!data) return;
    if (data.reflectionHistory) this.reflectionHistory = data.reflectionHistory;
    if (data.insights) this.insights = data.insights;
    if (data.biases) this.biases = data.biases;
    if (data.behaviorCounts) this.behaviorCounts = data.behaviorCounts;
    if (data._lastInsightAt) this._lastInsightAt = data._lastInsightAt;
  }

  snapshot() {
    return {
      reflectionHistory: this.reflectionHistory.slice(-6),
      insights: this.insights.slice(-6),
      biases: this.biases.slice(-5),
      behaviorCounts: this.behaviorCounts,
    };
  }
}

module.exports = { ReflectionLoop };
