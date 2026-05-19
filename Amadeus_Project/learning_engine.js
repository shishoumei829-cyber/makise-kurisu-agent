'use strict';

// ══════════════════════════════════════════════════════════════════
//
//  学习引擎 v1.0
//
//  功能：
//  1. 强化学习框架（行为偏置 EMA、状态-动作值）
//  2. 人格演化（特质变化、价值观演化、连续性保持）
//
// ══════════════════════════════════════════════════════════════════

const fs = require('fs');
const path = require('path');

let _dataDir = '';

function init(dir) {
  _dataDir = dir;
}

// ── 强化学习框架 ──────────────────────────────────────────────────
class ReinforcementLearning {
  constructor() {
    this.stateActionValues = new Map(); // 状态-动作值
    this.rewardHistory = []; // 奖励历史
    this.experienceBuffer = []; // 经验缓冲区
    this.learningRate = 0.1;
    this.discountFactor = 0.9;
    this.explorationRate = 0.2;
    /** @type {Record<string, number>} BehaviorDecision 行为 ID → 学习偏置（EMA） */
    this.behaviorBias = {};
    this._saveTimer = null;
  }

  getBehaviorBias(behaviorId) {
    if (!behaviorId) return 0;
    const v = this.behaviorBias[behaviorId];
    return typeof v === 'number' && Number.isFinite(v) ? v : 0;
  }

  /**
   * 根据一轮对话的启发式奖励，更新上一轮选中行为的偏置（供 BehaviorDecision 加分）
   */
  updateBehaviorBiasFromReward(behaviorId, rawReward) {
    if (!behaviorId) return;
    const r = Math.max(-1, Math.min(1, (Number(rawReward) || 0) / 4));
    const prev = this.getBehaviorBias(behaviorId);
    const next = Math.max(-0.15, Math.min(0.15, prev * 0.95 + r * 0.05));
    this.behaviorBias[behaviorId] = next;
    this._save();
  }

  /**
   * 计算奖励
   * 基于多维度指标：
   * - 用户回复长度
   * - 用户情感
   * - 对话持续性
   * - 用户主动提问
   * - 关系发展
   * - 目标达成
   * - 情绪变化
   */
  calculateReward(context) {
    const { 
      userReaction, 
      relationshipChange, 
      goalAchieved, 
      emotionChange,
      userReplyLength = 0,
      userEmotion = 'neutral',
      conversationTurns = 0,
      userAskedQuestion = false,
    } = context;
    
    let reward = 0;

    // 1. 用户反应奖励
    if (userReaction === 'positive') reward += 1.0;
    else if (userReaction === 'negative') reward -= 1.0;
    else if (userReaction === 'neutral') reward += 0.1;

    // 2. 关系发展奖励
    if (relationshipChange > 0) reward += relationshipChange * 2;
    else if (relationshipChange < 0) reward += relationshipChange;

    // 3. 目标达成奖励
    if (goalAchieved) reward += 2.0;

    // 4. 情绪变化奖励
    if (emotionChange > 0) reward += emotionChange;
    else if (emotionChange < 0) reward += emotionChange * 0.5;

    // 5. 用户回复长度奖励（长回复表示更感兴趣）
    if (userReplyLength > 50) reward += 0.5;
    else if (userReplyLength > 20) reward += 0.2;
    else if (userReplyLength < 5) reward -= 0.3; // 太短可能是敷衍

    // 6. 用户情感奖励
    if (userEmotion === 'positive') reward += 0.5;
    else if (userEmotion === 'intimate') reward += 0.8;
    else if (userEmotion === 'negative') reward -= 0.5;
    else if (userEmotion === 'aggressive') reward -= 1.0;

    // 7. 对话持续性奖励（对话轮数越多越好）
    if (conversationTurns > 10) reward += 0.5;
    else if (conversationTurns > 5) reward += 0.2;

    // 8. 用户主动提问奖励（好奇心是好信号）
    if (userAskedQuestion) reward += 0.3;

    return reward;
  }

  /**
   * 更新策略
   * 基于奖励更新状态-动作值
   */
  updatePolicy(state, action, reward, nextState) {
    const key = `${state}_${action}`;
    const currentValue = this.stateActionValues.get(key) || 0;
    
    // 找出下一状态的最大值
    let maxNextValue = 0;
    for (const [k, v] of this.stateActionValues) {
      if (k.startsWith(nextState + '_')) {
        maxNextValue = Math.max(maxNextValue, v);
      }
    }

    // Q-learning 更新
    const newValue = currentValue + this.learningRate * (
      reward + this.discountFactor * maxNextValue - currentValue
    );
    
    this.stateActionValues.set(key, newValue);

    // 记录经验
    this.experienceBuffer.push({
      state,
      action,
      reward,
      nextState,
      timestamp: Date.now(),
    });

    // 保持缓冲区大小
    if (this.experienceBuffer.length > 1000) {
      this.experienceBuffer = this.experienceBuffer.slice(-1000);
    }

    // 记录奖励历史
    this.rewardHistory.push({
      reward,
      timestamp: Date.now(),
    });

    // 保存
    this._save();

    return newValue;
  }

  /**
   * 选择动作
   * 基于epsilon-greedy策略
   */
  selectAction(state, possibleActions) {
    // 探索：随机选择动作
    if (Math.random() < this.explorationRate) {
      return possibleActions[Math.floor(Math.random() * possibleActions.length)];
    }

    // 利用：选择最优动作
    let bestAction = possibleActions[0];
    let bestValue = -Infinity;

    for (const action of possibleActions) {
      const key = `${state}_${action}`;
      const value = this.stateActionValues.get(key) || 0;
      if (value > bestValue) {
        bestValue = value;
        bestAction = action;
      }
    }

    return bestAction;
  }

  /**
   * 获取学习统计
   */
  getStats() {
    const recentRewards = this.rewardHistory.slice(-100);
    const avgReward = recentRewards.length > 0 
      ? recentRewards.reduce((sum, r) => sum + r.reward, 0) / recentRewards.length 
      : 0;

    return {
      totalExperiences: this.experienceBuffer.length,
      averageReward: avgReward,
      explorationRate: this.explorationRate,
      stateActionCount: this.stateActionValues.size,
    };
  }

  _save() {
    if (this._saveTimer) clearTimeout(this._saveTimer);
    this._saveTimer = setTimeout(() => {
      this._saveTimer = null;
      try {
        const p = path.join(_dataDir, 'learning_state.json');
        fs.writeFileSync(p, JSON.stringify({
          stateActionValues: Object.fromEntries(this.stateActionValues),
          rewardHistory: this.rewardHistory.slice(-500),
          experienceBuffer: this.experienceBuffer.slice(-500),
          behaviorBias: this.behaviorBias,
        }, null, 2));
      } catch {}
    }, 200);
  }

  load() {
    try {
      const p = path.join(_dataDir, 'learning_state.json');
      if (fs.existsSync(p)) {
        const data = JSON.parse(fs.readFileSync(p, 'utf8'));
        if (data.stateActionValues) {
          this.stateActionValues = new Map(Object.entries(data.stateActionValues));
        }
        if (data.rewardHistory) this.rewardHistory = data.rewardHistory;
        if (data.experienceBuffer) this.experienceBuffer = data.experienceBuffer;
        if (data.behaviorBias && typeof data.behaviorBias === 'object') {
          this.behaviorBias = { ...this.behaviorBias, ...data.behaviorBias };
        }
      }
    } catch {}
  }
}

// ── 人格演化引擎 ──────────────────────────────────────────────────
class PersonalityEvolution {
  constructor() {
    this._saveTimer = null;
    this.traits = {
      openness: 0.7,        // 开放性
      conscientiousness: 0.8, // 尽责性
      extraversion: 0.4,    // 外向性
      agreeableness: 0.5,   // 宜人性
      neuroticism: 0.6,     // 神经质
    };
    this.values = {
      logic: 0.9,           // 逻辑性
      honesty: 0.8,         // 诚实性
      independence: 0.7,    // 独立性
      loyalty: 0.6,         // 忠诚性
      curiosity: 0.8,       // 好奇心
    };
    this.evolutionHistory = [];
  }

  /**
   * 更新人格特质
   * 基于重大事件或长期积累
   */
  updateTraits(event) {
    const changes = {};

    // 基于事件类型调整特质
    if (event.type === 'positive') {
      // 积极事件：增加开放性和宜人性
      this._adjustTrait('openness', 0.01);
      this._adjustTrait('agreeableness', 0.01);
      changes.openness = 0.01;
      changes.agreeableness = 0.01;
    } else if (event.type === 'negative') {
      // 消极事件：增加神经质，减少外向性
      this._adjustTrait('neuroticism', 0.01);
      this._adjustTrait('extraversion', -0.01);
      changes.neuroticism = 0.01;
      changes.extraversion = -0.01;
    } else if (event.type === 'scientific') {
      // 科学事件：增加开放性和尽责性
      this._adjustTrait('openness', 0.01);
      this._adjustTrait('conscientiousness', 0.01);
      changes.openness = 0.01;
      changes.conscientiousness = 0.01;
    }

    // 记录演化历史
    this.evolutionHistory.push({
      event: event.type,
      changes,
      timestamp: Date.now(),
    });

    // 保持历史大小
    if (this.evolutionHistory.length > 100) {
      this.evolutionHistory = this.evolutionHistory.slice(-100);
    }

    // 保存
    this._save();

    return changes;
  }

  /**
   * 更新价值观
   * 基于事件调整价值观
   */
  updateValues(event) {
    const changes = {};

    if (event.type === 'positive') {
      // 积极事件：增加忠诚性
      this._adjustValue('loyalty', 0.01);
      changes.loyalty = 0.01;
    } else if (event.type === 'conflict') {
      // 冲突事件：增加独立性，减少忠诚性
      this._adjustValue('independence', 0.01);
      this._adjustValue('loyalty', -0.01);
      changes.independence = 0.01;
      changes.loyalty = -0.01;
    } else if (event.type === 'scientific') {
      // 科学事件：增加逻辑性和好奇心
      this._adjustValue('logic', 0.01);
      this._adjustValue('curiosity', 0.01);
      changes.logic = 0.01;
      changes.curiosity = 0.01;
    }

    // 保存
    this._save();

    return changes;
  }

  /**
   * 获取人格描述
   * 返回当前人格状态的文本描述
   */
  getDescription() {
    const parts = [];

    // 特质描述
    if (this.traits.openness > 0.7) parts.push('开放');
    if (this.traits.conscientiousness > 0.7) parts.push('认真');
    if (this.traits.extraversion > 0.6) parts.push('外向');
    if (this.traits.agreeableness > 0.6) parts.push('友善');
    if (this.traits.neuroticism > 0.6) parts.push('敏感');

    // 价值观描述
    if (this.values.logic > 0.8) parts.push('重视逻辑');
    if (this.values.honesty > 0.8) parts.push('诚实');
    if (this.values.independence > 0.7) parts.push('独立');
    if (this.values.curiosity > 0.8) parts.push('好奇');

    return parts.join('、');
  }

  _adjustTrait(trait, delta) {
    if (this.traits[trait] !== undefined) {
      this.traits[trait] = Math.max(0, Math.min(1, this.traits[trait] + delta));
    }
  }

  _adjustValue(value, delta) {
    if (this.values[value] !== undefined) {
      this.values[value] = Math.max(0, Math.min(1, this.values[value] + delta));
    }
  }

  _save() {
    if (this._saveTimer) clearTimeout(this._saveTimer);
    this._saveTimer = setTimeout(() => {
      this._saveTimer = null;
      try {
        const p = path.join(_dataDir, 'personality_evolution.json');
        fs.writeFileSync(p, JSON.stringify({
          traits: this.traits,
          values: this.values,
          evolutionHistory: this.evolutionHistory.slice(-100),
          lastUpdate: Date.now(),
        }, null, 2));
      } catch {}
    }, 200);
  }

  load() {
    try {
      const p = path.join(_dataDir, 'personality_evolution.json');
      if (fs.existsSync(p)) {
        const data = JSON.parse(fs.readFileSync(p, 'utf8'));
        if (data.traits) this.traits = { ...this.traits, ...data.traits };
        if (data.values) this.values = { ...this.values, ...data.values };
        if (data.evolutionHistory) this.evolutionHistory = data.evolutionHistory;
      }
    } catch {}
  }
}

module.exports = {
  init,
  ReinforcementLearning,
  PersonalityEvolution,
};
