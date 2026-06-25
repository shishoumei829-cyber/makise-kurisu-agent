'use strict';

/**
 * 强化学习桥接：将对话奖励与数字生命子系统状态耦合，输出行为偏置建议。
 */
class RlBridge {
  constructor() {
    this.lastStateKey = '';
    this.lastAction = '';
    this.lastReward = 0;
    this.rewardWindow = [];
    this.behaviorHints = {};
  }

  /**
   * @param {import('../../learning_engine').ReinforcementLearning} rl
   */
  recordTurn(rl, ctx = {}) {
    if (!rl) return null;

    const {
      pad,
      relScore,
      behaviorId,
      userText,
      recognizedEmotion,
      goalAchieved,
      relationshipDelta,
    } = ctx;

    const stateKey = rl.buildStateKey(pad, relScore);
    const reward = rl.calculateReward({
      userReaction: recognizedEmotion?.emotion === 'positive' ? 'positive'
        : recognizedEmotion?.emotion === 'negative' ? 'negative' : 'neutral',
      relationshipChange: relationshipDelta || 0,
      goalAchieved: Boolean(goalAchieved),
      emotionChange: recognizedEmotion?.intensity || 0,
      userReplyLength: String(userText || '').length,
      userEmotion: recognizedEmotion?.emotion || 'neutral',
      userAskedQuestion: /[?？]/.test(String(userText || '')),
    });

    if (this.lastStateKey && this.lastAction) {
      rl.updatePolicy(this.lastStateKey, this.lastAction, reward, stateKey);
    }
    if (behaviorId) {
      rl.updateBehaviorBiasFromReward(behaviorId, reward);
    }

    this.lastStateKey = stateKey;
    this.lastAction = behaviorId || this.lastAction;
    this.lastReward = reward;
    this.rewardWindow.push({ reward, at: Date.now(), behaviorId });
    if (this.rewardWindow.length > 50) this.rewardWindow.shift();

    return { stateKey, reward };
  }

  /**
   * @param {import('../../learning_engine').ReinforcementLearning} rl
   */
  behaviorBoosts(rl, possibleActions = []) {
    if (!rl || !possibleActions.length) return {};
    const stateKey = this.lastStateKey || 'P0_A0_R0';
    const chosen = rl.selectAction(stateKey, possibleActions);
    const boosts = {};
    for (const a of possibleActions) {
      boosts[a] = a === chosen ? 0.08 : 0;
      boosts[a] += rl.getBehaviorBias(a) || 0;
    }
    this.behaviorHints = boosts;
    return boosts;
  }

  averageReward() {
    if (!this.rewardWindow.length) return 0;
    return this.rewardWindow.reduce((s, r) => s + r.reward, 0) / this.rewardWindow.length;
  }

  load(data) {
    if (!data) return;
    if (data.lastStateKey) this.lastStateKey = data.lastStateKey;
    if (data.lastAction) this.lastAction = data.lastAction;
    if (data.rewardWindow) this.rewardWindow = data.rewardWindow;
  }

  snapshot() {
    return {
      lastStateKey: this.lastStateKey,
      lastAction: this.lastAction,
      lastReward: this.lastReward,
      averageReward: this.averageReward(),
      rewardWindow: this.rewardWindow.slice(-10),
      behaviorHints: this.behaviorHints,
    };
  }
}

module.exports = { RlBridge };
