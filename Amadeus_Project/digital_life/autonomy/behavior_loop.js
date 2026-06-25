'use strict';

const { AUTONOMY_ACTIONS } = require('./constants');

/**
 * 自主行为环：把内驱力冲动转化为可执行意图（开口/等待/内化），并产出目标种子
 */
class AutonomousBehaviorLoop {
  constructor(driveDynamics, curiosity, creativity) {
    this.drive = driveDynamics;
    this.curiosity = curiosity;
    this.creativity = creativity;
    this.lastDecision = null;
    this.decisionLog = [];
  }

  /**
   * @param {object} ctx
   * @returns 自主行为决策
   */
  execute(ctx = {}) {
    const {
      isAutonomyTick = false,
      idleMs = 0,
      pad = {},
      relScore = 0,
      userPresenceActive = false,
      dnd = false,
      proactiveQuotaOk = true,
      sheSpokeRecently = false,
      dreamCarryover = null,
    } = ctx;

    const idleMin = idleMs / 60000;
    const urges = this.drive.getActiveUrges();
    const primary = urges[0] || null;
    const energy = this.drive.internalState.energy;
    const social = this.drive.internalState.socialBattery;

    let action = AUTONOMY_ACTIONS.WAIT;
    let shouldAct = false;
    let speakHint = '';
    let suppressProactive = false;
    const goalSeeds = [];

    if (!primary && dreamCarryover?.proactiveEligible && idleMin >= 20) {
      action = AUTONOMY_ACTIONS.SPEAK;
      shouldAct = true;
      speakHint = `梦境残念：${dreamCarryover.hint || dreamCarryover.mood || '有话想说'}`;
      this._logDecision({ action, reason: 'dream_carryover', primary: null });
      return this._pack(action, shouldAct, speakHint, false, goalSeeds, null);
    }

    if (userPresenceActive || dnd) {
      action = AUTONOMY_ACTIONS.WAIT;
      suppressProactive = true;
      speakHint = '他之前说过别打扰——冲动在，但行为要克制';
      this._logDecision({ action, reason: 'presence_dnd', primary });
      return this._pack(action, shouldAct, speakHint, suppressProactive, goalSeeds, primary);
    }

    if (!proactiveQuotaOk || sheSpokeRecently) {
      action = AUTONOMY_ACTIONS.REFLECT;
      speakHint = '刚说过话或配额用尽——先内化';
      this._logDecision({ action, reason: 'quota_or_gap', primary });
      return this._pack(action, shouldAct, speakHint, true, goalSeeds, primary);
    }

    if (energy < 0.28 || social < 0.18) {
      action = AUTONOMY_ACTIONS.REFLECT;
      speakHint = '社交余量或能量偏低，更想安静';
      this._logDecision({ action, reason: 'low_energy', primary });
      return this._pack(action, false, speakHint, true, goalSeeds, primary);
    }

    if (!primary) {
      if (idleMin > 40 && relScore > 0.3 && energy > 0.4) {
        action = AUTONOMY_ACTIONS.SPEAK;
        shouldAct = Math.random() < 0.35;
        speakHint = '没有强冲动，但很久没聊——可能轻轻敲一句';
      } else {
        action = AUTONOMY_ACTIONS.WAIT;
      }
      this._logDecision({ action, reason: 'no_urge', primary });
      return this._pack(action, shouldAct, speakHint, !shouldAct, goalSeeds, primary);
    }

    const intent = primary.intentKey || primary.intent;
    const intensity = primary.effectiveIntensity();

    if (intent === 'STAY_SILENT' || intent === 'HOLD_BACK') {
      action = AUTONOMY_ACTIONS.REFLECT;
      shouldAct = false;
      speakHint = primary.promptHint;
      suppressProactive = intensity > 0.55;
    } else if (intent === 'REACH_OUT') {
      const threshold = Math.max(0.35, 0.55 - relScore * 0.15 - Math.min(idleMin / 120, 0.2));
      shouldAct = isAutonomyTick && idleMin >= 12 && intensity >= threshold;
      action = shouldAct ? AUTONOMY_ACTIONS.SPEAK : AUTONOMY_ACTIONS.WAIT;
      speakHint = primary.promptHint;
    } else if (intent === 'ASK_QUESTION' || intent === 'EXPLORE_TOPIC') {
      shouldAct = isAutonomyTick && intensity > 0.5 && idleMin >= 15 && Math.random() < 0.45 + intensity * 0.2;
      action = shouldAct ? AUTONOMY_ACTIONS.SPEAK : AUTONOMY_ACTIONS.REFLECT;
      speakHint = primary.promptHint;
      goalSeeds.push(this._goalFromUrge(primary, '好奇驱动'));
    } else if (intent === 'CREATE_IDEA') {
      const idea = this.creativity.generateIdea({
        pad,
        driveIntent: intent,
        lastEvent: primary.target,
      });
      shouldAct = isAutonomyTick && pad.A > 0.35 && intensity > 0.48 && Math.random() < 0.4;
      action = shouldAct ? AUTONOMY_ACTIONS.SPEAK : AUTONOMY_ACTIONS.REFLECT;
      speakHint = `创造冲动：${idea.slice(0, 80)}`;
      goalSeeds.push({
        id: 'CREATIVE_IMPULSE',
        label: '有个新念头',
        priority: intensity,
        prompt_injection: speakHint,
      });
    } else if (intent === 'DEEPEN_BOND' || intent === 'SELF_EXPRESSION') {
      shouldAct = isAutonomyTick && relScore > 0.25 && idleMin >= 18 && intensity > 0.45;
      shouldAct = shouldAct && Math.random() < 0.35 + relScore * 0.25;
      action = shouldAct ? AUTONOMY_ACTIONS.SPEAK : AUTONOMY_ACTIONS.WAIT;
      speakHint = primary.promptHint;
      goalSeeds.push(this._goalFromUrge(primary, '关系驱动'));
    } else if (intent === 'DEFEND_SELF' || intent === 'PLAYFUL_JAB') {
      shouldAct = isAutonomyTick && intensity > 0.52 && Math.random() < 0.25;
      action = shouldAct ? AUTONOMY_ACTIONS.SPEAK : AUTONOMY_ACTIONS.WAIT;
      speakHint = primary.promptHint;
    } else {
      shouldAct = isAutonomyTick && intensity > 0.5 && idleMin >= 20 && Math.random() < 0.3;
      action = shouldAct ? AUTONOMY_ACTIONS.SPEAK : AUTONOMY_ACTIONS.WAIT;
      speakHint = primary.promptHint;
    }

    if (pad.P < -0.35 && intent !== 'REACH_OUT') {
      shouldAct = false;
      action = AUTONOMY_ACTIONS.REFLECT;
      speakHint = '情绪低落——冲动在，但不想开口';
    }

    this._logDecision({ action, shouldAct, intent, intensity, idleMin });
    return this._pack(action, shouldAct, speakHint, suppressProactive, goalSeeds, primary);
  }

  _goalFromUrge(urge, labelPrefix) {
    return {
      id: `URGE_${urge.drive}`,
      label: `${labelPrefix}：${urge.target || urge.intent}`,
      priority: urge.effectiveIntensity(),
      turns_remaining: 3,
      behavior_hint: urge.promptHint,
      prompt_injection: urge.promptHint,
    };
  }

  _pack(action, shouldAct, speakHint, suppressProactive, goalSeeds, primaryUrge) {
    const decision = {
      action,
      shouldAct,
      speakHint,
      suppressProactive,
      goalSeeds,
      primaryUrge: primaryUrge ? {
        drive: primaryUrge.drive,
        intent: primaryUrge.intent,
        intensity: primaryUrge.effectiveIntensity(),
        target: primaryUrge.target,
      } : null,
      at: Date.now(),
    };
    this.lastDecision = decision;
    return decision;
  }

  _logDecision(entry) {
    this.decisionLog.push({ ...entry, at: Date.now() });
    if (this.decisionLog.length > 80) this.decisionLog.shift();
  }

  load(data) {
    if (!data) return;
    if (data.decisionLog) this.decisionLog = data.decisionLog.slice(-80);
    if (data.lastDecision) this.lastDecision = data.lastDecision;
  }

  snapshot() {
    return {
      lastDecision: this.lastDecision,
      decisionLog: this.decisionLog.slice(-10),
    };
  }
}

module.exports = { AutonomousBehaviorLoop };
