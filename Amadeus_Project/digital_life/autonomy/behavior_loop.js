'use strict';

const { AUTONOMY_ACTIONS } = require('./constants');
const { shouldSpeakNow, readSocialField } = require('../../cognitive/socialRead');

/**
 * 自主行为环：把内驱冲动变成「要不要开口」。
 * 时机看读场，不看随机骰和「空闲满 N 分钟」硬门。
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
      facePresent = false,
      social: socialIn = null,
    } = ctx;

    const idleMin = idleMs / 60000;
    const urges = this.drive.getActiveUrges();
    const primary = urges[0] || null;
    const energy = this.drive.internalState.energy;
    const socialBattery = this.drive.internalState.socialBattery;
    const social = socialIn || readSocialField({
      ...ctx,
      facePresent,
      idleMs,
      dnd,
    });

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
      return this._pack(action, shouldAct, speakHint, false, goalSeeds, null, social);
    }

    if (userPresenceActive || dnd) {
      action = AUTONOMY_ACTIONS.WAIT;
      suppressProactive = true;
      speakHint = '他之前说过别打扰——冲动在，但行为要克制';
      this._logDecision({ action, reason: 'presence_dnd', primary });
      return this._pack(action, shouldAct, speakHint, suppressProactive, goalSeeds, primary, social);
    }

    // 配额不再挡；刚说过话才先内化（留一口气）
    void proactiveQuotaOk;
    if (sheSpokeRecently) {
      action = AUTONOMY_ACTIONS.REFLECT;
      speakHint = '刚说过话——先喘口气';
      this._logDecision({ action, reason: 'just_spoke', primary });
      return this._pack(action, shouldAct, speakHint, true, goalSeeds, primary, social);
    }

    if (energy < 0.28 || socialBattery < 0.18) {
      action = AUTONOMY_ACTIONS.REFLECT;
      speakHint = '社交余量或能量偏低，更想安静';
      this._logDecision({ action, reason: 'low_energy', primary });
      return this._pack(action, false, speakHint, true, goalSeeds, primary, social);
    }

    if (!primary) {
      action = AUTONOMY_ACTIONS.WAIT;
      speakHint = facePresent
        ? '人在旁边，但心里还没攒够想说的话——可以只是坐着'
        : '没有强冲动，先待着';
      this._logDecision({ action, reason: 'no_urge', primary, social });
      return this._pack(action, false, speakHint, true, goalSeeds, null, social);
    }

    const intent = primary.intentKey || primary.intent;
    const intensity = primary.effectiveIntensity();

    if (intent === 'STAY_SILENT' || intent === 'HOLD_BACK') {
      action = AUTONOMY_ACTIONS.REFLECT;
      shouldAct = false;
      speakHint = primary.promptHint;
      suppressProactive = intensity > 0.55;
      this._logDecision({ action, reason: 'hold_back', intent, intensity });
      return this._pack(action, shouldAct, speakHint, suppressProactive, goalSeeds, primary, social);
    }

    if (pad.P < -0.35 && intent !== 'REACH_OUT') {
      this._logDecision({ action: AUTONOMY_ACTIONS.REFLECT, reason: 'low_mood', intent, intensity });
      return this._pack(
        AUTONOMY_ACTIONS.REFLECT,
        false,
        '情绪低落——冲动在，但不想开口',
        true,
        goalSeeds,
        primary,
        social,
      );
    }

    const gate = shouldSpeakNow(intensity, social, { relScore });
    if (!isAutonomyTick || !gate.ok) {
      action = AUTONOMY_ACTIONS.WAIT;
      speakHint = gate.reason === 'urge_not_ripe'
        ? `想说，但还没到时候（冲动 ${(intensity * 100).toFixed(0)}% / 需要 ${(gate.need * 100).toFixed(0)}%）`
        : primary.promptHint;
      this._logDecision({
        action,
        reason: gate.reason || 'not_now',
        intent,
        intensity,
        need: gate.need,
        social,
      });
      return this._pack(action, false, speakHint, false, goalSeeds, primary, social);
    }

    shouldAct = true;
    action = AUTONOMY_ACTIONS.SPEAK;
    speakHint = primary.promptHint;

    if (intent === 'ASK_QUESTION' || intent === 'EXPLORE_TOPIC') {
      goalSeeds.push(this._goalFromUrge(primary, '好奇驱动'));
    } else if (intent === 'CREATE_IDEA') {
      const idea = this.creativity.generateIdea({
        pad,
        driveIntent: intent,
        lastEvent: primary.target,
      });
      speakHint = `创造冲动：${idea.slice(0, 80)}`;
      goalSeeds.push({
        id: 'CREATIVE_IMPULSE',
        label: '有个新念头',
        priority: intensity,
        prompt_injection: speakHint,
      });
    } else if (intent === 'DEEPEN_BOND' || intent === 'SELF_EXPRESSION' || intent === 'REACH_OUT') {
      goalSeeds.push(this._goalFromUrge(primary, '关系驱动'));
    }

    this._logDecision({
      action,
      shouldAct,
      reason: gate.reason,
      intent,
      intensity,
      need: gate.need,
      idleMin,
      social,
    });
    return this._pack(action, shouldAct, speakHint, suppressProactive, goalSeeds, primary, social);
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

  _pack(action, shouldAct, speakHint, suppressProactive, goalSeeds, primaryUrge, social = null) {
    const decision = {
      action,
      shouldAct,
      speakHint,
      suppressProactive,
      goalSeeds,
      primaryUrge: primaryUrge ? {
        id: primaryUrge.id,
        drive: primaryUrge.drive,
        intent: primaryUrge.intent,
        intentKey: primaryUrge.intentKey,
        intensity: primaryUrge.effectiveIntensity(),
        target: primaryUrge.target,
        promptHint: primaryUrge.promptHint,
      } : null,
      social,
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
