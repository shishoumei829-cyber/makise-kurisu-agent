'use strict';

const { PERSONALITY_MILESTONES } = require('./constants');

function clamp01(v) {
  return Math.max(0, Math.min(1, v));
}

/**
 * 人格轨迹：记录特质漂移、里程碑、与事件耦合的演化叙事。
 */
class PersonalityTrajectory {
  constructor() {
    this.traits = {
      openness: 0.62,
      conscientiousness: 0.58,
      extraversion: 0.42,
      agreeableness: 0.48,
      neuroticism: 0.55,
    };
    this.milestones = [];
    this.driftLog = [];
    this._lastEventType = 'neutral';
  }

  ingestExternalTraits(externalTraits = {}) {
    for (const [k, v] of Object.entries(externalTraits)) {
      if (typeof this.traits[k] === 'number' && typeof v === 'number') {
        this.traits[k] = clamp01(this.traits[k] * 0.7 + v * 0.3);
      }
    }
  }

  updateFromEvent(event = {}) {
    const type = event.type || 'neutral';
    const content = String(event.content || '');
    const delta = {};

    if (type === 'scientific' || /物理|实验|理论|论文/.test(content)) {
      delta.openness = 0.012;
      delta.conscientiousness = 0.008;
    }
    if (type === 'intimate' || type === 'positive') {
      delta.extraversion = 0.01;
      delta.agreeableness = 0.008;
      delta.neuroticism = -0.006;
    }
    if (type === 'negative' || type === 'conflict') {
      delta.neuroticism = 0.015;
      delta.extraversion = -0.008;
    }
    if (/孤独|一个人|没人/.test(content)) {
      delta.extraversion = -0.01;
      delta.openness = 0.005;
    }

    this._applyDelta(delta, `event:${type}`);
    this._lastEventType = type;
    this._checkMilestones();
    return { traits: { ...this.traits }, delta };
  }

  _applyDelta(delta, source) {
    const applied = {};
    for (const [k, d] of Object.entries(delta)) {
      if (typeof this.traits[k] !== 'number') continue;
      const prev = this.traits[k];
      this.traits[k] = clamp01(prev + d);
      applied[k] = this.traits[k] - prev;
    }
    if (Object.keys(applied).length) {
      this.driftLog.push({ at: Date.now(), source, applied });
      if (this.driftLog.length > 80) this.driftLog = this.driftLog.slice(-80);
    }
  }

  _checkMilestones() {
    for (const [trait, cfg] of Object.entries(PERSONALITY_MILESTONES)) {
      const val = this.traits[trait];
      if (val == null) continue;
      const hitHigh = cfg.high != null && val >= cfg.high;
      const hitLow = cfg.low != null && val <= cfg.low;
      if (!hitHigh && !hitLow) continue;
      const key = `${trait}_${hitHigh ? 'high' : 'low'}`;
      if (this.milestones.some((m) => m.key === key)) continue;
      this.milestones.push({
        key,
        trait,
        label: cfg.label,
        value: val,
        at: Date.now(),
      });
      if (this.milestones.length > 20) this.milestones = this.milestones.slice(-20);
    }
  }

  getDescription() {
    const parts = [];
    if (this.traits.openness > 0.7) parts.push('更愿意探索新话题');
    if (this.traits.neuroticism > 0.65) parts.push('情绪更敏感');
    if (this.traits.extraversion < 0.38) parts.push('偏独处式回应');
    if (this.traits.agreeableness > 0.6) parts.push('语气更柔和');
    const latest = this.milestones[this.milestones.length - 1];
    if (latest) parts.push(latest.label);
    return parts.length ? parts.join('；') : '人格轨迹平稳';
  }

  toPromptLine() {
    const desc = this.getDescription();
    return desc ? `人格漂移：${desc}` : '';
  }

  load(data) {
    if (!data) return;
    if (data.traits) this.traits = { ...this.traits, ...data.traits };
    if (data.milestones) this.milestones = data.milestones;
    if (data.driftLog) this.driftLog = data.driftLog;
  }

  snapshot() {
    return {
      traits: { ...this.traits },
      milestones: this.milestones.slice(-5),
      driftLog: this.driftLog.slice(-8),
      description: this.getDescription(),
    };
  }
}

module.exports = { PersonalityTrajectory };
