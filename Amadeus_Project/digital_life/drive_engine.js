'use strict';

const { DRIVE_TYPES } = require('../autonomy_enhanced');

/**
 * 统一内驱力引擎：多种 drive 竞争，产出主导冲动与行为偏置。
 */
class AutonomousBehaviorEngine {
  constructor() {
    this.internalState = {
      energy: 1.0,
      mood: 0.0,
      curiosity: 0.5,
      creativity: 0.5,
    };
    this.driveLevels = {};
    for (const key of Object.keys(DRIVE_TYPES)) {
      this.driveLevels[key] = 0.35;
    }
    this.urgeQueue = [];
    this._lastDominant = null;
  }

  updateInternalState(pad, memory, motivationState = {}) {
    const { P = 0, A = 0, S = 0 } = pad || {};
    this.internalState.mood = P;
    this.internalState.energy = Math.max(0.2, Math.min(1, 0.6 + A * 0.3));
    this.internalState.curiosity = Math.max(0, Math.min(1, motivationState.curiosity ?? 0.5));
    this.internalState.creativity = Math.max(0, Math.min(1, 0.4 + A * 0.25));

    const rel = memory && typeof memory.getRelationshipScore === 'function'
      ? memory.getRelationshipScore()
      : 0;

    this.driveLevels.CURIOSITY = Math.min(1, this.internalState.curiosity + (A > 0.3 ? 0.15 : 0));
    this.driveLevels.CREATIVITY = Math.min(1, this.internalState.creativity);
    this.driveLevels.EXPLORATION = Math.min(1, 0.3 + A * 0.2);
    this.driveLevels.CONNECTION = Math.min(1, 0.25 + Math.max(0, rel) * 0.5 + S * 0.2);
    this.driveLevels.AUTONOMY = Math.min(1, 0.45 + (pad?.D > 0.4 ? 0.1 : 0));
    this.driveLevels.MASTERY = /科学|实验|理论/.test(memory?.events?.slice(-3).map((e) => e.content).join('') || '')
      ? 0.65 : 0.35;
    this.driveLevels.MEANING = Math.min(1, 0.3 + S * 0.25);
    this.driveLevels.PLAYFULNESS = P > 0.15 ? 0.5 : 0.25;
    this.driveLevels.PROTECTION = rel < 0 ? 0.55 : 0.3;
    this.driveLevels.CREATION = this.driveLevels.CREATIVITY * 0.9;
  }

  generateUrge() {
    const ranked = Object.entries(this.driveLevels)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3);
    this.urgeQueue = ranked.map(([key, strength]) => ({
      drive: key,
      label: DRIVE_TYPES[key] || key,
      strength,
    }));
    this._lastDominant = ranked[0] ? ranked[0][0] : null;
    return this.urgeQueue[0] || null;
  }

  /** @returns {Record<string, number>} behaviorId -> score boost */
  behaviorBoosts() {
    const boosts = {};
    const dom = this._lastDominant || this.generateUrge()?.drive;
    if (!dom) return boosts;

    const map = {
      CURIOSITY: { ENGAGE: 0.12, CASUAL: 0.06 },
      CREATIVITY: { ENGAGE: 0.1, APPROACH: 0.08 },
      EXPLORATION: { ENGAGE: 0.14, CASUAL: 0.05 },
      CONNECTION: { APPROACH: 0.18, CASUAL: 0.1 },
      AUTONOMY: { DEFEND: 0.1, DEFLECT: 0.08 },
      MASTERY: { ENGAGE: 0.2 },
      MEANING: { APPROACH: 0.12, ENGAGE: 0.06 },
      PLAYFULNESS: { CASUAL: 0.15, DEFLECT: 0.08 },
      PROTECTION: { DEFEND: 0.15, WITHDRAW: 0.1 },
      CREATION: { ENGAGE: 0.1, APPROACH: 0.06 },
    };
    const m = map[dom] || {};
    for (const [id, v] of Object.entries(m)) boosts[id] = v;
    return boosts;
  }

  toPromptLine() {
    const urge = this.urgeQueue[0] || this.generateUrge();
    if (!urge) return '';
    const secondary = this.urgeQueue.slice(1, 2).map((u) => u.label).join('、');
    return `内驱：${urge.label}(${urge.strength.toFixed(2)})${secondary ? `；次：${secondary}` : ''}`;
  }

  load(data) {
    if (!data || typeof data !== 'object') return;
    if (data.internalState) this.internalState = { ...this.internalState, ...data.internalState };
    if (data.driveLevels) this.driveLevels = { ...this.driveLevels, ...data.driveLevels };
    if (Array.isArray(data.urgeQueue)) this.urgeQueue = data.urgeQueue;
  }

  snapshot() {
    return {
      internalState: { ...this.internalState },
      driveLevels: { ...this.driveLevels },
      urgeQueue: this.urgeQueue.slice(0, 3),
      dominant: this._lastDominant,
    };
  }
}

module.exports = { AutonomousBehaviorEngine, DRIVE_TYPES };
