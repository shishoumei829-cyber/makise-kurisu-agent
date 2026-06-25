'use strict';

/** 兼容旧测试与导入 */
const { DriveDynamics, AutonomySubsystem, DRIVE_TYPES } = require('./autonomy');

/** @deprecated 请使用 AutonomySubsystem */
class AutonomousBehaviorEngine {
  constructor() {
    this._sub = new AutonomySubsystem();
  }

  updateInternalState(pad, memory, motivationState = {}) {
    const relScore = memory?.getRelationshipScore?.() ?? 0;
    const recent = memory?.events?.slice(-3).map((e) => e.content).join('') || '';
    this._sub.drives.tick(0, {
      pad,
      memory,
      motivationState,
      relScore,
      userText: recent,
    });
    this._sub.drives.generateUrges({ pad, relScore, memory, userText: recent });
  }

  generateUrge() {
    const u = this._sub.drives.getActiveUrges()[0];
    if (!u) return null;
    return { drive: u.drive, label: DRIVE_TYPES[u.drive], strength: u.effectiveIntensity() };
  }

  behaviorBoosts() {
    return this._sub.drives.behaviorBoostsFromUrges();
  }

  toPromptLine() {
    return this._sub.drives.toPromptBlock();
  }

  load(data) {
    this._sub.drives.load(data);
  }

  snapshot() {
    return this._sub.drives.snapshot();
  }
}

module.exports = { AutonomousBehaviorEngine, DRIVE_TYPES };
