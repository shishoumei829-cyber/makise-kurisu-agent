'use strict';

const fs = require('fs');
const path = require('path');
const { EnvironmentSense } = require('./environment');
const { TimeSense } = require('./time_perception');
const { ExpressionMapper } = require('./expression_mapper');

/**
 * 模块四 · 具身化 — 环境、时间、PAD→表情
 */
class EmbodimentSubsystem {
  constructor() {
    this.environment = new EnvironmentSense();
    this.time = new TimeSense();
    this.expression = new ExpressionMapper();
    this._dataDir = '';
    this._saveTimer = null;
    this.moduleVersion = 2;
  }

  init(dataDir) {
    this._dataDir = dataDir;
    this.load();
  }

  _statePath() {
    return path.join(this._dataDir, 'embodiment_subsystem.json');
  }

  load() {
    try {
      const p = this._statePath();
      if (!fs.existsSync(p)) return;
      const data = JSON.parse(fs.readFileSync(p, 'utf8'));
      this.environment.load(data.environment);
      this.time.load(data.time);
      this.expression.load(data.expression);
    } catch { /* ignore */ }
  }

  _save() {
    if (!this._dataDir) return;
    if (this._saveTimer) clearTimeout(this._saveTimer);
    this._saveTimer = setTimeout(() => {
      this._saveTimer = null;
      try {
        fs.writeFileSync(this._statePath(), JSON.stringify({
          moduleVersion: this.moduleVersion,
          environment: this.environment.snapshot(),
          time: this.time.snapshot(),
          expression: this.expression.snapshot(),
          savedAt: Date.now(),
        }, null, 2));
      } catch { /* ignore */ }
    }, 200);
  }

  onVision(visionText) {
    const scene = this.environment.understandScene(visionText);
    this._save();
    return scene;
  }

  onConversationTurn(ctx = {}) {
    const { pad, userModel, idleMs = 0 } = ctx;
    const expression = this.expression.mapFromPad(pad, {
      userText: ctx.userText,
      mainEvent: ctx.mainEvent,
      recognized: ctx.recognized,
      pendingNeed: ctx.pendingNeed,
      relScore: ctx.relScore,
      now: ctx.now,
    });
    const timeLine = this.time.toPromptLine(userModel, idleMs);
    this._save();
    return { expression, timeLine };
  }

  buildPromptBlock(ctx = {}) {
    const lines = [];
    const timeLine = ctx.timeLine || this.time.toPromptLine(ctx.userModel, ctx.idleMs || 0);
    if (timeLine) lines.push(`时间感：${timeLine}`);
    const envLine = this.environment.toPromptLine();
    if (envLine) lines.push(envLine);
    return lines.filter(Boolean).join('\n');
  }

  getPublicState() {
    return {
      module: 'embodiment',
      version: this.moduleVersion,
      environment: this.environment.snapshot(),
      time: this.time.snapshot(),
      expression: this.expression.snapshot(),
    };
  }
}

module.exports = {
  EmbodimentSubsystem,
  EnvironmentSense,
  TimeSense,
  ExpressionMapper,
};
