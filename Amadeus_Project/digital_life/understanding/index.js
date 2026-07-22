'use strict';

const fs = require('fs');
const path = require('path');
const { EmotionalResonance } = require('./emotional_resonance');
const { MentalModel } = require('./mental_model');
const { SubtextDetector } = require('./subtext');

/**
 * 模块三 · 用户理解 — 情感共鸣、心智模型、言外之意
 */
class UnderstandingSubsystem {
  constructor() {
    this.resonance = new EmotionalResonance();
    this.mentalModel = new MentalModel();
    this.subtext = new SubtextDetector();
    this._dataDir = '';
    this._saveTimer = null;
    this._lastRecognized = null;
    this._lastSubtext = null;
    this.moduleVersion = 2;
  }

  init(dataDir) {
    this._dataDir = dataDir;
    this.load();
  }

  _statePath() {
    return path.join(this._dataDir, 'understanding_subsystem.json');
  }

  load() {
    try {
      const p = this._statePath();
      if (!fs.existsSync(p)) return;
      const data = JSON.parse(fs.readFileSync(p, 'utf8'));
      this.resonance.load(data.resonance);
      this.mentalModel.load(data.mentalModel);
      this.subtext.load(data.subtext);
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
          resonance: this.resonance.snapshot(),
          mentalModel: this.mentalModel.snapshot(),
          subtext: this.subtext.snapshot(),
          savedAt: Date.now(),
        }, null, 2));
      } catch { /* ignore */ }
    }, 200);
  }

  onConversationTurn(ctx = {}) {
    const { userText, userModel, closeness = 0 } = ctx;
    const emotionHistory = userModel?.model?.patterns?.emotion_history || [];

    const recognized = this.resonance.recognizeEmotion(userText, emotionHistory);
    const padDelta = this.resonance.adjustEmotionalState(recognized, closeness);
    const resonanceLine = this.resonance.toPromptLine(recognized, closeness);

    if (!ctx.memoryAdmission || ctx.memoryAdmission.allowProfile === true || ctx.memoryAdmission.allowInference === true) {
      this.mentalModel.ingestUserText(userText, userModel, ctx.memoryAdmission);
    }
    if (ctx.inferredBdi) this.mentalModel.applyInferredBdi(ctx.inferredBdi);
    const hypothesis = this.mentalModel.buildHypothesis(recognized);

    const subtextAnalysis = this.subtext.analyze(userText, recognized, {
      persist: !ctx.memoryAdmission || ctx.memoryAdmission.allowInference === true,
    });
    const subtextLine = this.subtext.toPromptLine(subtextAnalysis);

    this._lastRecognized = recognized;
    this._lastSubtext = subtextAnalysis;
    this._save();

    return {
      recognized,
      padDelta,
      resonanceLine,
      subtextLine,
      mentalModelLine: this.mentalModel.toPromptBlock(),
      hypothesis,
      pendingNeed: this.subtext.getPendingNeed(),
    };
  }

  buildPromptBlock(ctx = {}) {
    const lines = [];
    if (ctx.resonanceLine) lines.push(ctx.resonanceLine);
    else if (this._lastRecognized) {
      const line = this.resonance.toPromptLine(this._lastRecognized, ctx.closeness || 0);
      if (line) lines.push(line);
    }
    const mm = this.mentalModel.toPromptBlock();
    if (mm) lines.push(mm);
    if (ctx.subtextLine) lines.push(ctx.subtextLine);
    else if (this._lastSubtext) {
      const st = this.subtext.toPromptLine(this._lastSubtext);
      if (st) lines.push(st);
    }
    return lines.filter(Boolean).join('\n');
  }

  getPublicState() {
    return {
      module: 'understanding',
      version: this.moduleVersion,
      resonance: this.resonance.snapshot(),
      mentalModel: this.mentalModel.snapshot(),
      subtext: this.subtext.snapshot(),
      lastRecognized: this._lastRecognized,
    };
  }
}

module.exports = {
  UnderstandingSubsystem,
  EmotionalResonance,
  MentalModel,
  SubtextDetector,
};
