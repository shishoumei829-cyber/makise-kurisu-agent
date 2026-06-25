'use strict';

const fs = require('fs');
const path = require('path');
const { BeliefRevision } = require('./belief_revision');
const { ReflectionLoop } = require('./reflection_loop');

/**
 * 模块五 · 元认知 — 信念修正、反思反馈环
 */
class MetacognitionSubsystem {
  constructor() {
    this.beliefs = new BeliefRevision();
    this.reflection = new ReflectionLoop();
    this._dataDir = '';
    this._saveTimer = null;
    this._latestInsight = '';
    this.moduleVersion = 2;
  }

  init(dataDir) {
    this._dataDir = dataDir;
    this.load();
  }

  _statePath() {
    return path.join(this._dataDir, 'metacognition_subsystem.json');
  }

  load() {
    try {
      const p = this._statePath();
      if (!fs.existsSync(p)) return;
      const data = JSON.parse(fs.readFileSync(p, 'utf8'));
      this.beliefs.load(data.beliefs);
      this.reflection.load(data.reflection);
      if (data._latestInsight) this._latestInsight = data._latestInsight;
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
          beliefs: this.beliefs.snapshot(),
          reflection: this.reflection.snapshot(),
          _latestInsight: this._latestInsight,
          savedAt: Date.now(),
        }, null, 2));
      } catch { /* ignore */ }
    }, 200);
  }

  onConversationTurn(ctx = {}) {
    const { mainEvent, decision, chatTurnCounter, chatMinimal } = ctx;

    if (mainEvent) this.beliefs.updateWorldview(mainEvent);

    if (decision) {
      this.reflection.reflectOnDecision(decision);
    }

    let insight = null;
    if (this.reflection.shouldSurfaceInsight(chatTurnCounter || 0, chatMinimal !== false)) {
      insight = this.reflection.generateInsight(this.beliefs);
      if (insight) {
        this._latestInsight = insight.content;
        this.beliefs.ingestMetacognitionInsight(insight);
      }
    }

    this._save();
    return {
      insight,
      insightInjection: insight ? this.reflection.insightToGoalInjection(insight) : '',
      beliefLines: this.beliefs.toPromptLines(2),
      conflicts: this.beliefs.conflicts,
    };
  }

  buildPromptBlock(ctx = {}) {
    const lines = [];
    const beliefs = this.beliefs.toPromptLines(2);
    if (beliefs.length) lines.push(beliefs.join('\n'));
    const insight = ctx.metacognitionInsight || this._latestInsight;
    if (insight) lines.push(`自省碎片：${String(insight).slice(0, 90)}`);
    return lines.filter(Boolean).join('\n');
  }

  getPublicState() {
    return {
      module: 'metacognition',
      version: this.moduleVersion,
      beliefs: this.beliefs.snapshot(),
      reflection: this.reflection.snapshot(),
      latestInsight: this._latestInsight,
    };
  }
}

module.exports = {
  MetacognitionSubsystem,
  BeliefRevision,
  ReflectionLoop,
};
