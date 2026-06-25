'use strict';

const fs = require('fs');
const path = require('path');
const { AutonomousBehaviorEngine } = require('./drive_engine');
const { CreativityModule } = require('./creativity');
const { MemoryReorganization } = require('./memory_reorganization');
const { DreamEngine } = require('./dream_engine');
const { BeliefRevision } = require('./belief_revision');
const { EmotionalResonance } = require('./emotional_resonance');
const { TimePerception } = require('./time_perception');
const { EnvironmentUnderstanding } = require('./environment');

/**
 * 数字生命编排器：统一调度各子系统并持久化。
 */
class DigitalLifeOrchestrator {
  constructor() {
    this.drive = new AutonomousBehaviorEngine();
    this.creativity = new CreativityModule();
    this.memoryReorg = new MemoryReorganization();
    this.dream = new DreamEngine();
    this.beliefs = new BeliefRevision();
    this.resonance = new EmotionalResonance();
    this.time = new TimePerception();
    this.environment = new EnvironmentUnderstanding();
    this._dataDir = '';
    this._saveTimer = null;
    this._lastConsolidation = 0;
  }

  init(dataDir) {
    this._dataDir = dataDir;
    this.load();
  }

  _statePath() {
    return path.join(this._dataDir, 'digital_life_state.json');
  }

  load() {
    try {
      const p = this._statePath();
      if (!fs.existsSync(p)) return;
      const data = JSON.parse(fs.readFileSync(p, 'utf8'));
      this.drive.load(data.drive);
      this.creativity.load(data.creativity);
      this.memoryReorg.load(data.memoryReorg);
      this.dream.load(data.dream);
      this.beliefs.load(data.beliefs);
      this.resonance.load(data.resonance);
      this.time.load(data.time);
      this.environment.load(data.environment);
      this._lastConsolidation = data._lastConsolidation || 0;
    } catch { /* ignore */ }
  }

  _save() {
    if (!this._dataDir) return;
    if (this._saveTimer) clearTimeout(this._saveTimer);
    this._saveTimer = setTimeout(() => {
      this._saveTimer = null;
      try {
        const payload = {
          drive: this.drive.snapshot(),
          creativity: this.creativity.snapshot(),
          memoryReorg: this.memoryReorg.snapshot(),
          dream: this.dream.snapshot(),
          beliefs: this.beliefs.snapshot(),
          resonance: this.resonance.snapshot(),
          time: this.time.snapshot(),
          environment: this.environment.snapshot(),
          _lastConsolidation: this._lastConsolidation,
          savedAt: Date.now(),
        };
        fs.writeFileSync(this._statePath(), JSON.stringify(payload, null, 2));
      } catch { /* ignore */ }
    }, 250);
  }

  onUserTurn(ctx = {}) {
    const { pad, memory, motivationState, userText, userModel, mainEvent } = ctx;

    this.drive.updateInternalState(pad, memory, motivationState);
    this.drive.generateUrge();

    const recognized = this.resonance.recognizeEmotion(
      userText,
      userModel?.model?.patterns?.emotion_history || [],
    );
    const padDelta = this.resonance.adjustEmotionalState(
      recognized,
      userModel?.model?.relationship?.closeness || 0,
    );

    if (mainEvent) this.beliefs.updateWorldview(mainEvent);

    this.creativity.learnAssociation(
      ...(String(userText || '').match(/[\u4e00-\u9fa5]{2,}/g) || []).slice(0, 2),
    );

    this._save();
    return {
      recognized,
      padDelta,
      driveBoosts: this.drive.behaviorBoosts(),
      resonanceLine: this.resonance.toPromptLine(recognized),
    };
  }

  runIdleCycle(ctx = {}) {
    const { idleMs, pad, memory, memorySystem } = ctx;
    const idleMin = Math.floor((Number(idleMs) || 0) / 60000);
    const result = { consolidated: false, dream: null, insights: [] };

    if (Date.now() - this._lastConsolidation > 15 * 60 * 1000 && memorySystem) {
      const pack = this.memoryReorg.consolidate(memorySystem);
      result.insights = pack.insights;
      result.consolidated = true;
      this._lastConsolidation = Date.now();
    }

    if (this.dream.shouldDream(idleMs)) {
      const insights = result.insights.length
        ? result.insights
        : this.dream.consolidateMemory(this.memoryReorg, memorySystem);
      result.dream = this.dream.generateDream({
        insights,
        pad,
        recentEvents: memory?.events?.slice(-5) || [],
        idleMin,
      });
    }

    this.drive.updateInternalState(pad, memorySystem, ctx.motivationState);
    this.drive.generateUrge();
    this._save();
    return result;
  }

  onVision(visionText) {
    const scene = this.environment.understandScene(visionText);
    this.environment.inferUserState(scene);
    this._save();
    return scene;
  }

  buildPromptContext(ctx = {}) {
    const lines = [];
    const driveLine = this.drive.toPromptLine();
    if (driveLine) lines.push(driveLine);

    const creative = this.creativity.toPromptLine(ctx);
    if (creative) lines.push(creative);

    const resonance = ctx.resonanceLine || '';
    if (resonance) lines.push(resonance);

    const beliefs = this.beliefs.toPromptLines(2);
    if (beliefs.length) lines.push(beliefs.join('\n'));

    const timeLine = this.time.toPromptLine(ctx.userModel);
    if (timeLine) lines.push(`时间感：${timeLine}`);

    const envLine = this.environment.toPromptLine();
    if (envLine) lines.push(envLine);

    const dreamLine = this.dream.latestDreamLine();
    if (dreamLine && (ctx.includeDream !== false)) lines.push(dreamLine);

    if (ctx.metacognitionInsight) {
      lines.push(`自省碎片：${String(ctx.metacognitionInsight).slice(0, 80)}`);
    }

    return lines.filter(Boolean).join('\n');
  }

  getPublicState() {
    return {
      drive: this.drive.snapshot(),
      creativity: this.creativity.snapshot(),
      memoryReorg: this.memoryReorg.snapshot(),
      dream: this.dream.snapshot(),
      beliefs: this.beliefs.snapshot(),
      resonance: this.resonance.snapshot(),
      time: this.time.snapshot(),
      environment: this.environment.snapshot(),
    };
  }
}

module.exports = { DigitalLifeOrchestrator };
