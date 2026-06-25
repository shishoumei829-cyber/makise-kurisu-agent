'use strict';

const fs = require('fs');
const path = require('path');
const { AutonomySubsystem } = require('./autonomy');
const { MemoryReorganization } = require('./memory_reorganization');
const { DreamEngine } = require('./dream_engine');
const { BeliefRevision } = require('./belief_revision');
const { EmotionalResonance } = require('./emotional_resonance');
const { TimePerception } = require('./time_perception');
const { EnvironmentUnderstanding } = require('./environment');

/**
 * 数字生命编排器
 * 模块一（自主性）由 AutonomySubsystem 深度实现；其余模块待逐轮打磨
 */
class DigitalLifeOrchestrator {
  constructor() {
    this.autonomy = new AutonomySubsystem();
    this.memoryReorg = new MemoryReorganization();
    this.dream = new DreamEngine();
    this.beliefs = new BeliefRevision();
    this.resonance = new EmotionalResonance();
    this.time = new TimePerception();
    this.environment = new EnvironmentUnderstanding();
    this._dataDir = '';
    this._saveTimer = null;
    this._lastConsolidation = 0;
    this._lastAutonomyBehavior = null;
  }

  init(dataDir) {
    this._dataDir = dataDir;
    this.autonomy.init(dataDir);
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
      if (data.memoryReorg) this.memoryReorg.load(data.memoryReorg);
      if (data.dream) this.dream.load(data.dream);
      if (data.beliefs) this.beliefs.load(data.beliefs);
      if (data.resonance) this.resonance.load(data.resonance);
      if (data.time) this.time.load(data.time);
      if (data.environment) this.environment.load(data.environment);
      this._lastConsolidation = data._lastConsolidation || 0;
      if (data.drive && !fs.existsSync(this.autonomy._statePath())) {
        this.autonomy.loadLegacy(data.drive);
      }
    } catch { /* ignore */ }
  }

  _save() {
    if (!this._dataDir) return;
    if (this._saveTimer) clearTimeout(this._saveTimer);
    this._saveTimer = setTimeout(() => {
      this._saveTimer = null;
      try {
        const payload = {
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
    const { pad, memory, motivationState, userText, userModel, mainEvent, selfModel, relScore, behaviorId, idleMs } = ctx;

    const autonomyOut = this.autonomy.onConversationTurn({
      pad,
      memory,
      motivationState,
      userText,
      selfModel,
      relScore,
      behaviorId,
      idleMs,
    });

    const recognized = this.resonance.recognizeEmotion(
      userText,
      userModel?.model?.patterns?.emotion_history || [],
    );
    const padDelta = this.resonance.adjustEmotionalState(
      recognized,
      userModel?.model?.relationship?.closeness || 0,
    );

    if (mainEvent) this.beliefs.updateWorldview(mainEvent);

    this._save();
    return {
      recognized,
      padDelta,
      driveBoosts: autonomyOut.behaviorBoosts,
      goalSeeds: autonomyOut.goalSeeds,
      autonomyPrompt: autonomyOut.promptBlock,
      resonanceLine: this.resonance.toPromptLine(recognized),
      openQuestions: autonomyOut.openQuestions,
    };
  }

  evaluateAutonomy(ctx = {}) {
    const behavior = this.autonomy.onIdle({
      pad: ctx.pad,
      memory: ctx.memory || ctx.memorySystem,
      memorySystem: ctx.memorySystem,
      motivationState: ctx.motivationState,
      relScore: ctx.relScore,
      idleMs: ctx.idleMs,
      userPresenceActive: ctx.userPresenceActive,
      dnd: ctx.dnd,
      proactiveQuotaOk: ctx.proactiveQuotaOk,
      sheSpokeRecently: ctx.sheSpokeRecently,
    });
    this._lastAutonomyBehavior = behavior;
    this._save();
    return behavior;
  }

  runIdleCycle(ctx = {}) {
    const { idleMs, pad, memory, memorySystem } = ctx;
    const idleMin = Math.floor((Number(idleMs) || 0) / 60000);
    const result = { consolidated: false, dream: null, insights: [], autonomy: null };

    const relScore = memorySystem?.getRelationshipScore?.() ?? 0;
    result.autonomy = this.evaluateAutonomy({
      ...ctx,
      relScore,
      memorySystem,
    });

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

    const autonomyBlock = this.autonomy.buildPromptBlock({
      pad: ctx.pad,
      memory: ctx.memory,
      selfModel: ctx.selfModel,
      relScore: ctx.relScore,
      userText: ctx.userText,
      autonomyHint: ctx.autonomyHint,
    });
    if (autonomyBlock) lines.push(autonomyBlock);

    const resonance = ctx.resonanceLine || '';
    if (resonance) lines.push(resonance);

    const beliefs = this.beliefs.toPromptLines(2);
    if (beliefs.length) lines.push(beliefs.join('\n'));

    const timeLine = this.time.toPromptLine(ctx.userModel);
    if (timeLine) lines.push(`时间感：${timeLine}`);

    const envLine = this.environment.toPromptLine();
    if (envLine) lines.push(envLine);

    const dreamLine = this.dream.latestDreamLine();
    if (dreamLine && ctx.includeDream !== false) lines.push(dreamLine);

    if (ctx.metacognitionInsight) {
      lines.push(`自省碎片：${String(ctx.metacognitionInsight).slice(0, 80)}`);
    }

    return lines.filter(Boolean).join('\n');
  }

  getPublicState() {
    return {
      autonomy: this.autonomy.getPublicState(),
      memoryReorg: this.memoryReorg.snapshot(),
      dream: this.dream.snapshot(),
      beliefs: this.beliefs.snapshot(),
      resonance: this.resonance.snapshot(),
      time: this.time.snapshot(),
      environment: this.environment.snapshot(),
      lastAutonomyBehavior: this._lastAutonomyBehavior,
    };
  }

  /** 兼容旧字段 drive */
  get drive() {
    return this.autonomy.drives;
  }
}

module.exports = { DigitalLifeOrchestrator };
