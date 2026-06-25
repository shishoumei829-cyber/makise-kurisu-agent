'use strict';

const fs = require('fs');
const path = require('path');
const { AutonomySubsystem } = require('./autonomy');
const { EvolutionSubsystem } = require('./evolution');
const { UnderstandingSubsystem } = require('./understanding');
const { EmbodimentSubsystem } = require('./embodiment');
const { MetacognitionSubsystem } = require('./metacognition');

/**
 * 数字生命编排器
 * 五大子系统：自主性 / 自我进化 / 用户理解 / 具身化 / 元认知
 */
class DigitalLifeOrchestrator {
  constructor() {
    this.autonomy = new AutonomySubsystem();
    this.evolution = new EvolutionSubsystem();
    this.understanding = new UnderstandingSubsystem();
    this.embodiment = new EmbodimentSubsystem();
    this.metacognition = new MetacognitionSubsystem();
    this._dataDir = '';
    this._saveTimer = null;
    this._lastAutonomyBehavior = null;
    this._lastIdleCycle = null;
  }

  init(dataDir) {
    this._dataDir = dataDir;
    this.autonomy.init(dataDir);
    this.evolution.init(dataDir);
    this.understanding.init(dataDir);
    this.embodiment.init(dataDir);
    this.metacognition.init(dataDir);
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

      if (data.drive && !fs.existsSync(this.autonomy._statePath())) {
        this.autonomy.loadLegacy(data.drive);
      }
      if (data.memoryReorg && !fs.existsSync(this.evolution._statePath())) {
        this.evolution.memory.load(data.memoryReorg);
      }
      if (data.dream && !fs.existsSync(this.evolution._statePath())) {
        this.evolution.dream.load(data.dream);
      }
      if (data.beliefs && !fs.existsSync(this.metacognition._statePath())) {
        this.metacognition.beliefs.load(data.beliefs);
      }
      if (data.resonance && !fs.existsSync(this.understanding._statePath())) {
        this.understanding.resonance.load(data.resonance);
      }
      if (data.time && !fs.existsSync(this.embodiment._statePath())) {
        this.embodiment.time.load(data.time);
      }
      if (data.environment && !fs.existsSync(this.embodiment._statePath())) {
        this.embodiment.environment.load(data.environment);
      }
      if (data._lastConsolidation) {
        this.evolution._lastConsolidation = data._lastConsolidation;
      }
    } catch { /* ignore */ }
  }

  _saveLegacyIndex() {
    if (!this._dataDir) return;
    if (this._saveTimer) clearTimeout(this._saveTimer);
    this._saveTimer = setTimeout(() => {
      this._saveTimer = null;
      try {
        fs.writeFileSync(this._statePath(), JSON.stringify({
          memoryReorg: this.evolution.memory.snapshot(),
          dream: this.evolution.dream.snapshot(),
          beliefs: this.metacognition.beliefs.snapshot(),
          resonance: this.understanding.resonance.snapshot(),
          time: this.embodiment.time.snapshot(),
          environment: this.embodiment.environment.snapshot(),
          _lastConsolidation: this.evolution._lastConsolidation,
          savedAt: Date.now(),
        }, null, 2));
      } catch { /* ignore */ }
    }, 250);
  }

  onUserTurn(ctx = {}) {
    const {
      pad,
      memory,
      motivationState,
      userText,
      userModel,
      mainEvent,
      selfModel,
      relScore,
      behaviorId,
      idleMs,
      rl,
      decision,
      chatTurnCounter,
      chatMinimal,
    } = ctx;

    const closeness = userModel?.model?.relationship?.closeness ?? relScore ?? 0;

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

    const understandingOut = this.understanding.onConversationTurn({
      userText,
      userModel,
      closeness,
      inferredBdi: ctx.inferredBdi,
    });

    const embodimentOut = this.embodiment.onConversationTurn({
      pad,
      userModel,
      idleMs,
    });

    const evolutionOut = this.evolution.onConversationTurn({
      mainEvent,
      externalTraits: ctx.externalTraits,
      rl,
      pad,
      relScore,
      behaviorId: ctx.previousBehaviorId || behaviorId,
      userText,
      recognized: understandingOut.recognized,
      goalAchieved: ctx.goalAchieved,
      relationshipDelta: ctx.relationshipDelta,
    });

    this._saveLegacyIndex();
    return {
      recognized: understandingOut.recognized,
      padDelta: understandingOut.padDelta,
      driveBoosts: autonomyOut.behaviorBoosts,
      goalSeeds: autonomyOut.goalSeeds,
      autonomyPrompt: autonomyOut.promptBlock,
      resonanceLine: understandingOut.resonanceLine,
      subtextLine: understandingOut.subtextLine,
      mentalModelLine: understandingOut.mentalModelLine,
      pendingNeed: understandingOut.pendingNeed,
      personalityLine: evolutionOut.personalityLine,
      expression: embodimentOut.expression,
      timeLine: embodimentOut.timeLine,
      beliefLines: this.metacognition.beliefs.toPromptLines(2),
      openQuestions: autonomyOut.openQuestions,
    };
  }

  afterBehaviorDecision(ctx = {}) {
    const metacogOut = this.metacognition.onConversationTurn({
      mainEvent: ctx.mainEvent,
      decision: ctx.decision,
      chatTurnCounter: ctx.chatTurnCounter,
      chatMinimal: ctx.chatMinimal,
    });
    this._saveLegacyIndex();
    return metacogOut;
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
      dreamCarryover: this.evolution.dream.getCarryover(),
    });
    this._lastAutonomyBehavior = behavior;
    this._saveLegacyIndex();
    return behavior;
  }

  runIdleCycle(ctx = {}) {
    const { idleMs, pad, memorySystem } = ctx;
    const relScore = memorySystem?.getRelationshipScore?.() ?? ctx.relScore ?? 0;

    const autonomy = this.evaluateAutonomy({
      ...ctx,
      relScore,
      memorySystem,
    });

    const evolution = this.evolution.runIdleCycle({
      idleMs,
      pad,
      memorySystem,
    });

    const result = {
      consolidated: evolution.consolidated,
      dream: evolution.dream,
      insights: evolution.insights,
      carryover: evolution.carryover,
      autonomy,
    };

    this._lastIdleCycle = result;
    this._saveLegacyIndex();
    return result;
  }

  onVision(visionText) {
    return this.embodiment.onVision(visionText);
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

    const understandingBlock = this.understanding.buildPromptBlock({
      resonanceLine: ctx.resonanceLine,
      subtextLine: ctx.subtextLine,
      closeness: ctx.closeness,
    });
    if (understandingBlock) lines.push(understandingBlock);

    if (ctx.mentalModelLine) lines.push(ctx.mentalModelLine);

    const metacogBlock = this.metacognition.buildPromptBlock({
      metacognitionInsight: ctx.metacognitionInsight,
    });
    if (metacogBlock) lines.push(metacogBlock);

    const evolutionBlock = this.evolution.buildPromptBlock({
      includeDream: ctx.includeDream !== false,
    });
    if (evolutionBlock) lines.push(evolutionBlock);

    const embodimentBlock = this.embodiment.buildPromptBlock({
      userModel: ctx.userModel,
      idleMs: ctx.idleMs,
      timeLine: ctx.timeLine,
    });
    if (embodimentBlock) lines.push(embodimentBlock);

    if (ctx.pendingNeed) {
      lines.push(`未满足需求：${ctx.pendingNeed}`);
    }

    const dreamHint = this.evolution.dream.proactiveHint();
    if (dreamHint && ctx.includeDream !== false) lines.push(dreamHint);

    return lines.filter(Boolean).join('\n');
  }

  getPublicState() {
    return {
      autonomy: this.autonomy.getPublicState(),
      evolution: this.evolution.getPublicState(),
      understanding: this.understanding.getPublicState(),
      embodiment: this.embodiment.getPublicState(),
      metacognition: this.metacognition.getPublicState(),
      lastAutonomyBehavior: this._lastAutonomyBehavior,
      lastIdleCycle: this._lastIdleCycle,
      // 兼容旧 UI 字段
      memoryReorg: this.evolution.memory.snapshot(),
      dream: this.evolution.dream.snapshot(),
      beliefs: this.metacognition.beliefs.snapshot(),
      resonance: this.understanding.resonance.snapshot(),
      time: this.embodiment.time.snapshot(),
      environment: this.embodiment.environment.snapshot(),
    };
  }

  /** 兼容旧字段 */
  get drive() {
    return this.autonomy.drives;
  }

  get memoryReorg() {
    return this.evolution.memory;
  }

  get dream() {
    return this.evolution.dream;
  }

  get beliefs() {
    return this.metacognition.beliefs;
  }

  get resonance() {
    return this.understanding.resonance;
  }

  get time() {
    return this.embodiment.time;
  }

  get environment() {
    return this.embodiment.environment;
  }
}

module.exports = { DigitalLifeOrchestrator };
