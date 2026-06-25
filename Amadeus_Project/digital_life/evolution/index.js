'use strict';

const fs = require('fs');
const path = require('path');
const { MemoryConsolidation } = require('./memory_consolidation');
const { DreamEngine } = require('./dream_engine');
const { PersonalityTrajectory } = require('./personality_trajectory');
const { RlBridge } = require('./rl_bridge');

/**
 * 模块二 · 自我进化 — 记忆重组、梦境、人格轨迹、RL 桥接
 */
class EvolutionSubsystem {
  constructor() {
    this.memory = new MemoryConsolidation();
    this.dream = new DreamEngine();
    this.personality = new PersonalityTrajectory();
    this.rlBridge = new RlBridge();
    this._dataDir = '';
    this._saveTimer = null;
    this._lastConsolidation = 0;
    this.moduleVersion = 2;
  }

  init(dataDir) {
    this._dataDir = dataDir;
    this.load();
  }

  _statePath() {
    return path.join(this._dataDir, 'evolution_subsystem.json');
  }

  load() {
    try {
      const p = this._statePath();
      if (!fs.existsSync(p)) return;
      const data = JSON.parse(fs.readFileSync(p, 'utf8'));
      this.memory.load(data.memory);
      this.dream.load(data.dream);
      this.personality.load(data.personality);
      this.rlBridge.load(data.rlBridge);
      this._lastConsolidation = data._lastConsolidation || 0;
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
          memory: this.memory.snapshot(),
          dream: this.dream.snapshot(),
          personality: this.personality.snapshot(),
          rlBridge: this.rlBridge.snapshot(),
          _lastConsolidation: this._lastConsolidation,
          savedAt: Date.now(),
        }, null, 2));
      } catch { /* ignore */ }
    }, 200);
  }

  onConversationTurn(ctx = {}) {
    const { mainEvent, externalTraits, rl, pad, relScore, behaviorId, userText, recognized } = ctx;
    if (mainEvent) this.personality.updateFromEvent(mainEvent);
    if (externalTraits) this.personality.ingestExternalTraits(externalTraits);

    let rlOut = null;
    if (rl) {
      rlOut = this.rlBridge.recordTurn(rl, {
        pad,
        relScore,
        behaviorId,
        userText,
        recognizedEmotion: recognized,
        goalAchieved: ctx.goalAchieved,
        relationshipDelta: ctx.relationshipDelta,
      });
    }

    this._save();
    return {
      personalityLine: this.personality.toPromptLine(),
      rlOut,
    };
  }

  runIdleCycle(ctx = {}) {
    const { idleMs, pad, memorySystem } = ctx;
    const idleMin = Math.floor((Number(idleMs) || 0) / 60000);
    const result = { consolidated: false, dream: null, insights: [], carryover: null };

    if (Date.now() - this._lastConsolidation > 15 * 60 * 1000 && memorySystem) {
      const pack = this.memory.consolidate(memorySystem);
      result.insights = pack.insights;
      result.consolidated = true;
      this._lastConsolidation = Date.now();
    }

    if (this.dream.shouldDream(idleMs)) {
      const insights = result.insights.length
        ? result.insights
        : this.dream.consolidateMemory(this.memory, memorySystem);
      result.dream = this.dream.generateDream({
        insights,
        pad,
        recentEvents: memorySystem?.events?.slice(-6) || [],
        idleMin,
        associations: this.memory.topAssociations(5).map((t) => {
          const [a, b] = t.pair.split('↔');
          return { a, b };
        }),
      });
      result.carryover = this.dream.getCarryover();
    }

    this._save();
    return result;
  }

  buildPromptBlock(ctx = {}) {
    const lines = [];
    const mem = this.memory.toPromptBlock();
    if (mem) lines.push(mem);
    const dream = this.dream.toPromptBlock(ctx.includeDream !== false);
    if (dream) lines.push(dream);
    const pers = this.personality.toPromptLine();
    if (pers) lines.push(pers);
    return lines.filter(Boolean).join('\n');
  }

  getPublicState() {
    return {
      module: 'evolution',
      version: this.moduleVersion,
      memory: this.memory.snapshot(),
      dream: this.dream.snapshot(),
      personality: this.personality.snapshot(),
      rlBridge: this.rlBridge.snapshot(),
      _lastConsolidation: this._lastConsolidation,
    };
  }

  /** 兼容旧 API */
  get memoryReorg() {
    return this.memory;
  }
}

module.exports = {
  EvolutionSubsystem,
  MemoryConsolidation,
  DreamEngine,
  PersonalityTrajectory,
  RlBridge,
};
