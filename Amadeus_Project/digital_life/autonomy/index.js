'use strict';

const fs = require('fs');
const path = require('path');
const { DRIVE_TYPES } = require('./constants');
const { DriveDynamics } = require('./drive_dynamics');
const { CuriosityEngine } = require('./curiosity');
const { CreativityModule } = require('./creativity');
const { AutonomousBehaviorLoop } = require('./behavior_loop');

/**
 * 模块一 · 自主性增强 — 统一入口
 * 内驱力动力学 + 好奇心 + 创造性 + 自主行为环
 */
class AutonomySubsystem {
  constructor() {
    this.drives = new DriveDynamics();
    this.curiosity = new CuriosityEngine();
    this.creativity = new CreativityModule();
    this.behaviorLoop = new AutonomousBehaviorLoop(this.drives, this.curiosity, this.creativity);
    this._dataDir = '';
    this._saveTimer = null;
    this.moduleVersion = 2;
  }

  init(dataDir) {
    this._dataDir = dataDir;
    this.load();
  }

  _statePath() {
    return path.join(this._dataDir, 'autonomy_subsystem.json');
  }

  load() {
    try {
      const p = this._statePath();
      if (!fs.existsSync(p)) return;
      const data = JSON.parse(fs.readFileSync(p, 'utf8'));
      this.drives.load(data.drives);
      this.curiosity.load(data.curiosity);
      this.creativity.load(data.creativity);
      this.behaviorLoop.load(data.behaviorLoop);
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
          drives: this.drives.snapshot(),
          curiosity: this.curiosity.snapshot(),
          creativity: this.creativity.snapshot(),
          behaviorLoop: this.behaviorLoop.snapshot(),
          savedAt: Date.now(),
        }, null, 2));
      } catch { /* ignore */ }
    }, 200);
  }

  /**
   * 用户轮次 / 对话轮次
   */
  onConversationTurn(ctx = {}) {
    const {
      pad,
      memory,
      motivationState,
      userText,
      selfModel,
      relScore,
      behaviorId,
    } = ctx;

    let tokens = [];
    if (userText) {
      this.curiosity.ingestText(userText);
      tokens = (String(userText).match(/[\u4e00-\u9fa5]{2,}/g) || []).slice(0, 3);
      if (tokens.length >= 2) this.creativity.learnAssociation(tokens[0], tokens[1]);
      this.curiosity.markAnswered(null, userText);
    }

    const openQuestions = this.curiosity.getOpenQuestions(5);
    const curious = this.curiosity.generateCuriousQuestions(
      { pad, selfModel, relScore },
      memory,
    );

    this.drives.tick(0, {
      pad,
      memory,
      motivationState,
      userText,
      relScore,
      idleMs: ctx.idleMs || 0,
      userWasPushy: /你怎么不回|人呢|已读不回/.test(String(userText || '')),
      openQuestions: curious,
      behaviorId,
      turnEngaged: Boolean(userText && userText.length > 6),
    });

    this.drives.generateUrges({
      pad,
      relScore,
      idleMs: ctx.idleMs || 0,
      openQuestions: curious,
      focusTopic: tokens?.[0] || '',
      userText,
    });

    this._save();
    return {
      behaviorBoosts: this.drives.behaviorBoostsFromUrges(),
      goalSeeds: this._buildGoalSeeds(curious, selfModel, memory, pad, relScore),
      promptBlock: this.buildPromptBlock(ctx),
      openQuestions,
    };
  }

  _buildGoalSeeds(curious, selfModel, memory, pad, relScore) {
    const seeds = [];
    for (const u of this.drives.getActiveUrges().slice(0, 2)) {
      seeds.push({
        id: `URGE_${u.drive}_${Date.now() % 10000}`,
        label: `${DRIVE_TYPES[u.drive] || u.drive}`,
        priority: u.effectiveIntensity(),
        turns_remaining: 3,
        behavior_hint: u.promptHint,
        prompt_injection: u.promptHint,
      });
    }
    for (const q of curious.slice(0, 1)) {
      if (!seeds.find((s) => s.label.includes('好奇'))) {
        seeds.push({
          id: 'CURIOSITY_GAP',
          label: `好奇：${q.topic}`,
          priority: q.priority,
          turns_remaining: 4,
          prompt_injection: `内部好奇：${q.question}。仅在与本轮焦点相容时顺带带出。`,
        });
      }
    }
    return seeds;
  }

  /**
   * 空闲/主动轮
   */
  onIdle(ctx = {}) {
    const openQuestions = this.curiosity.getOpenQuestions(5);
    this.drives.tick(ctx.idleMs || 60000, {
      pad: ctx.pad,
      memory: ctx.memory || ctx.memorySystem,
      motivationState: ctx.motivationState,
      relScore: ctx.relScore,
      idleMs: ctx.idleMs,
      openQuestions,
    });
    this.drives.generateUrges({
      pad: ctx.pad,
      relScore: ctx.relScore,
      idleMs: ctx.idleMs,
      openQuestions,
    });

    const behavior = this.behaviorLoop.execute({
      isAutonomyTick: true,
      idleMs: ctx.idleMs || 0,
      pad: ctx.pad,
      relScore: ctx.relScore ?? 0,
      userPresenceActive: ctx.userPresenceActive,
      dnd: ctx.dnd,
      proactiveQuotaOk: ctx.proactiveQuotaOk !== false,
      sheSpokeRecently: ctx.sheSpokeRecently,
    });

    this._save();
    return behavior;
  }

  buildPromptBlock(ctx = {}) {
    const lines = [];
    const driveBlock = this.drives.toPromptBlock();
    if (driveBlock) lines.push(driveBlock);

    const curiosityBlock = this.curiosity.toPromptBlock(
      ctx.memory,
      ctx.selfModel,
      { pad: ctx.pad, relScore: ctx.relScore, selfModel: ctx.selfModel },
    );
    if (curiosityBlock) lines.push(curiosityBlock);

    const creative = this.creativity.toPromptBlock({
      pad: ctx.pad,
      userText: ctx.userText,
      driveIntent: this.drives.getActiveUrges()[0]?.intent,
    });
    if (creative) lines.push(creative);

    if (ctx.autonomyHint) lines.push(ctx.autonomyHint);
    return lines.filter(Boolean).join('\n');
  }

  getPublicState() {
    return {
      module: 'autonomy',
      version: this.moduleVersion,
      drives: this.drives.snapshot(),
      curiosity: this.curiosity.snapshot(),
      creativity: this.creativity.snapshot(),
      behaviorLoop: this.behaviorLoop.snapshot(),
    };
  }

  /** 兼容旧 API */
  behaviorBoosts() {
    return this.drives.behaviorBoostsFromUrges();
  }

  toPromptLine() {
    return this.drives.toPromptBlock();
  }

  snapshot() {
    return this.getPublicState();
  }

  loadLegacy(data) {
    if (!data) return;
    this.drives.load(data);
  }
}

module.exports = {
  AutonomySubsystem,
  DriveDynamics,
  CuriosityEngine,
  CreativityModule,
  AutonomousBehaviorLoop,
  DRIVE_TYPES,
};
