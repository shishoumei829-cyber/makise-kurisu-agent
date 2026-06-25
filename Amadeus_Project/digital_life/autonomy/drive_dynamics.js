'use strict';

const {
  DRIVE_TYPES,
  ACTION_INTENTS,
  DRIVE_PHYSIOLOGY,
  DRIVE_INHIBITION,
  INTENT_BEHAVIOR_MAP,
  TOPIC_CLUSTERS,
} = require('./constants');

function clamp01(v) {
  return Math.max(0, Math.min(1, v));
}

function nowMs() {
  return Date.now();
}

/**
 * 单条冲动：有生命周期、可被满足或过期
 */
class Urge {
  constructor(spec) {
    this.id = spec.id || `urge_${nowMs()}_${Math.random().toString(36).slice(2, 7)}`;
    this.drive = spec.drive;
    /** @type {keyof typeof ACTION_INTENTS} */
    this.intentKey = spec.intentKey || spec.intent || '';
    this.intent = spec.intentLabel || ACTION_INTENTS[this.intentKey] || spec.intent || '';
    this.intensity = clamp01(spec.intensity ?? 0.5);
    this.target = spec.target || '';
    this.promptHint = spec.promptHint || '';
    this.createdAt = spec.createdAt || nowMs();
    this.expiresAt = spec.expiresAt || (this.createdAt + 25 * 60 * 1000);
    this.satisfied = false;
    this.satisfiedAt = null;
    this.source = spec.source || 'drive';
  }

  isActive(t = nowMs()) {
    return !this.satisfied && t < this.expiresAt;
  }

  effectiveIntensity(t = nowMs()) {
    if (!this.isActive(t)) return 0;
    const age = t - this.createdAt;
    const ttl = this.expiresAt - this.createdAt;
    const fade = ttl > 0 ? 1 - (age / ttl) * 0.35 : 1;
    return this.intensity * fade;
  }

  satisfy(reason = '') {
    this.satisfied = true;
    this.satisfiedAt = nowMs();
    this._satisfyReason = reason;
  }
}

/**
 * 内驱力动力学：激活、衰减、饱和、交叉抑制、冲动队列
 */
class DriveDynamics {
  constructor() {
    this.activations = {};
    this.satiation = {};
    this.refractoryUntil = {};
    this._lastTickAt = nowMs();
    this.urgeQueue = [];
    this.urgeHistory = [];
    this.activationHistory = [];
    this.internalState = {
      energy: 1.0,
      mood: 0,
      arousal: 0,
      socialBattery: 0.75,
      focusTopic: '',
    };

    for (const key of Object.keys(DRIVE_TYPES)) {
      const phys = DRIVE_PHYSIOLOGY[key] || { basal: 0.4 };
      this.activations[key] = phys.basal;
      this.satiation[key] = 0;
      this.refractoryUntil[key] = 0;
    }
  }

  tick(dtMs, ctx = {}) {
    const t = nowMs();
    const dt = Math.max(0, dtMs || (t - this._lastTickAt));
    this._lastTickAt = t;

    this._decayTowardBasal(dt);
    this._applyContextStimuli(ctx);
    this._applyCrossInhibition();
    this._updateInternalState(ctx);
    this._pruneUrges();
    this._recordActivationSnapshot();

    return this.getNetActivations();
  }

  _decayTowardBasal(dt) {
    for (const key of Object.keys(DRIVE_TYPES)) {
      const phys = DRIVE_PHYSIOLOGY[key];
      if (!phys) continue;
      const half = phys.halfLife || 60000;
      const lambda = Math.LN2 / half;
      const factor = Math.exp(-lambda * dt);
      const basal = phys.basal;
      this.activations[key] = basal + (this.activations[key] - basal) * factor;
      const satDecay = phys.satiationDecay || 0.05;
      this.satiation[key] = Math.max(0, this.satiation[key] - satDecay * (dt / 60000));
      if (tMs() > (this.refractoryUntil[key] || 0)) {
        /* refractory ended */
      }
    }
  }

  _stimulate(drive, amount, opts = {}) {
    if (tMs() < (this.refractoryUntil[drive] || 0)) return;
    const sat = this.satiation[drive] || 0;
    const damped = amount * (1 - sat * 0.65);
    this.activations[drive] = clamp01((this.activations[drive] || 0) + damped);
    if (opts.satiate) {
      this.satiation[drive] = clamp01(sat + (opts.satiateAmount || 0.15));
      this.refractoryUntil[drive] = tMs() + (opts.refractoryMs || 3 * 60 * 1000);
    }
  }

  _applyContextStimuli(ctx) {
    const pad = ctx.pad || {};
    const P = pad.P || 0;
    const A = pad.A || 0;
    const S = pad.S || 0;
    const D = pad.D || 0;
    const rel = ctx.relScore ?? 0;
    const idleMin = (ctx.idleMs || 0) / 60000;
    const text = String(ctx.userText || '');
    const mot = ctx.motivationState || {};

    if (mot.curiosity != null) {
      this.activations.CURIOSITY = clamp01(this.activations.CURIOSITY * 0.4 + mot.curiosity * 0.6);
    }

    if (/科学|量子|神经|实验|理论|物理|数学/.test(text)) {
      this._stimulate('CURIOSITY', 0.18);
      this._stimulate('MASTERY', 0.22);
      this._stimulate('EXPLORATION', 0.12);
      this.internalState.focusTopic = text.slice(0, 40);
    }
    if (/喜欢|爱你|在乎|想你/.test(text)) {
      this._stimulate('CONNECTION', 0.20);
      this._stimulate('AUTONOMY', 0.10);
      this._stimulate('PROTECTION', 0.08);
    }
    if (/笨蛋|滚|闭嘴|讨厌/.test(text)) {
      this._stimulate('PROTECTION', 0.28);
      this._stimulate('AUTONOMY', 0.15);
      this._stimulate('CONNECTION', -0.12);
    }
    if (/难受|孤独|寂寞|烦|累/.test(text)) {
      this._stimulate('CONNECTION', 0.14);
      this._stimulate('MEANING', 0.10);
    }
    if (/为什么|怎么|什么|吗|？|\?/.test(text)) {
      this._stimulate('CURIOSITY', 0.12);
    }
    if (P > 0.25) this._stimulate('PLAYFULNESS', 0.10);
    if (P < -0.25) this._stimulate('PROTECTION', 0.08);
    if (A > 0.45) {
      this._stimulate('CREATIVITY', 0.14);
      this._stimulate('CREATION', 0.12);
    }
    if (S > 0.45) this._stimulate('CONNECTION', 0.12);
    if (D > 0.5) this._stimulate('AUTONOMY', 0.08);
    if (rel > 0.35) this._stimulate('CONNECTION', 0.08 * rel);
    if (rel < -0.1) this._stimulate('PROTECTION', 0.15);

    if (idleMin > 15) this._stimulate('CONNECTION', 0.06 + Math.min(0.12, idleMin * 0.002));
    if (idleMin > 30) this._stimulate('MEANING', 0.08);
    if (idleMin > 45) this._stimulate('CURIOSITY', 0.05);

    if (ctx.userAnsweredUrge) {
      this._satisfyMatchingUrges(ctx.userAnsweredUrge);
    }
    if (ctx.behaviorId === 'ENGAGE' && ctx.turnEngaged) {
      this._stimulate('MASTERY', 0.05, { satiate: true, satiateAmount: 0.12 });
    }
    if (ctx.behaviorId === 'APPROACH' && ctx.turnEngaged) {
      this._stimulate('CONNECTION', 0.06, { satiate: true, satiateAmount: 0.10 });
    }

    const mem = ctx.memory;
    if (mem?.events?.length) {
      const recent = mem.events.slice(-5).map((e) => e.content).join('');
      if (/科学|实验/.test(recent)) this._stimulate('MASTERY', 0.06);
    }
  }

  _applyCrossInhibition() {
    const net = { ...this.activations };
    for (const [source, targets] of Object.entries(DRIVE_INHIBITION)) {
      if (source === 'STAY_SILENT') continue;
      const srcVal = this.activations[source] || 0;
      if (srcVal < 0.45) continue;
      for (const [target, strength] of Object.entries(targets)) {
        if (!net[target]) continue;
        net[target] = clamp01(net[target] - srcVal * strength * 0.5);
      }
    }
    const autonomy = this.activations.AUTONOMY || 0;
    const connection = this.activations.CONNECTION || 0;
    if (autonomy > 0.6 && connection > 0.5) {
      const tension = Math.min(autonomy, connection) * 0.25;
      net.CONNECTION = clamp01(net.CONNECTION - tension);
      net.AUTONOMY = clamp01(net.AUTONOMY - tension * 0.5);
    }
    this.activations = net;
  }

  _updateInternalState(ctx) {
    const pad = ctx.pad || {};
    this.internalState.mood = pad.P || 0;
    this.internalState.arousal = pad.A || 0;
    const spoke = ctx.sheSpoke === true;
    const userSpoke = Boolean(ctx.userText && ctx.userText.length > 2);
    if (userSpoke) {
      this.internalState.socialBattery = clamp01(this.internalState.socialBattery - 0.03);
      this.internalState.energy = clamp01(this.internalState.energy - 0.02);
    }
    if (spoke) {
      this.internalState.socialBattery = clamp01(this.internalState.socialBattery - 0.04);
    }
    const idleMin = (ctx.idleMs || 0) / 60000;
    if (idleMin > 5) {
      this.internalState.socialBattery = clamp01(this.internalState.socialBattery + 0.02 * Math.min(idleMin / 10, 1));
      this.internalState.energy = clamp01(this.internalState.energy + 0.015 * Math.min(idleMin / 15, 1));
    }
    this.internalState.energy = clamp01(0.35 + (pad.A || 0) * 0.25 + this.internalState.socialBattery * 0.4);
  }

  generateUrges(ctx = {}) {
    const ranked = this.getRankedDrives(5);
    const newUrges = [];
    const t = nowMs();

    for (const { drive, activation } of ranked) {
      if (activation < 0.36) continue;
      const intentKey = this._driveToIntent(drive, ctx);
      if (!intentKey) continue;
      if (this.urgeQueue.some((u) => u.drive === drive && u.isActive(t))) continue;

      const urge = new Urge({
        drive,
        intentKey,
        intensity: activation,
        target: this._intentTarget(drive, intentKey, ctx),
        promptHint: this._promptHint(drive, intentKey, ctx),
        expiresAt: t + this._urgeTtl(drive, activation),
        source: 'drive_dynamics',
      });
      newUrges.push(urge);
      this.urgeQueue.push(urge);
    }

    this.urgeQueue.sort((a, b) => b.effectiveIntensity() - a.effectiveIntensity());
    if (this.urgeQueue.length > 8) {
      const dropped = this.urgeQueue.splice(8);
      for (const u of dropped) {
        if (!u.satisfied) this.urgeHistory.push({ ...u, dropped: true });
      }
    }
    return newUrges;
  }

  _driveToIntent(drive, ctx) {
    const map = {
      CURIOSITY: () => (ctx.openQuestions?.length ? 'ASK_QUESTION' : 'EXPLORE_TOPIC'),
      CREATIVITY: () => 'CREATE_IDEA',
      EXPLORATION: () => 'EXPLORE_TOPIC',
      MASTERY: () => 'EXPLORE_TOPIC',
      CONNECTION: () => (ctx.idleMs > 20 * 60 * 1000 ? 'REACH_OUT' : 'DEEPEN_BOND'),
      AUTONOMY: () => (ctx.userWasPushy ? 'DEFEND_SELF' : 'HOLD_BACK'),
      MEANING: () => 'SELF_EXPRESSION',
      PLAYFULNESS: () => 'PLAYFUL_JAB',
      PROTECTION: () => 'DEFEND_SELF',
      CREATION: () => 'CREATE_IDEA',
    };
    const fn = map[drive];
    return fn ? fn() : null;
  }

  _intentTarget(drive, intentKey, ctx) {
    if (ctx.focusTopic) return ctx.focusTopic;
    if (ctx.openQuestions?.[0]) return ctx.openQuestions[0].topic;
    if (intentKey === 'REACH_OUT') return 'partner';
    if (drive === 'MASTERY' || drive === 'CURIOSITY') return this.internalState.focusTopic || '未知话题';
    return '';
  }

  _promptHint(drive, intentKey, ctx) {
    const label = DRIVE_TYPES[drive] || drive;
    const intent = ACTION_INTENTS[intentKey] || intentKey;
    const target = this._intentTarget(drive, intentKey, ctx);
    const hints = {
      ASK_QUESTION: `内驱(${label})：真的想知道「${target}」——自然带进对话，别像审讯`,
      REACH_OUT: `内驱(${label})：有一阵没聊了，可能想轻轻敲他一下，但不要查岗`,
      EXPLORE_TOPIC: `内驱(${label})：想顺着${target || '当前话题'}往深处聊，别硬转`,
      SELF_EXPRESSION: `内驱(${label})：想说点对自己有意义的话，但用她的方式`,
      HOLD_BACK: `内驱(${label})：先憋着也行，不必每句都回应得很满`,
      DEEPEN_BOND: `内驱(${label})：想拉近一点，接他的话再多一句实在的`,
      DEFEND_SELF: `内驱(${label})：边界感上来了，语气可以硬但别关死话题`,
      PLAYFUL_JAB: `内驱(${label})：想调侃一句，别恶意`,
      CREATE_IDEA: `内驱(${label})：脑子里有个模糊联想，可以试探性抛出`,
      STAY_SILENT: `内驱：现在更想安静待着`,
    };
    return hints[intentKey] || `内驱(${label})活跃：${intent}`;
  }

  _urgeTtl(drive, activation) {
    const base = {
      CONNECTION: 35 * 60 * 1000,
      CURIOSITY: 40 * 60 * 1000,
      PROTECTION: 15 * 60 * 1000,
      PLAYFULNESS: 12 * 60 * 1000,
    };
    return (base[drive] || 25 * 60 * 1000) * (0.7 + activation * 0.5);
  }

  _satisfyMatchingUrges(topicOrIntent) {
    const key = String(topicOrIntent || '').toLowerCase();
    for (const u of this.urgeQueue) {
      if (!u.isActive()) continue;
      if (u.intentKey === topicOrIntent || u.intent === topicOrIntent || (u.target && key.includes(String(u.target).toLowerCase()))) {
        u.satisfy('context_match');
        this._stimulate(u.drive, 0, { satiate: true, satiateAmount: 0.2 });
      }
    }
  }

  satisfyUrge(urgeId, reason) {
    const u = this.urgeQueue.find((x) => x.id === urgeId);
    if (u) {
      u.satisfy(reason);
      this._stimulate(u.drive, 0, { satiate: true, satiateAmount: 0.25, refractoryMs: 5 * 60 * 1000 });
      this.urgeHistory.push({ id: u.id, drive: u.drive, intent: u.intent, satisfiedAt: nowMs(), reason });
      if (this.urgeHistory.length > 100) this.urgeHistory.shift();
    }
  }

  _pruneUrges() {
    const t = nowMs();
    const active = [];
    for (const u of this.urgeQueue) {
      if (u.isActive(t)) active.push(u);
      else if (!u.satisfied) this.urgeHistory.push({ id: u.id, expired: true, at: t });
    }
    this.urgeQueue = active;
  }

  _recordActivationSnapshot() {
    this.activationHistory.push({
      at: nowMs(),
      activations: { ...this.activations },
      energy: this.internalState.energy,
    });
    if (this.activationHistory.length > 120) this.activationHistory.shift();
  }

  getRankedDrives(n = 10) {
    return Object.entries(this.activations)
      .map(([drive, activation]) => ({ drive, activation: clamp01(activation) }))
      .sort((a, b) => b.activation - a.activation)
      .slice(0, n);
  }

  getNetActivations() {
    return { ...this.activations };
  }

  getDominantDrive() {
    return this.getRankedDrives(1)[0] || null;
  }

  getActiveUrges() {
    return this.urgeQueue.filter((u) => u.isActive()).slice(0, 5);
  }

  behaviorBoostsFromUrges() {
    const boosts = {};
    for (const u of this.getActiveUrges()) {
      const map = INTENT_BEHAVIOR_MAP[u.intentKey] || {};
      const scale = u.effectiveIntensity();
      for (const [bid, val] of Object.entries(map)) {
        boosts[bid] = (boosts[bid] || 0) + val * scale;
      }
    }
    return boosts;
  }

  toPromptBlock() {
    const lines = [];
    const dom = this.getDominantDrive();
    if (dom) {
      lines.push(`【内驱力】主导：${DRIVE_TYPES[dom.drive]} ${dom.activation.toFixed(2)}；能量 ${this.internalState.energy.toFixed(2)} / 社交余量 ${this.internalState.socialBattery.toFixed(2)}`);
    }
    const urges = this.getActiveUrges().slice(0, 2);
    for (const u of urges) {
      lines.push(u.promptHint);
    }
    const tension = [];
    if ((this.activations.CONNECTION || 0) > 0.55 && (this.activations.AUTONOMY || 0) > 0.55) {
      tension.push('想靠近又想保持独立');
    }
    if ((this.activations.CURIOSITY || 0) > 0.6 && (this.activations.PROTECTION || 0) > 0.5) {
      tension.push('好奇但也在戒备');
    }
    if (tension.length) lines.push(`内在张力：${tension.join('；')}`);
    return lines.join('\n');
  }

  load(data) {
    if (!data) return;
    if (data.activations) this.activations = { ...this.activations, ...data.activations };
    if (data.satiation) this.satiation = { ...this.satiation, ...data.satiation };
    if (data.refractoryUntil) this.refractoryUntil = { ...this.refractoryUntil, ...data.refractoryUntil };
    if (data.internalState) this.internalState = { ...this.internalState, ...data.internalState };
    if (data._lastTickAt) this._lastTickAt = data._lastTickAt;
    if (Array.isArray(data.urgeHistory)) this.urgeHistory = data.urgeHistory.slice(-100);
    if (Array.isArray(data.activationHistory)) this.activationHistory = data.activationHistory.slice(-120);
    if (Array.isArray(data.urgeQueue)) {
      this.urgeQueue = data.urgeQueue.map((u) => Object.assign(new Urge({}), u));
    }
  }

  snapshot() {
    return {
      activations: { ...this.activations },
      satiation: { ...this.satiation },
      refractoryUntil: { ...this.refractoryUntil },
      internalState: { ...this.internalState },
      urgeQueue: this.getActiveUrges().map((u) => ({
        id: u.id,
        drive: u.drive,
        intentKey: u.intentKey,
        intent: u.intent,
        intensity: u.intensity,
        target: u.target,
        promptHint: u.promptHint,
        createdAt: u.createdAt,
        expiresAt: u.expiresAt,
        satisfied: u.satisfied,
        source: u.source,
      })),
      urgeHistory: this.urgeHistory.slice(-15),
      activationHistory: this.activationHistory.slice(-20),
      ranked: this.getRankedDrives(5),
      dominant: this.getDominantDrive(),
      _lastTickAt: this._lastTickAt,
    };
  }
}

function tMs() {
  return Date.now();
}

module.exports = { DriveDynamics, Urge, clamp01 };
