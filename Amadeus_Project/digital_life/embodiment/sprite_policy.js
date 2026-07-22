'use strict';

const SPRITE_REGISTRY = {
  neutral: {
    id: 'neutral',
    path: 'assets/Live2d/kurisu/kurisu_stand.png',
    approved: true,
    outfit: 'casual_red_tie',
    tags: ['neutral', 'standing', 'safe-default'],
  },
  thinking: {
    id: 'thinking',
    path: '',
    approved: false,
    outfit: 'casual_red_tie',
    tags: ['thinking', 'focused'],
  },
  skeptical: {
    id: 'skeptical',
    path: '',
    approved: false,
    outfit: 'casual_red_tie',
    tags: ['skeptical', 'debate'],
  },
  annoyed: {
    id: 'annoyed',
    path: '',
    approved: false,
    outfit: 'casual_red_tie',
    tags: ['annoyed', 'boundary'],
  },
  shy_denial: {
    id: 'shy_denial',
    path: '',
    approved: false,
    outfit: 'casual_red_tie',
    tags: ['shy', 'denial', 'intimacy'],
  },
  soft: {
    id: 'soft',
    path: '',
    approved: false,
    outfit: 'casual_red_tie',
    tags: ['warm', 'soft'],
  },
  tired: {
    id: 'tired',
    path: '',
    approved: false,
    outfit: 'casual_red_tie',
    tags: ['low-energy'],
  },
};

const DEFAULT_MIN_HOLD_MS = 45 * 1000;

function normText(s) {
  return String(s || '').replace(/\s+/g, '');
}

function hasAny(text, patterns) {
  return patterns.some((re) => re.test(text));
}

class SpritePolicy {
  constructor(opts = {}) {
    this.currentSpriteId = 'neutral';
    this.lastChangedAt = 0;
    this.lastTrigger = 'init';
    this.minHoldMs = Number.isFinite(opts.minHoldMs) ? opts.minHoldMs : DEFAULT_MIN_HOLD_MS;
  }

  decide(ctx = {}) {
    const now = Number.isFinite(ctx.now) ? ctx.now : Date.now();
    const candidate = this._candidateFromContext(ctx);
    const current = SPRITE_REGISTRY[this.currentSpriteId] || SPRITE_REGISTRY.neutral;
    const target = SPRITE_REGISTRY[candidate.spriteId] || SPRITE_REGISTRY.neutral;
    const elapsed = this.lastChangedAt ? now - this.lastChangedAt : Infinity;

    if (candidate.spriteId === this.currentSpriteId) {
      return this._result(current, {
        hold: true,
        trigger: candidate.trigger,
        changeReason: 'same_state',
      });
    }

    if (elapsed < this.minHoldMs) {
      return this._result(current, {
        hold: true,
        trigger: candidate.trigger,
        changeReason: 'cooldown',
        blockedSpriteId: candidate.spriteId,
      });
    }

    if (!target.approved || !target.path) {
      return this._result(current, {
        hold: true,
        trigger: candidate.trigger,
        changeReason: 'asset_not_approved',
        blockedSpriteId: candidate.spriteId,
      });
    }

    this.currentSpriteId = target.id;
    this.lastChangedAt = now;
    this.lastTrigger = candidate.trigger;
    return this._result(target, {
      hold: false,
      trigger: candidate.trigger,
      changeReason: candidate.reason,
    });
  }

  _candidateFromContext(ctx = {}) {
    const text = normText(ctx.userText);
    const preset = ctx.preset || 'neutral';
    const pad = ctx.pad || {};
    const P = Number(pad.P) || 0;
    const A = Number(pad.A) || 0;
    const D = Number(pad.D) || 0;
    const mainType = ctx.mainEvent?.type || '';
    const pendingNeed = String(ctx.pendingNeed || '');

    if (hasAny(text, [/喜欢|想你|可爱|害羞|脸红|亲|抱|老婆|女朋友|调戏|撩/]) || preset === 'shy') {
      return { spriteId: 'shy_denial', trigger: 'intimacy_tease', reason: 'user_intimacy_or_tease' };
    }

    if (hasAny(text, [/不对|证据|逻辑|为什么|怎么证明|反驳|漏洞|bug|问题在哪/]) || (A > 0.45 && D > 0.15)) {
      return { spriteId: 'skeptical', trigger: 'technical_challenge', reason: 'debate_or_analysis' };
    }

    if (hasAny(text, [/滚|闭嘴|烦|讨厌|废物|没用|越界/]) || mainType === 'negative' || P < -0.45) {
      return { spriteId: 'annoyed', trigger: 'boundary_or_negative', reason: 'boundary_or_negative_event' };
    }

    if (hasAny(text, [/累|困|睡|熬夜|撑不住|难受/]) || pendingNeed.includes('rest') || (P < -0.2 && A < -0.1)) {
      return { spriteId: 'tired', trigger: 'low_energy', reason: 'fatigue_or_late_care' };
    }

    if (preset === 'warm' || (P > 0.25 && A < 0.35 && D < 0.1)) {
      return { spriteId: 'soft', trigger: 'warmth', reason: 'warm_low_intensity_state' };
    }

    if (preset === 'alert' || preset === 'excited' || A > 0.55) {
      return { spriteId: 'thinking', trigger: 'high_arousal_focus', reason: 'focused_or_high_arousal' };
    }

    return { spriteId: 'neutral', trigger: 'neutral', reason: 'neutral_default' };
  }

  _result(sprite, meta) {
    return {
      spriteId: sprite.id,
      spritePath: sprite.path,
      assetApproved: sprite.approved === true,
      outfit: sprite.outfit,
      tags: sprite.tags || [],
      hold: meta.hold === true,
      trigger: meta.trigger || '',
      changeReason: meta.changeReason || '',
      blockedSpriteId: meta.blockedSpriteId || '',
    };
  }

  load(data) {
    if (!data) return;
    if (SPRITE_REGISTRY[data.currentSpriteId]) this.currentSpriteId = data.currentSpriteId;
    if (typeof data.lastChangedAt === 'number') this.lastChangedAt = data.lastChangedAt;
    if (data.lastTrigger) this.lastTrigger = data.lastTrigger;
  }

  snapshot() {
    return {
      currentSpriteId: this.currentSpriteId,
      lastChangedAt: this.lastChangedAt,
      lastTrigger: this.lastTrigger,
      registryVersion: 1,
    };
  }
}

module.exports = { SpritePolicy, SPRITE_REGISTRY };
