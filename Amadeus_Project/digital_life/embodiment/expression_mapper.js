'use strict';

const { EXPRESSION_PRESETS } = require('./constants');

function distPad(a, b) {
  const keys = ['P', 'A', 'D'];
  let s = 0;
  for (const k of keys) {
    const d = (a[k] || 0) - (b[k] || 0);
    s += d * d;
  }
  return Math.sqrt(s);
}

/**
 * PAD → 表情预设 / Live2D sprite 索引 / TTS 情绪提示。
 */
class ExpressionMapper {
  constructor() {
    this.currentPreset = 'neutral';
    this.spriteIndex = 0;
    this.ttsHint = 'default';
    this._lastPad = null;
  }

  mapFromPad(pad = {}) {
    const P = Number(pad.P) || 0;
    const A = Number(pad.A) || 0;
    const D = Number(pad.D) || 0;
    const S = Number(pad.S) || 0.5;
    const norm = { P, A, D };

    let best = EXPRESSION_PRESETS[0];
    let bestDist = Infinity;
    for (const preset of EXPRESSION_PRESETS) {
      const d = distPad(norm, preset.pad);
      if (d < bestDist) {
        bestDist = d;
        best = preset;
      }
    }

    this.currentPreset = best.id;
    this.spriteIndex = this._spriteFromPad(P, A, D, S);
    this.ttsHint = this._ttsFromPreset(best.id, pad);
    this._lastPad = { P, A, D, S };

    return {
      preset: best.id,
      label: best.label,
      spriteIndex: this.spriteIndex,
      ttsHint: this.ttsHint,
      filter: this._cssFilter(P, A),
    };
  }

  _spriteFromPad(P, A, D, S) {
    if (P < -0.35 && A < 0) return 0;
    if (P > 0.35 && A > 0.45) return 2;
    if (A > 0.55 && D < 0) return 3;
    if (P < -0.2 && A > 0.35) return 4;
    if (S < 0.35) return 5;
    return 1;
  }

  _ttsFromPreset(presetId, pad) {
    if (presetId === 'shy' || (pad.D < -0.25 && pad.A > 0.2)) return 'shy';
    if (presetId === 'cold' || pad.P < -0.35) return 'cold';
    if (presetId === 'excited' || pad.A > 0.55) return 'excited';
    if (presetId === 'sad') return 'sad';
    return 'default';
  }

  _cssFilter(P, A) {
    const warmth = 1 + P * 0.12;
    const sat = 1 + A * 0.08;
    return `saturate(${sat.toFixed(2)}) brightness(${warmth.toFixed(2)})`;
  }

  load(data) {
    if (!data) return;
    if (data.currentPreset) this.currentPreset = data.currentPreset;
    if (typeof data.spriteIndex === 'number') this.spriteIndex = data.spriteIndex;
    if (data.ttsHint) this.ttsHint = data.ttsHint;
  }

  snapshot() {
    return {
      currentPreset: this.currentPreset,
      spriteIndex: this.spriteIndex,
      ttsHint: this.ttsHint,
      lastPad: this._lastPad,
    };
  }
}

module.exports = { ExpressionMapper, EXPRESSION_PRESETS };
