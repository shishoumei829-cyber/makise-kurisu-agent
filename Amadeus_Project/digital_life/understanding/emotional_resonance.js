'use strict';

const { EMOTION_LEX } = require('./constants');

function clamp01(v) {
  return Math.max(0, Math.min(1, v));
}

/**
 * 情感共鸣：多层情绪识别、共情调节、PAD 镜像与历史追踪。
 */
class EmotionalResonance {
  constructor() {
    this.empathyLevel = 0.5;
    this.emotionalMirror = [];
    this.regulationMode = 'balanced';
    this._emotionStreak = { emotion: 'neutral', count: 0 };
  }

  recognizeEmotion(text, emotionHistory = []) {
    const t = String(text || '');
    let emotion = 'neutral';
    let intensity = 0.25;
    const hits = [];

    for (const [label, words] of Object.entries(EMOTION_LEX)) {
      for (const w of words) {
        if (t.includes(w)) {
          emotion = label;
          intensity = Math.min(1, intensity + 0.22);
          hits.push(w);
        }
      }
    }

    if (/[?？]/.test(t) && emotion === 'neutral') {
      emotion = 'curious';
      intensity = 0.35;
    }

    if (emotionHistory.length) {
      const last = emotionHistory[emotionHistory.length - 1];
      if (last?.emotion === emotion) {
        intensity = Math.min(1, intensity + 0.12);
        this._emotionStreak = { emotion, count: (this._emotionStreak.count || 0) + 1 };
      } else {
        this._emotionStreak = { emotion, count: 1 };
      }
    }

    if (this._emotionStreak.count >= 3 && (emotion === 'negative' || emotion === 'anxious')) {
      intensity = Math.min(1, intensity + 0.1);
    }

    return { emotion, intensity, hits, streak: this._emotionStreak.count };
  }

  _selectRegulation(recognized, closeness) {
    const { emotion, intensity } = recognized || {};
    if (emotion === 'aggressive') return 'shield';
    if (emotion === 'intimate' && closeness < 0.35) return 'gentle_distance';
    if ((emotion === 'negative' || emotion === 'anxious') && intensity > 0.6) return 'validate_first';
    if (emotion === 'positive' && closeness > 0.4) return 'warm_match';
    return 'balanced';
  }

  generateEmpathy(recognized, closeness = 0) {
    const { emotion, intensity } = recognized || {};
    this.regulationMode = this._selectRegulation(recognized, closeness);

    const lines = {
      positive: '他心情不错，我也会稍微放松一点',
      negative: '他不太好，我不想用套话打发',
      anxious: '他在焦虑，先别急着讲道理',
      intimate: closeness > 0.4
        ? '他说得很直，我会听但也会本能想躲一下'
        : '亲近信号来了，但关系还不够稳，别太冲',
      aggressive: '他在刺，我先稳住边界',
      curious: '他在问，说明还愿意继续',
      neutral: '',
    };

    const line = lines[emotion] || '';
    return line && intensity > 0.32 ? line : '';
  }

  adjustEmotionalState(recognized, closeness = 0) {
    const { emotion, intensity } = recognized || { emotion: 'neutral', intensity: 0 };
    this.regulationMode = this._selectRegulation(recognized, closeness);
    let scale = 0.08 * intensity * (0.45 + closeness * 0.55);
    if (this.regulationMode === 'shield') scale *= 0.5;
    if (this.regulationMode === 'validate_first') scale *= 1.15;

    const padDelta = { P: 0, A: 0, D: 0 };

    if (emotion === 'positive') padDelta.P = scale;
    if (emotion === 'negative' || emotion === 'anxious') {
      padDelta.P = -scale * 0.65;
      padDelta.A = scale * 0.45;
    }
    if (emotion === 'intimate') {
      padDelta.P = scale * 0.45;
      padDelta.A = scale * 0.75;
      padDelta.D = -scale * 0.35;
    }
    if (emotion === 'aggressive') {
      padDelta.P = -scale * 0.4;
      padDelta.A = scale * 0.5;
      padDelta.D = scale * 0.3;
    }
    if (emotion === 'curious') {
      padDelta.A = scale * 0.35;
      padDelta.P = scale * 0.15;
    }

    if (scale > 0.02) {
      this.emotionalMirror.push({ emotion, intensity, mode: this.regulationMode, at: Date.now() });
      if (this.emotionalMirror.length > 50) this.emotionalMirror.shift();
      this.empathyLevel = clamp01(this.empathyLevel + scale * 0.25);
    }

    return padDelta;
  }

  toPromptLine(recognized, closeness = 0) {
    const empathy = this.generateEmpathy(recognized, closeness);
    if (!empathy) return '';
    return `共情：${empathy}（共鸣 ${this.empathyLevel.toFixed(2)} · ${this.regulationMode}）`;
  }

  load(data) {
    if (!data) return;
    if (typeof data.empathyLevel === 'number') this.empathyLevel = data.empathyLevel;
    if (data.emotionalMirror) this.emotionalMirror = data.emotionalMirror;
    if (data.regulationMode) this.regulationMode = data.regulationMode;
    if (data._emotionStreak) this._emotionStreak = data._emotionStreak;
  }

  snapshot() {
    return {
      empathyLevel: this.empathyLevel,
      regulationMode: this.regulationMode,
      emotionalMirror: this.emotionalMirror.slice(-6),
      streak: this._emotionStreak,
    };
  }
}

module.exports = { EmotionalResonance };
