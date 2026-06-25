'use strict';

const EMOTION_LEX = {
  positive: ['开心', '高兴', '快乐', '喜欢', '感谢', '哈哈'],
  negative: ['难过', '伤心', '生气', '烦', '讨厌', '累', '孤独', '寂寞'],
  anxious: ['害怕', '焦虑', '担心', '不安', '紧张'],
  intimate: ['想你', '喜欢你', '爱你', '在乎'],
};

/**
 * 情感共鸣：识别用户情绪并调整自身 PAD。
 */
class EmotionalResonance {
  constructor() {
    this.empathyLevel = 0.5;
    this.emotionalMirror = [];
  }

  recognizeEmotion(text, emotionHistory = []) {
    const t = String(text || '');
    let emotion = 'neutral';
    let intensity = 0.3;

    for (const [label, words] of Object.entries(EMOTION_LEX)) {
      for (const w of words) {
        if (t.includes(w)) {
          emotion = label;
          intensity = Math.min(1, intensity + 0.25);
        }
      }
    }

    if (emotionHistory.length) {
      const last = emotionHistory[emotionHistory.length - 1];
      if (last && last.emotion === emotion) intensity = Math.min(1, intensity + 0.1);
    }

    return { emotion, intensity };
  }

  generateEmpathy(recognized) {
    const { emotion, intensity } = recognized || {};
    const lines = {
      positive: '他心情不错，我也会稍微放松一点',
      negative: '他不太好，我不想用套话打发',
      anxious: '他在焦虑，先别急着讲道理',
      intimate: '他说得很直，我会本能想躲一下但也在听',
      neutral: '',
    };
    const line = lines[emotion] || '';
    return line && intensity > 0.35 ? line : '';
  }

  adjustEmotionalState(recognized, closeness = 0) {
    const { emotion, intensity } = recognized || { emotion: 'neutral', intensity: 0 };
    const scale = 0.08 * intensity * (0.5 + closeness * 0.5);
    const padDelta = { P: 0, A: 0, D: 0 };

    if (emotion === 'positive') padDelta.P = scale;
    if (emotion === 'negative' || emotion === 'anxious') {
      padDelta.P = -scale * 0.6;
      padDelta.A = scale * 0.4;
    }
    if (emotion === 'intimate') {
      padDelta.P = scale * 0.5;
      padDelta.A = scale * 0.7;
      padDelta.D = -scale * 0.3;
    }

    if (scale > 0.02) {
      this.emotionalMirror.push({ emotion, intensity, at: Date.now() });
      if (this.emotionalMirror.length > 40) this.emotionalMirror.shift();
      this.empathyLevel = Math.max(0.2, Math.min(0.95, this.empathyLevel + scale * 0.2));
    }

    return padDelta;
  }

  toPromptLine(recognized) {
    const empathy = this.generateEmpathy(recognized);
    if (!empathy) return '';
    return `共情：${empathy}（共鸣度 ${this.empathyLevel.toFixed(2)}）`;
  }

  load(data) {
    if (!data) return;
    if (typeof data.empathyLevel === 'number') this.empathyLevel = data.empathyLevel;
    if (data.emotionalMirror) this.emotionalMirror = data.emotionalMirror;
  }

  snapshot() {
    return {
      empathyLevel: this.empathyLevel,
      emotionalMirror: this.emotionalMirror.slice(-5),
    };
  }
}

module.exports = { EmotionalResonance };
