'use strict';

const { SUBTEXT_PATTERNS } = require('./constants');

/**
 * 言外之意：表面话术 vs 隐含需求检测。
 */
class SubtextDetector {
  constructor() {
    this.recent = [];
    this._pendingNeeds = [];
  }

  analyze(text, recognized = {}) {
    const t = String(text || '');
    const hits = [];

    for (const pat of SUBTEXT_PATTERNS) {
      if (pat.re.test(t)) {
        hits.push({
          label: pat.label,
          implication: pat.implication,
          confidence: 0.55 + (recognized.intensity || 0) * 0.2,
        });
      }
    }

    if (/…|\.{3}/.test(t) && recognized.emotion === 'negative') {
      hits.push({
        label: '未尽之言',
        implication: '话没说完，可能在等对方接话',
        confidence: 0.6,
      });
    }

    if (/好吧|行吧|嗯/.test(t) && t.length < 12) {
      hits.push({
        label: '敷衍信号',
        implication: '可能失望或不想继续深聊',
        confidence: 0.5,
      });
    }

    const result = {
      surface: t.slice(0, 80),
      hits,
      primaryNeed: this._inferNeed(hits, recognized),
      at: Date.now(),
    };

    if (result.primaryNeed) {
      this._pendingNeeds.push(result.primaryNeed);
      if (this._pendingNeeds.length > 10) this._pendingNeeds.shift();
    }

    this.recent.push(result);
    if (this.recent.length > 20) this.recent.shift();
    return result;
  }

  _inferNeed(hits, recognized) {
    if (!hits.length) return '';
    const labels = hits.map((h) => h.label);
    if (labels.includes('求关注')) return '需要回应与在场感';
    if (labels.includes('掩饰')) return '需要被温柔追问而非逼问';
    if (labels.includes('试探')) return '需要确认你在意他';
    if (labels.includes('退缩') || labels.includes('撤回')) return '需要安全邀请继续说';
    if (recognized.emotion === 'aggressive') return '需要边界感但不冷战';
    return hits[0].implication;
  }

  toPromptLine(analysis) {
    if (!analysis?.hits?.length) return '';
    const top = analysis.hits[0];
    const need = analysis.primaryNeed ? `；深层：${analysis.primaryNeed}` : '';
    return `言外之意(${top.label})：${top.implication}${need}`;
  }

  getPendingNeed() {
    return this._pendingNeeds[this._pendingNeeds.length - 1] || '';
  }

  load(data) {
    if (!data) return;
    if (data.recent) this.recent = data.recent;
    if (data._pendingNeeds) this._pendingNeeds = data._pendingNeeds;
  }

  snapshot() {
    return {
      recent: this.recent.slice(-4),
      pendingNeed: this.getPendingNeed(),
    };
  }
}

module.exports = { SubtextDetector };
