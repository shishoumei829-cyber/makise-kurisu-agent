'use strict';

const CORRECTION_RE = /不可能|你怎么能|你又乱说|办不到|够不着|说错|别瞎说|胡说|骗人|乱承诺/;

/**
 * Learner — Monitor / 用户纠正 → SelfModel 张力。
 */
class BrainLearner {
  constructor(brainSelfModel) {
    this.brainSelfModel = brainSelfModel;
    this._lastEvents = [];
  }

  detectCorrectionEvent(userText, previousDraft = '') {
    const t = String(userText || '');
    if (!CORRECTION_RE.test(t)) return null;
    const lower = t.toLowerCase();
    if (/拿|咖啡|接|送|买|过来|跑腿|物理/.test(t + previousDraft)) {
      return { type: 'correction.physical', tensionKey: 'physical_promise', weight: 2 };
    }
    if (/编造|没说过|记得|实录|谁是谁/.test(t)) {
      return { type: 'correction.epistemic', tensionKey: 'epistemic_fabrication', weight: 2 };
    }
    return { type: 'correction.generic', tensionKey: 'identity_ooc', weight: 1 };
  }

  observe(ctx = {}) {
    const { monitorResult, userText, draft, brainSelfModel } = ctx;
    const sm = brainSelfModel || this.brainSelfModel;
    if (!sm) return;

    const events = [];

    if (monitorResult && !monitorResult.pass) {
      for (const v of monitorResult.violations || []) {
        if (v.id === 'effector.physical') {
          sm.recordTension('physical_promise', 1);
          events.push('tension:physical_promise');
        }
        if (v.id.startsWith('epistemic') || v.id.startsWith('dialogue')) {
          sm.recordTension('dialogue_inconsistency', 1);
          events.push('tension:dialogue');
        }
        if (v.id.includes('identity') || v.id.includes('ooc')) {
          sm.recordTension('identity_ooc', 1);
          events.push('tension:identity_ooc');
        }
      }
    }

    const correction = this.detectCorrectionEvent(userText, draft);
    if (correction) {
      sm.recordTension(correction.tensionKey, correction.weight);
      events.push(correction.type);
    }

    if (events.length) {
      this._lastEvents = events.slice(-20);
      console.log(`[brain/learner] ${events.join('; ')}`);
    }
  }

  getLastEvents() {
    return [...this._lastEvents];
  }
}

module.exports = { BrainLearner };
