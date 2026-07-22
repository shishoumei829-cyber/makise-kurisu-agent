'use strict';

const { TOM_SIGNALS } = require('./constants');

function clamp01(v) {
  return Math.max(0, Math.min(1, v));
}

/**
 * 心智模型：用户信念/欲望/意图推断、偏好图谱、关系假设。
 */
class MentalModel {
  constructor() {
    this.beliefs = [];
    this.desires = [];
    this.intentions = [];
    this.preferences = new Map();
    this.hypotheses = [];
    this.confidence = 0.45;
  }

  _extractMatches(text, re) {
    const t = String(text || '');
    const out = [];
    const parts = t.split(/[。！？!?；;]/);
    for (const p of parts) {
      if (re.test(p)) out.push(p.trim().slice(0, 60));
    }
    return out;
  }

  ingestUserText(text, userModel = null, admission = null) {
    const t = String(text || '');
    if (!t) return null;

    const beliefs = this._extractMatches(t, TOM_SIGNALS.belief);
    const desires = this._extractMatches(t, TOM_SIGNALS.desire);
    const intentions = this._extractMatches(t, TOM_SIGNALS.intention);

    for (const b of beliefs) this._pushUnique(this.beliefs, b, 'belief');
    for (const d of desires) this._pushUnique(this.desires, d, 'desire');
    for (const i of intentions) this._pushUnique(this.intentions, i, 'intention');

    if (!admission || admission.allowProfile === true) {
      for (const token of t.match(/[\u4e00-\u9fa5]{2,8}/g) || []) {
        if (/^(?:我|你|他|她|我们|他们|这个|那个|什么|怎么|为什么|因为|所以|但是|不过)$/.test(token)) continue;
        const prev = this.preferences.get(token) || 0;
        this.preferences.set(token, clamp01(prev + 0.12));
      }
    }

    if (userModel?.model?.preferences) {
      for (const [k, v] of Object.entries(userModel.model.preferences)) {
        if (v > 0.4) {
          const prev = this.preferences.get(k) || 0;
          this.preferences.set(k, clamp01(Math.max(prev, v * 0.8)));
        }
      }
    }

    this._trim();
    this.confidence = clamp01(0.35 + this.beliefs.length * 0.04 + this.desires.length * 0.03);
    return { beliefs, desires, intentions };
  }

  applyInferredBdi(bdi = {}) {
    if (!bdi) return;
    for (const b of bdi.beliefs || []) this._pushUnique(this.beliefs, b, 'bdi_belief');
    for (const d of bdi.desires || []) this._pushUnique(this.desires, d, 'bdi_desire');
    for (const i of bdi.intentions || []) this._pushUnique(this.intentions, i, 'bdi_intention');
    this._trim();
    this.confidence = clamp01(this.confidence + 0.05);
  }

  _pushUnique(arr, text, source) {
    const norm = String(text).trim();
    if (!norm || arr.some((x) => x.text === norm)) return;
    arr.push({ text: norm, source, at: Date.now(), confidence: 0.55 });
    if (arr.length > 30) arr.shift();
  }

  _trim() {
    if (this.beliefs.length > 25) this.beliefs = this.beliefs.slice(-25);
    if (this.desires.length > 20) this.desires = this.desires.slice(-20);
    if (this.intentions.length > 20) this.intentions = this.intentions.slice(-20);
    if (this.preferences.size > 80) {
      const top = [...this.preferences.entries()].sort((a, b) => b[1] - a[1]).slice(0, 60);
      this.preferences = new Map(top);
    }
  }

  topPreferences(n = 5) {
    return [...this.preferences.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, n)
      .map(([topic, score]) => ({ topic, score }));
  }

  buildHypothesis(recognized) {
    const { emotion } = recognized || {};
    let hypothesis = '';
    if (emotion === 'negative' && this.desires.length) {
      hypothesis = `他可能想要：${this.desires[this.desires.length - 1].text}`;
    } else if (emotion === 'anxious' && this.beliefs.length) {
      hypothesis = `他担心：${this.beliefs[this.beliefs.length - 1].text}`;
    } else if (this.intentions.length) {
      hypothesis = `他打算：${this.intentions[this.intentions.length - 1].text}`;
    }
    if (hypothesis) {
      this.hypotheses.push({ text: hypothesis, at: Date.now() });
      if (this.hypotheses.length > 15) this.hypotheses.shift();
    }
    return hypothesis;
  }

  purgeTopics(fragments = []) {
    const list = fragments.map(String).filter(Boolean);
    const hit = (value) => list.some((fragment) => String(value || '').includes(fragment));
    this.beliefs = this.beliefs.filter((item) => !hit(item.text));
    this.desires = this.desires.filter((item) => !hit(item.text));
    this.intentions = this.intentions.filter((item) => !hit(item.text));
    this.hypotheses = this.hypotheses.filter((item) => !hit(item.text));
    for (const key of [...this.preferences.keys()]) {
      if (hit(key)) this.preferences.delete(key);
    }
  }

  toPromptBlock() {
    const lines = [];
    const prefs = this.topPreferences(3).map((p) => p.topic);
    if (prefs.length) lines.push(`用户兴趣线索：${prefs.join('、')}`);
    if (this.desires.length) {
      lines.push(`推测欲望：${this.desires.slice(-1)[0].text}`);
    }
    if (this.beliefs.length) {
      lines.push(`推测信念：${this.beliefs.slice(-1)[0].text}`);
    }
    const hyp = this.hypotheses[this.hypotheses.length - 1];
    if (hyp) lines.push(`心智假设：${hyp.text}`);
    return lines.join('\n');
  }

  load(data) {
    if (!data) return;
    if (data.beliefs) this.beliefs = data.beliefs;
    if (data.desires) this.desires = data.desires;
    if (data.intentions) this.intentions = data.intentions;
    if (data.preferences) this.preferences = new Map(Object.entries(data.preferences));
    if (data.hypotheses) this.hypotheses = data.hypotheses;
    if (typeof data.confidence === 'number') this.confidence = data.confidence;
  }

  snapshot() {
    return {
      beliefs: this.beliefs.slice(-5),
      desires: this.desires.slice(-5),
      intentions: this.intentions.slice(-5),
      topPreferences: this.topPreferences(6),
      hypotheses: this.hypotheses.slice(-3),
      confidence: this.confidence,
    };
  }
}

module.exports = { MentalModel };
