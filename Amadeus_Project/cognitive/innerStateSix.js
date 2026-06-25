'use strict';

const fs = require('fs');
const path = require('path');

function clamp(v, lo = 0, hi = 1) {
  return Math.max(lo, Math.min(hi, v));
}

const DIMS = ['security', 'energy', 'connection', 'certainty', 'weight', 'boundary'];

/**
 * 六维内部状态：跨会话持久化，影响表达方式。
 */
class InnerStateSix {
  constructor(dataDir) {
    this._path = path.join(dataDir, 'inner_state_six.json');
    this.state = {
      security: 0.52,
      energy: 0.58,
      connection: 0.38,
      certainty: 0.55,
      weight: 0.42,
      boundary: 0.68,
    };
    this.history = [];
    this._load();
  }

  _load() {
    try {
      if (!fs.existsSync(this._path)) return;
      const data = JSON.parse(fs.readFileSync(this._path, 'utf8'));
      if (data.state) this.state = { ...this.state, ...data.state };
      if (data.history) this.history = data.history;
    } catch { /* ignore */ }
  }

  _save() {
    try {
      fs.writeFileSync(this._path, JSON.stringify({
        state: this.state,
        history: this.history.slice(-40),
        savedAt: Date.now(),
      }, null, 2));
    } catch { /* ignore */ }
  }

  updateFromTurn(ctx = {}) {
    const { pad, relScore = 0, userEmotion, mainEvent, userText, behaviorId } = ctx;
    const prev = { ...this.state };
    const P = pad?.P || 0;
    const A = pad?.A || 0;
    const S = pad?.S ?? 0.5;

    if (userEmotion === 'positive' || userEmotion === 'intimate') {
      this.state.connection = clamp(this.state.connection + 0.04);
      this.state.security = clamp(this.state.security + 0.02);
      this.state.boundary = clamp(this.state.boundary - 0.02);
    }
    if (userEmotion === 'negative' || userEmotion === 'anxious') {
      this.state.weight = clamp(this.state.weight + 0.05);
      this.state.security = clamp(this.state.security - 0.03);
      this.state.energy = clamp(this.state.energy - 0.02);
    }
    if (userEmotion === 'aggressive') {
      this.state.boundary = clamp(this.state.boundary + 0.06);
      this.state.connection = clamp(this.state.connection - 0.03);
    }

    if (mainEvent?.type === 'intimate' || mainEvent?.type === 'positive') {
      this.state.connection = clamp(this.state.connection + 0.03);
    }
    if (mainEvent?.type === 'conflict' || mainEvent?.type === 'negative') {
      this.state.boundary = clamp(this.state.boundary + 0.04);
      this.state.certainty = clamp(this.state.certainty - 0.03);
    }
    if (mainEvent?.type === 'scientific') {
      this.state.certainty = clamp(this.state.certainty + 0.03);
      this.state.energy = clamp(this.state.energy + 0.02);
    }

    if (P > 0.25) this.state.energy = clamp(this.state.energy + 0.015);
    if (P < -0.25) {
      this.state.weight = clamp(this.state.weight + 0.02);
      this.state.energy = clamp(this.state.energy - 0.02);
    }
    if (A > 0.5) this.state.energy = clamp(this.state.energy + 0.02);
    if (S > 0.55) this.state.connection = clamp(this.state.connection + 0.02);

    this.state.connection = clamp(this.state.connection + relScore * 0.02 - 0.01);
    this.state.security = clamp(this.state.security + relScore * 0.015);

    if (/你怎么不回|人呢|已读不回/.test(String(userText || ''))) {
      this.state.weight = clamp(this.state.weight + 0.04);
      this.state.connection = clamp(this.state.connection + 0.02);
    }
    if (behaviorId === 'WITHDRAW' || behaviorId === 'DEFLECT') {
      this.state.boundary = clamp(this.state.boundary + 0.02);
    }
    if (behaviorId === 'APPROACH' || behaviorId === 'ENGAGE') {
      this.state.connection = clamp(this.state.connection + 0.015);
      this.state.boundary = clamp(this.state.boundary - 0.01);
    }

    const delta = {};
    for (const k of DIMS) {
      delta[k] = this.state[k] - prev[k];
    }
    this.history.push({ at: Date.now(), delta, behaviorId });
    if (this.history.length > 80) this.history = this.history.slice(-80);
    this._save();
    return { state: { ...this.state }, delta };
  }

  toPromptBlock() {
    const s = this.state;
    const lines = [
      `六维内在（勿照念，只影响语气节奏）：`,
      `安全感${s.security.toFixed(2)} 精力${s.energy.toFixed(2)} 连接${s.connection.toFixed(2)}`,
      `确定感${s.certainty.toFixed(2)} 心事感${s.weight.toFixed(2)} 边界${s.boundary.toFixed(2)}`,
    ];
    const hints = [];
    if (s.security < 0.4) hints.push('安全感偏低，措辞更谨慎');
    if (s.energy < 0.35) hints.push('精力不足，句子更短');
    if (s.connection > 0.6) hints.push('连接感高，可多一点追问');
    if (s.certainty < 0.4) hints.push('不确定时少断言、多求证');
    if (s.weight > 0.6) hints.push('心里有事，别用套话打发');
    if (s.boundary > 0.72) hints.push('边界感强，亲近也要留余地');
    if (hints.length) lines.push(`倾向：${hints.join('；')}`);
    return lines.join('\n');
  }

  snapshot() {
    return { state: { ...this.state }, history: this.history.slice(-8) };
  }
}

module.exports = { InnerStateSix, DIMS };
