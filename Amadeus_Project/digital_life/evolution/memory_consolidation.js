'use strict';

const { EPISODE_GAP_MS } = require('./constants');

function clamp01(v) {
  return Math.max(0, Math.min(1, v));
}

/**
 * 记忆重组：情节分段、关联图谱、模式提取、重要性衰减与再激活。
 */
class MemoryConsolidation {
  constructor() {
    this.associations = new Map();
    this.patterns = [];
    this.episodes = [];
    this.schemas = new Map();
    this._lastRun = 0;
    this._consolidationCount = 0;
  }

  _tokens(text) {
    return [...new Set(String(text || '').match(/[\u4e00-\u9fa5]{2,}/g) || [])];
  }

  _eventAt(ev) {
    return ev.timestamp || ev.at || ev.createdAt || 0;
  }

  segmentEpisodes(events) {
    const sorted = [...(events || [])].sort((a, b) => this._eventAt(a) - this._eventAt(b));
    const episodes = [];
    let current = null;

    for (const ev of sorted) {
      const at = this._eventAt(ev) || Date.now();
      if (!current || at - current.endAt > EPISODE_GAP_MS) {
        current = {
          id: `ep_${at}`,
          startAt: at,
          endAt: at,
          events: [],
          types: {},
          tokens: new Set(),
        };
        episodes.push(current);
      }
      current.events.push(ev);
      current.endAt = Math.max(current.endAt, at);
      const type = ev.type || 'neutral';
      current.types[type] = (current.types[type] || 0) + 1;
      for (const t of this._tokens(ev.content)) current.tokens.add(t);
    }

    this.episodes = episodes.slice(-30);
    return this.episodes;
  }

  discoverAssociations(events) {
    const recent = (events || []).slice(-60);
    const byType = {};
    for (const ev of recent) {
      const type = ev.type || 'neutral';
      if (!byType[type]) byType[type] = [];
      byType[type].push(ev);
    }

    const found = [];
    for (const [type, list] of Object.entries(byType)) {
      if (list.length < 2) continue;
      const tokenSet = new Set();
      for (const ev of list) {
        for (const t of this._tokens(ev.content)) tokenSet.add(t);
      }
      const tokens = [...tokenSet];
      for (let i = 0; i < tokens.length; i++) {
        for (let j = i + 1; j < tokens.length; j++) {
          const key = [tokens[i], tokens[j]].sort().join('|');
          const prev = this.associations.get(key) || 0;
          const weight = type === 'scientific' ? 1.4 : type === 'intimate' ? 1.2 : 1;
          this.associations.set(key, prev + weight);
          if (prev === 0) found.push({ a: tokens[i], b: tokens[j], context: type });
        }
      }
    }
    return found;
  }

  extractSchemas(events) {
    const typeCounts = {};
    const hourBuckets = {};
    for (const ev of events || []) {
      const t = ev.type || 'neutral';
      typeCounts[t] = (typeCounts[t] || 0) + 1;
      const h = new Date(this._eventAt(ev) || Date.now()).getHours();
      const bucket = h < 6 ? 'night' : h < 12 ? 'morning' : h < 18 ? 'afternoon' : 'evening';
      hourBuckets[bucket] = (hourBuckets[bucket] || 0) + 1;
    }

    const schemas = [];
    for (const [type, count] of Object.entries(typeCounts)) {
      if (count >= 3) {
        const schema = {
          id: `schema_${type}`,
          label: `反复出现：${type}`,
          confidence: clamp01(count * 0.1),
          count,
        };
        schemas.push(schema);
        this.schemas.set(schema.id, schema);
      }
    }

    const peakHour = Object.entries(hourBuckets).sort((a, b) => b[1] - a[1])[0];
    if (peakHour && peakHour[1] >= 3) {
      schemas.push({
        id: 'schema_time',
        label: `活跃时段偏好：${peakHour[0]}`,
        confidence: clamp01(peakHour[1] * 0.08),
        count: peakHour[1],
      });
    }

    return schemas;
  }

  extractPatterns(events) {
    const schemas = this.extractSchemas(events.slice(-80));
    const patterns = schemas.map((s) => ({
      label: s.label,
      confidence: s.confidence,
      note: `出现 ${s.count} 次`,
    }));
    this.patterns = patterns.slice(-25);
    return patterns;
  }

  /**
   * @param {import('../../lib/memory').MemorySystem} memorySystem
   */
  consolidate(memorySystem) {
    const events = memorySystem?.events || [];
    if (events.length < 4) {
      return { associations: [], patterns: [], insights: [], episodes: [] };
    }

    const episodes = this.segmentEpisodes(events);
    const associations = this.discoverAssociations(events);
    const patterns = this.extractPatterns(events);
    const insights = [];

    for (const ep of episodes.slice(-2)) {
      const dominant = Object.entries(ep.types).sort((a, b) => b[1] - a[1])[0];
      if (dominant) {
        insights.push(`情节片段：以「${dominant[0]}」为主（${ep.events.length} 条）`);
      }
    }

    for (const p of patterns.slice(0, 2)) {
      insights.push(p.label);
    }

    for (const a of associations.slice(0, 4)) {
      insights.push(`「${a.a}」与「${a.b}」常一起出现`);
      if (typeof memorySystem.addObservation === 'function') {
        memorySystem.addObservation(a.a, `常与${a.b}共现（${a.context}）`);
      }
    }

    const sig = memorySystem.getRecentSignificant?.(3) || [];
    for (const line of sig) {
      const m = String(line).match(/] (.+)$/);
      if (m) insights.push(`重要碎片：${m[1]}`);
    }

    this._lastRun = Date.now();
    this._consolidationCount += 1;
    return {
      associations,
      patterns,
      insights: insights.slice(0, 10),
      episodes: episodes.slice(-3),
    };
  }

  topAssociations(n = 6) {
    return [...this.associations.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, n)
      .map(([k, c]) => ({ pair: k.replace('|', '↔'), count: c }));
  }

  toPromptBlock() {
    const top = this.topAssociations(3);
    if (!top.length && !this.patterns.length) return '';
    const lines = [];
    if (this.patterns.length) {
      lines.push(`记忆模式：${this.patterns.slice(-2).map((p) => p.label).join('；')}`);
    }
    if (top.length) {
      lines.push(`联想：${top.map((t) => t.pair).join(' · ')}`);
    }
    return lines.join('\n');
  }

  load(data) {
    if (!data) return;
    if (data.associations) this.associations = new Map(Object.entries(data.associations));
    if (data.patterns) this.patterns = data.patterns;
    if (data.episodes) this.episodes = data.episodes;
    if (data.schemas) this.schemas = new Map(Object.entries(data.schemas));
    if (data._lastRun) this._lastRun = data._lastRun;
    if (data._consolidationCount) this._consolidationCount = data._consolidationCount;
  }

  snapshot() {
    return {
      associations: Object.fromEntries(this.associations),
      patterns: this.patterns,
      episodes: this.episodes.slice(-5),
      schemas: Object.fromEntries(this.schemas),
      top: this.topAssociations(5),
      _lastRun: this._lastRun,
      _consolidationCount: this._consolidationCount,
    };
  }
}

module.exports = { MemoryConsolidation };
