'use strict';

/**
 * 记忆重组：离线整理事件、发现关联、提取模式。
 */
class MemoryReorganization {
  constructor() {
    this.associations = new Map();
    this.patterns = [];
    this._lastRun = 0;
  }

  _tokensFromEvents(events) {
    const tokens = new Set();
    for (const ev of events || []) {
      for (const t of String(ev.content || '').match(/[\u4e00-\u9fa5]{2,}/g) || []) {
        tokens.add(t);
      }
    }
    return [...tokens];
  }

  discoverAssociations(events) {
    const recent = (events || []).slice(-40);
    const byType = {};
    for (const ev of recent) {
      const type = ev.type || 'neutral';
      if (!byType[type]) byType[type] = [];
      byType[type].push(ev);
    }
    const found = [];
    for (const [type, list] of Object.entries(byType)) {
      if (list.length < 2) continue;
      const tokens = this._tokensFromEvents(list);
      for (let i = 0; i < tokens.length; i++) {
        for (let j = i + 1; j < tokens.length; j++) {
          const key = [tokens[i], tokens[j]].sort().join('|');
          const prev = this.associations.get(key) || 0;
          this.associations.set(key, prev + 1);
          if (prev === 0) {
            found.push({ a: tokens[i], b: tokens[j], context: type });
          }
        }
      }
    }
    return found;
  }

  extractPatterns(events) {
    const typeCounts = {};
    for (const ev of events || []) {
      const t = ev.type || 'neutral';
      typeCounts[t] = (typeCounts[t] || 0) + 1;
    }
    const patterns = [];
    for (const [type, count] of Object.entries(typeCounts)) {
      if (count >= 3) {
        patterns.push({
          label: `反复出现的事件类型：${type}`,
          confidence: Math.min(0.95, count * 0.12),
          note: `近 ${events.length} 条记忆中 ${count} 次`,
        });
      }
    }
    this.patterns = patterns.slice(-20);
    return patterns;
  }

  /**
   * @param {import('../lib/memory').MemorySystem} memorySystem
   */
  consolidate(memorySystem) {
    const events = memorySystem?.events || [];
    if (events.length < 4) return { associations: [], patterns: [], insights: [] };

    const associations = this.discoverAssociations(events);
    const patterns = this.extractPatterns(events.slice(-80));
    const insights = [];

    for (const p of patterns.slice(0, 2)) {
      insights.push(p.label);
    }
    for (const a of associations.slice(0, 3)) {
      insights.push(`「${a.a}」与「${a.b}」常一起出现`);
      if (typeof memorySystem.addObservation === 'function') {
        memorySystem.addObservation(a.a, `常与${a.b}共现`);
      }
    }

    const sig = memorySystem.getRecentSignificant(3);
    for (const line of sig) {
      const m = line.match(/] (.+)$/);
      if (m) insights.push(`重要碎片：${m[1]}`);
    }

    this._lastRun = Date.now();
    return { associations, patterns, insights: insights.slice(0, 8) };
  }

  topAssociations(n = 5) {
    return [...this.associations.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, n)
      .map(([k, c]) => ({ pair: k.replace('|', '↔'), count: c }));
  }

  load(data) {
    if (!data) return;
    if (data.associations) this.associations = new Map(Object.entries(data.associations));
    if (data.patterns) this.patterns = data.patterns;
    if (data._lastRun) this._lastRun = data._lastRun;
  }

  snapshot() {
    return {
      associations: Object.fromEntries(this.associations),
      patterns: this.patterns,
      _lastRun: this._lastRun,
      top: this.topAssociations(5),
    };
  }
}

module.exports = { MemoryReorganization };
