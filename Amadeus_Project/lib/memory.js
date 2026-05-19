const fs = require('fs');
const path = require('path');

const MEMORY_DECAY = 0.005;
const MEMORY_THRESHOLD = 0.02;
const MAX_EVENTS = 500;

function debounceFileWrite(ms, fn) {
  let t = null;
  return () => {
    if (t) clearTimeout(t);
    t = setTimeout(() => {
      t = null;
      fn();
    }, ms);
  };
}

class MemorySystem {
  constructor(dataDir, eventLogPath) {
    this.dataDir = dataDir;
    this.eventLogPath = eventLogPath;
    this.events = this._loadEvents();
    this.timeline = this._loadTimeline();
    this.observations = this._loadObservations();
    this.patterns = this._loadPatterns();

    this._scheduleSaveEvents = debounceFileWrite(200, () => {
      fs.writeFile(this.eventLogPath, JSON.stringify(this.events, null, 2), (err) => {
        if (err) console.error('[memory] Save events error:', err.message);
      });
    });
    this._scheduleSaveTimeline = debounceFileWrite(200, () => {
      fs.writeFile(path.join(this.dataDir, 'timeline.json'), JSON.stringify(this.timeline.slice(-200), null, 2), (err) => {
        if (err) console.error('[memory] Save timeline error:', err.message);
      });
    });
    this._scheduleSaveObservations = debounceFileWrite(200, () => {
      fs.writeFile(path.join(this.dataDir, 'observations.json'), JSON.stringify(this.observations, null, 2), (err) => {
        if (err) console.error('[memory] Save observations error:', err.message);
      });
    });
    this._scheduleSavePatterns = debounceFileWrite(200, () => {
      fs.writeFile(path.join(this.dataDir, 'patterns.json'), JSON.stringify(this.patterns.slice(-30), null, 2), (err) => {
        if (err) console.error('[memory] Save patterns error:', err.message);
      });
    });
  }

  _loadEvents() {
    try {
      if (fs.existsSync(this.eventLogPath)) {
        const raw = JSON.parse(fs.readFileSync(this.eventLogPath, 'utf8'));
        return Array.isArray(raw) ? raw : [];
      }
    } catch {}
    return [];
  }

  _loadTimeline() {
    try {
      const p = path.join(this.dataDir, 'timeline.json');
      if (fs.existsSync(p)) return JSON.parse(fs.readFileSync(p, 'utf8'));
    } catch {}
    return [];
  }

  _loadObservations() {
    try {
      const p = path.join(this.dataDir, 'observations.json');
      if (fs.existsSync(p)) return JSON.parse(fs.readFileSync(p, 'utf8'));
    } catch {}
    return {};
  }

  _loadPatterns() {
    try {
      const p = path.join(this.dataDir, 'patterns.json');
      if (fs.existsSync(p)) return JSON.parse(fs.readFileSync(p, 'utf8'));
    } catch {}
    return [];
  }

  addTimeline(entry) {
    const item = {
      time: new Date().toLocaleString('zh'),
      hour: new Date().getHours(),
      ...entry,
    };
    this.timeline.push(item);
    if (this.timeline.length > 200) this.timeline = this.timeline.slice(-200);
    this._scheduleSaveTimeline();
    return item;
  }

  getTodayTimeline() {
    const today = new Date().toDateString();
    const entries = this.timeline.filter(t => {
      try { return new Date(t.time).toDateString() === today; } catch { return false; }
    });
    if (!entries.length) return '';
    return entries.map(e => `[${e.hour}时] ${e.summary || e.type || ''}`).join(' | ');
  }

  addObservation(keyword, detail = '') {
    if (!keyword || keyword.length < 2) return null;
    const k = keyword.trim();
    if (!this.observations[k]) {
      this.observations[k] = { count: 0, firstSeen: Date.now(), lastSeen: Date.now(), hourDistribution: {}, details: [] };
    }
    const obs = this.observations[k];
    obs.count++;
    obs.lastSeen = Date.now();
    const hour = new Date().getHours();
    obs.hourDistribution[hour] = (obs.hourDistribution[hour] || 0) + 1;
    if (detail && !obs.details.includes(detail)) {
      obs.details.push(detail.substring(0, 60));
      if (obs.details.length > 10) obs.details.shift();
    }
    this._scheduleSaveObservations();

    if (obs.count >= 4 && !this.patterns.find(p => p.source === k)) {
      const peakHour = Object.entries(obs.hourDistribution).sort((a, b) => b[1] - a[1])[0];
      this.patterns.push({
        label: `观察到模式: ${k} (共${obs.count}次)`,
        confidence: Math.min(0.9, obs.count * 0.15),
        source: k,
        discoveredAt: Date.now(),
        note: peakHour ? `常出现在${peakHour[0]}点左右` : '',
      });
      this._scheduleSavePatterns();
    }
    return obs;
  }

  getPatterns(topK = 5) {
    return [...this.patterns].sort((a, b) => b.confidence - a.confidence).slice(0, topK);
  }

  getObservationsSummary(topK = 5) {
    const entries = Object.entries(this.observations)
      .filter(([, o]) => o.count >= 2)
      .sort((a, b) => b[1].count - a[1].count)
      .slice(0, topK);
    if (!entries.length) return '';
    return entries.map(([k, o]) => `${k}(${o.count}次) ${o.details.slice(-2).join('; ')}`).join('\n');
  }

  addEvent(type, content, importance, padDelta = {}) {
    const event = {
      id:         Date.now(),
      type,
      content:    content.substring(0, 80),
      importance: Math.max(0, Math.min(1, importance)),
      padDelta:   { P: padDelta.P||0, A: padDelta.A||0, D: padDelta.D||0 },
      timestamp:  Date.now(),
      weight:     importance,
    };
    this.events.push(event);
    if (this.events.length > MAX_EVENTS) {
      this.events.sort((a, b) => b.weight - a.weight);
      this.events = this.events.slice(0, MAX_EVENTS);
    }
    this._scheduleSaveEvents();
    this.addTimeline({ type: event.type, summary: event.content });
    return event;
  }

  decay() {
    const now = Date.now();
    this.events = this.events
      .map(ev => {
        const daysSince = (now - ev.timestamp) / 86400000;
        ev.weight = ev.importance * Math.exp(-MEMORY_DECAY * daysSince);
        return ev;
      })
      .filter(ev => ev.weight > MEMORY_THRESHOLD);
    this._scheduleSaveEvents();
  }

  getLongTermPadBias() {
    if (!this.events.length) return { P: 0, A: 0, D: 0 };
    let P = 0, A = 0, D = 0, totalW = 0;
    for (const ev of this.events) {
      P += ev.padDelta.P * ev.weight;
      A += ev.padDelta.A * ev.weight;
      D += ev.padDelta.D * ev.weight;
      totalW += ev.weight;
    }
    if (totalW === 0) return { P: 0, A: 0, D: 0 };
    const scale = Math.min(1, totalW);
    return {
      P: Math.max(-0.4, Math.min(0.4, (P / totalW) * scale)),
      A: Math.max(-0.3, Math.min(0.3, (A / totalW) * scale)),
      D: Math.max(-0.3, Math.min(0.3, (D / totalW) * scale)),
    };
  }

  getRecentSignificant(topK = 5) {
    return [...this.events]
      .sort((a, b) => b.weight - a.weight)
      .slice(0, topK)
      .map(ev => `[${ev.type}|w:${ev.weight.toFixed(2)}] ${ev.content}`);
  }

  getRelationshipScore() {
    let pos = 0, neg = 0;
    for (const ev of this.events) {
      if (['positive', 'intimate', 'scientific'].includes(ev.type)) pos += ev.weight;
      if (['negative', 'conflict'].includes(ev.type)) neg += ev.weight;
    }
    return Math.max(-1, Math.min(1, (pos - neg) / Math.max(1, pos + neg)));
  }
}

module.exports = { MemorySystem };
