'use strict';

const fs = require('fs');
const path = require('path');

/**
 * 行为数据上报管道（不含原生采集）：接收结构化行为 → 自然语言摘要 → 供 prompt / 数字生命消费。
 */
class BehaviorIngest {
  constructor(dataDir) {
    this._path = path.join(dataDir, 'behavior_context.json');
    this.reports = [];
    this.summaries = [];
    this._load();
  }

  _load() {
    try {
      if (!fs.existsSync(this._path)) return;
      const data = JSON.parse(fs.readFileSync(this._path, 'utf8'));
      if (data.reports) this.reports = data.reports;
      if (data.summaries) this.summaries = data.summaries;
    } catch { /* ignore */ }
  }

  _save() {
    try {
      fs.writeFileSync(this._path, JSON.stringify({
        reports: this.reports.slice(-80),
        summaries: this.summaries.slice(-30),
        savedAt: Date.now(),
      }, null, 2));
    } catch { /* ignore */ }
  }

  ingest(payload = {}) {
    const events = Array.isArray(payload.events) ? payload.events : [];
    const at = Date.now();
    const report = {
      at,
      source: payload.source || 'manual',
      events: events.slice(0, 40).map((e) => ({
        app: String(e.app || e.name || 'unknown').slice(0, 40),
        durationMin: Math.max(0, Number(e.durationMin ?? e.duration ?? 0)),
        category: String(e.category || '').slice(0, 24),
      })),
      note: String(payload.note || '').slice(0, 200),
    };
    if (!report.events.length && !report.note) return null;

    this.reports.push(report);
    const summary = this._toNaturalLanguage(report);
    if (summary) {
      this.summaries.push({ at, text: summary });
    }
    if (this.reports.length > 100) this.reports = this.reports.slice(-100);
    if (this.summaries.length > 40) this.summaries = this.summaries.slice(-40);
    this._save();
    return { report, summary };
  }

  _toNaturalLanguage(report) {
    const parts = [];
    if (report.note) parts.push(report.note);
    const byApp = {};
    for (const e of report.events) {
      byApp[e.app] = (byApp[e.app] || 0) + (e.durationMin || 0);
    }
    const top = Object.entries(byApp).sort((a, b) => b[1] - a[1]).slice(0, 4);
    for (const [app, min] of top) {
      if (min >= 5) parts.push(`他大约在 ${app} 上待了 ${Math.round(min)} 分钟`);
      else if (min > 0) parts.push(`他短暂打开过 ${app}`);
    }
    if (/抖音|tiktok|bilibili|哔哩|youtube/i.test(parts.join(' '))) {
      parts.push('像是在刷视频');
    }
    if (/微信|qq|telegram|discord/i.test(parts.join(' '))) {
      parts.push('像是在聊天应用里');
    }
    return parts.length ? parts.join('；') : '';
  }

  latestSummary() {
    const s = this.summaries[this.summaries.length - 1];
    return s?.text || '';
  }

  toPromptLine() {
    const s = this.latestSummary();
    if (!s) return '';
    return `【他最近在做什么】${s}（仅自然时顺带一提，别像监控汇报）`;
  }

  snapshot() {
    return {
      latest: this.latestSummary(),
      reportCount: this.reports.length,
      recent: this.summaries.slice(-3),
    };
  }
}

module.exports = { BehaviorIngest };
