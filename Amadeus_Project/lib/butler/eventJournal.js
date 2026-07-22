'use strict';

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

const MAX_RECENT_EVENTS = 2000;

function safePayload(value) {
  if (!value || typeof value !== 'object') return {};
  try {
    const json = JSON.stringify(value);
    if (json.length <= 16000) return JSON.parse(json);
    return { truncated: true, preview: json.slice(0, 12000) };
  } catch {
    return { serializationError: true };
  }
}

/**
 * 统一事件事实源（v2）：
 * 不再是 butler 私有日志——对话、感知、情绪、任务共用同一条 append-only 事实流。
 * `source` 字段区分子系统（butler / dialogue / perception / emotion / behavior…），
 * 订阅者（如意识管线）可实时收到新事件。
 */
class ButlerEventJournal {
  constructor(dataDir) {
    this.dir = path.join(dataDir, 'butler');
    this.path = path.join(this.dir, 'events.jsonl');
    fs.mkdirSync(this.dir, { recursive: true });
    this._subscribers = new Set();
  }

  append(type, payload = {}, meta = {}) {
    const event = {
      id: meta.id || `evt_${crypto.randomUUID()}`,
      ts: Number(meta.ts) || Date.now(),
      type: String(type || 'unknown').slice(0, 100),
      actor: String(meta.actor || 'system').slice(0, 40),
      source: String(meta.source || 'butler').slice(0, 80),
      correlationId: String(meta.correlationId || '').slice(0, 160),
      payload: safePayload(payload),
    };
    fs.appendFileSync(this.path, `${JSON.stringify(event)}\n`, 'utf8');
    for (const listener of this._subscribers) {
      try { listener(event); } catch { /* 订阅者异常不能污染事实源写入 */ }
    }
    return event;
  }

  /** 订阅新事件；返回退订函数 */
  subscribe(listener) {
    if (typeof listener !== 'function') throw new Error('journal subscriber must be a function');
    this._subscribers.add(listener);
    return () => this._subscribers.delete(listener);
  }

  recent(limit = 100, filter = {}) {
    if (!fs.existsSync(this.path)) return [];
    const rows = fs.readFileSync(this.path, 'utf8').split(/\r?\n/).filter(Boolean);
    const out = [];
    for (let i = rows.length - 1; i >= 0 && out.length < Math.min(MAX_RECENT_EVENTS, Math.max(1, limit)); i--) {
      try {
        const event = JSON.parse(rows[i]);
        if (filter.type && event.type !== filter.type) continue;
        if (filter.source && event.source !== filter.source) continue;
        if (filter.correlationId && event.correlationId !== filter.correlationId) continue;
        out.push(event);
      } catch { /* 跳过单行损坏，不影响其余事实 */ }
    }
    return out.reverse();
  }
}

module.exports = { ButlerEventJournal };
