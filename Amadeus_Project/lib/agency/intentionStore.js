'use strict';

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

class IntentionStore {
  constructor(dataDir, journal = null) {
    this.dir = path.join(dataDir, 'agency');
    this.path = path.join(this.dir, 'intentions.json');
    this.journal = journal;
    fs.mkdirSync(this.dir, { recursive: true });
    this.intentions = this._load();
  }

  _load() {
    try {
      const parsed = JSON.parse(fs.readFileSync(this.path, 'utf8'));
      return Array.isArray(parsed.intentions) ? parsed.intentions : [];
    } catch {
      return [];
    }
  }

  _save() {
    const temp = `${this.path}.tmp`;
    fs.writeFileSync(temp, JSON.stringify({
      version: 1,
      savedAt: Date.now(),
      intentions: this.intentions,
    }, null, 2), 'utf8');
    fs.renameSync(temp, this.path);
  }

  get(id) {
    return this.intentions.find((item) => item.id === id) || null;
  }

  list(filter = {}) {
    return this.intentions
      .filter((item) => {
        if (filter.status && item.status !== filter.status) return false;
        if (filter.openOnly) {
          return !['fulfilled', 'failed', 'cancelled', 'reneged'].includes(item.status);
        }
        return true;
      })
      .slice()
      .sort((a, b) => (a.trigger?.dueAt || a.createdAt || 0) - (b.trigger?.dueAt || b.createdAt || 0));
  }

  due(at = Date.now()) {
    return this.intentions.filter((item) => {
      if (item.status !== 'committed' && item.status !== 'due') return false;
      if (item.trigger?.kind !== 'at_time') return false;
      const dueAt = Number(item.trigger.dueAt);
      if (!Number.isFinite(dueAt) || dueAt > at) return false;
      const windowMs = Number(item.trigger.windowMs) || 15 * 60 * 1000;
      return at <= dueAt + windowMs;
    });
  }

  commit(input = {}) {
    const goal = String(input.goal || '').trim().slice(0, 300);
    if (!goal) throw new Error('intention goal required');
    const trigger = input.trigger && typeof input.trigger === 'object'
      ? { ...input.trigger }
      : { kind: 'immediate' };
    if (trigger.kind === 'at_time') {
      const dueAt = Number(trigger.dueAt);
      if (!Number.isFinite(dueAt) || dueAt <= Date.now() - 1000) {
        throw new Error('at_time intention needs future dueAt');
      }
      trigger.dueAt = dueAt;
      trigger.windowMs = Number(trigger.windowMs) || 15 * 60 * 1000;
    }

    const existingKey = String(input.dedupeKey || '').trim();
    if (existingKey) {
      const hit = this.intentions.find((item) => item.dedupeKey === existingKey
        && !['fulfilled', 'failed', 'cancelled', 'reneged'].includes(item.status));
      if (hit) return hit;
    }

    const intention = {
      id: `intention_${crypto.randomUUID()}`,
      goal,
      status: 'committed',
      trigger,
      effectors: Array.isArray(input.effectors) ? input.effectors.slice(0, 12) : [],
      plan: Array.isArray(input.plan) ? input.plan : undefined,
      source: String(input.source || 'user_request'),
      evidence: [],
      dedupeKey: existingKey || undefined,
      speakHint: String(input.speakHint || goal).slice(0, 240),
      userText: String(input.userText || '').slice(0, 400),
      herReplyExcerpt: String(input.herReplyExcerpt || '').slice(0, 240),
      taskId: String(input.taskId || ''),
      reminderId: String(input.reminderId || ''),
      createdAt: Date.now(),
      updatedAt: Date.now(),
    };
    this.intentions.push(intention);
    this._save();
    this.journal?.append('intention.committed', {
      intentionId: intention.id,
      goal: intention.goal,
      trigger: intention.trigger,
    }, { actor: 'agency', correlationId: intention.id });
    return intention;
  }

  update(id, patch = {}) {
    const item = this.get(id);
    if (!item) throw new Error('intention not found');
    Object.assign(item, patch, { updatedAt: Date.now() });
    this._save();
    return item;
  }

  markDue(id) {
    const item = this.get(id);
    if (!item) return null;
    if (item.status === 'committed') {
      item.status = 'due';
      item.updatedAt = Date.now();
      this._save();
      this.journal?.append('intention.due', { intentionId: id }, { actor: 'agency', correlationId: id });
    }
    return item;
  }

  markFulfilled(id, evidence = null) {
    const item = this.get(id);
    if (!item) throw new Error('intention not found');
    item.status = 'fulfilled';
    item.fulfilledAt = Date.now();
    item.updatedAt = item.fulfilledAt;
    if (evidence) item.evidence.push(evidence);
    this._save();
    this.journal?.append('intention.fulfilled', { intentionId: id }, { actor: 'agency', correlationId: id });
    return item;
  }

  markMissed(id) {
    const item = this.get(id);
    if (!item) return null;
    item.status = 'failed';
    item.failReason = 'missed_window';
    item.updatedAt = Date.now();
    this._save();
    return item;
  }
}

module.exports = { IntentionStore };
