'use strict';

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

class ReminderStore {
  constructor(dataDir, journal) {
    this.dir = path.join(dataDir, 'butler');
    this.path = path.join(this.dir, 'reminders.json');
    this.journal = journal;
    fs.mkdirSync(this.dir, { recursive: true });
    this.reminders = this._load();
  }

  _load() {
    try {
      const parsed = JSON.parse(fs.readFileSync(this.path, 'utf8'));
      return Array.isArray(parsed.reminders) ? parsed.reminders : [];
    } catch { return []; }
  }

  _save() {
    const temp = `${this.path}.tmp`;
    fs.writeFileSync(temp, JSON.stringify({ version: 1, savedAt: Date.now(), reminders: this.reminders }, null, 2), 'utf8');
    fs.renameSync(temp, this.path);
  }

  create(input = {}) {
    const content = String(input.content || '').trim();
    const dueAt = Number(input.dueAt);
    if (!content) throw new Error('reminder content required');
    if (!Number.isFinite(dueAt) || dueAt <= Date.now()) throw new Error('reminder dueAt must be in the future');
    const reminder = {
      id: `reminder_${crypto.randomUUID()}`,
      content: content.slice(0, 500),
      dueAt,
      status: 'scheduled',
      taskId: String(input.taskId || ''),
      createdAt: Date.now(),
      deliveredAt: null,
      acknowledgedAt: null,
    };
    this.reminders.push(reminder);
    this._save();
    this.journal?.append('reminder.created', { reminder }, { actor: 'executor', correlationId: reminder.taskId || reminder.id });
    return reminder;
  }

  get(id) { return this.reminders.find((item) => item.id === id) || null; }

  list(filter = {}) {
    return this.reminders
      .filter((item) => !filter.status || item.status === filter.status)
      .slice()
      .sort((a, b) => a.dueAt - b.dueAt);
  }

  due(at = Date.now()) {
    return this.reminders.filter((item) => item.status === 'scheduled' && item.dueAt <= at);
  }

  markDelivered(id) {
    const reminder = this.get(id);
    if (!reminder) throw new Error('reminder not found');
    if (reminder.status === 'scheduled') {
      reminder.status = 'delivered';
      reminder.deliveredAt = Date.now();
      this._save();
      this.journal?.append('reminder.delivered', { reminderId: id }, { actor: 'system', correlationId: reminder.taskId || id });
    }
    return reminder;
  }

  acknowledge(id) {
    const reminder = this.get(id);
    if (!reminder) throw new Error('reminder not found');
    reminder.status = 'acknowledged';
    reminder.acknowledgedAt = Date.now();
    this._save();
    this.journal?.append('reminder.acknowledged', { reminderId: id }, { actor: 'user', correlationId: reminder.taskId || id });
    return reminder;
  }
}

module.exports = { ReminderStore };
