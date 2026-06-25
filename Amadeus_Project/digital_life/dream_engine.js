'use strict';

/**
 * 梦境引擎：独处时整理记忆并生成内在叙事碎片。
 */
class DreamEngine {
  constructor() {
    this.dreamQueue = [];
    this.dreamJournal = [];
    this._lastDreamAt = 0;
  }

  consolidateMemory(memoryReorg, memorySystem) {
    if (!memoryReorg || !memorySystem) return [];
    return memoryReorg.consolidate(memorySystem).insights;
  }

  generateDream(context = {}) {
    const { insights = [], pad = {}, recentEvents = [], idleMin = 0 } = context;
    const fragments = [];
    const mood = pad.P > 0.2 ? '偏暖' : pad.P < -0.2 ? '偏沉' : '平静';

    if (insights.length) {
      fragments.push(`梦里反复出现：${insights[Math.floor(Math.random() * insights.length)]}`);
    }
    if (recentEvents.length) {
      const ev = recentEvents[Math.floor(Math.random() * recentEvents.length)];
      const content = typeof ev === 'string' ? ev : ev.content;
      if (content) fragments.push(`场景里闪过：${String(content).slice(0, 40)}`);
    }
    if (idleMin >= 60) {
      fragments.push('时间拉得很长，像实验室只剩仪器低鸣');
    } else if (idleMin >= 20) {
      fragments.push('安静太久，念头自己浮上来');
    }
    fragments.push(`整体色调：${mood}`);

    const dream = {
      text: fragments.join('；'),
      mood,
      at: Date.now(),
      idleMin,
    };
    this.dreamQueue.push(dream);
    if (this.dreamQueue.length > 10) this.dreamQueue.shift();
    this.dreamJournal.push(dream);
    if (this.dreamJournal.length > 50) this.dreamJournal = this.dreamJournal.slice(-50);
    this._lastDreamAt = dream.at;
    return dream;
  }

  shouldDream(idleMs, minIdleMs = 25 * 60 * 1000) {
    const idle = Number(idleMs) || 0;
    if (idle < minIdleMs) return false;
    if (Date.now() - this._lastDreamAt < 20 * 60 * 1000) return false;
    return true;
  }

  latestDreamLine() {
    const d = this.dreamJournal[this.dreamJournal.length - 1];
    if (!d) return '';
    return `独处残影：${d.text.slice(0, 100)}`;
  }

  load(data) {
    if (!data) return;
    if (data.dreamQueue) this.dreamQueue = data.dreamQueue;
    if (data.dreamJournal) this.dreamJournal = data.dreamJournal;
    if (data._lastDreamAt) this._lastDreamAt = data._lastDreamAt;
  }

  snapshot() {
    return {
      dreamQueue: this.dreamQueue.slice(-3),
      dreamJournal: this.dreamJournal.slice(-5),
      _lastDreamAt: this._lastDreamAt,
    };
  }
}

module.exports = { DreamEngine };
