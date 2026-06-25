'use strict';

const { SLEEP_PHASES } = require('./constants');

function clamp01(v) {
  return Math.max(0, Math.min(1, v));
}

/**
 * 梦境引擎：睡眠阶段、记忆碎片重组、情绪色调、主动开口 carryover。
 */
class DreamEngine {
  constructor() {
    this.dreamQueue = [];
    this.dreamJournal = [];
    this.carryover = null;
    this._lastDreamAt = 0;
    this._sleepPhase = SLEEP_PHASES.LIGHT;
    this._dreamCount = 0;
  }

  _pick(arr) {
    if (!arr?.length) return null;
    return arr[Math.floor(Math.random() * arr.length)];
  }

  _phaseFromIdle(idleMin) {
    if (idleMin >= 90) return SLEEP_PHASES.REM;
    if (idleMin >= 45) return SLEEP_PHASES.DEEP;
    return SLEEP_PHASES.LIGHT;
  }

  consolidateMemory(memoryReorg, memorySystem) {
    if (!memoryReorg || !memorySystem) return [];
    const pack = memoryReorg.consolidate(memorySystem);
    return pack.insights || [];
  }

  _composeNarrative({ insights, pad, recentEvents, idleMin, phase, associations }) {
    const fragments = [];
    const mood = pad.P > 0.25 ? '偏暖' : pad.P < -0.25 ? '偏沉' : pad.A > 0.5 ? '躁动' : '平静';

    if (phase === SLEEP_PHASES.REM && insights.length) {
      const insight = this._pick(insights);
      fragments.push(`梦里把「${insight}」拆成两半又拼回去`);
    } else if (insights.length) {
      fragments.push(`反复出现：${this._pick(insights)}`);
    }

    if (associations?.length) {
      const a = this._pick(associations);
      fragments.push(`两个词在梦里粘在一起：${a.a}和${a.b}`);
    }

    if (recentEvents.length) {
      const ev = this._pick(recentEvents);
      const content = typeof ev === 'string' ? ev : ev.content;
      if (content) fragments.push(`场景闪过：${String(content).slice(0, 36)}`);
    }

    if (idleMin >= 60) {
      fragments.push('时间被拉得很长，像只剩仪器低鸣');
    } else if (idleMin >= 25) {
      fragments.push('安静太久，念头自己浮上来');
    }

    if (pad.A > 0.55 && pad.P < 0) {
      fragments.push('有种想说又咽回去的感觉');
    }

    fragments.push(`色调：${mood}（${phase}）`);
    return { text: fragments.join('；'), mood, phase };
  }

  generateDream(context = {}) {
    const {
      insights = [],
      pad = {},
      recentEvents = [],
      idleMin = 0,
      associations = [],
    } = context;

    const phase = this._phaseFromIdle(idleMin);
    this._sleepPhase = phase;
    const narrative = this._composeNarrative({
      insights,
      pad,
      recentEvents,
      idleMin,
      phase,
      associations,
    });

    const dream = {
      text: narrative.text,
      mood: narrative.mood,
      phase,
      at: Date.now(),
      idleMin,
      intensity: clamp01(0.35 + idleMin / 120 + (insights.length ? 0.15 : 0)),
    };

    this.dreamQueue.push(dream);
    if (this.dreamQueue.length > 12) this.dreamQueue.shift();
    this.dreamJournal.push(dream);
    if (this.dreamJournal.length > 60) this.dreamJournal = this.dreamJournal.slice(-60);
    this._lastDreamAt = dream.at;
    this._dreamCount += 1;

    if (dream.intensity >= 0.55) {
      this.carryover = {
        hint: dream.text.slice(0, 90),
        mood: dream.mood,
        expiresAt: Date.now() + 40 * 60 * 1000,
        proactiveEligible: phase === SLEEP_PHASES.REM || idleMin >= 40,
      };
    }

    return dream;
  }

  getCarryover() {
    if (!this.carryover) return null;
    if (Date.now() > this.carryover.expiresAt) {
      this.carryover = null;
      return null;
    }
    return this.carryover;
  }

  shouldDream(idleMs, minIdleMs = 25 * 60 * 1000) {
    const idle = Number(idleMs) || 0;
    if (idle < minIdleMs) return false;
    if (Date.now() - this._lastDreamAt < 18 * 60 * 1000) return false;
    return true;
  }

  latestDreamLine() {
    const d = this.dreamJournal[this.dreamJournal.length - 1];
    if (!d) return '';
    return `独处残影：${d.text.slice(0, 100)}`;
  }

  proactiveHint() {
    const c = this.getCarryover();
    if (!c?.proactiveEligible) return '';
    return `醒来残念：${c.hint}（仅自然时顺带一提）`;
  }

  toPromptBlock(includeDream = true) {
    const lines = [];
    if (includeDream) {
      const line = this.latestDreamLine();
      if (line) lines.push(line);
    }
    const hint = this.proactiveHint();
    if (hint) lines.push(hint);
    return lines.join('\n');
  }

  load(data) {
    if (!data) return;
    if (data.dreamQueue) this.dreamQueue = data.dreamQueue;
    if (data.dreamJournal) this.dreamJournal = data.dreamJournal;
    if (data.carryover) this.carryover = data.carryover;
    if (data._lastDreamAt) this._lastDreamAt = data._lastDreamAt;
    if (data._sleepPhase) this._sleepPhase = data._sleepPhase;
    if (data._dreamCount) this._dreamCount = data._dreamCount;
  }

  snapshot() {
    return {
      dreamQueue: this.dreamQueue.slice(-3),
      dreamJournal: this.dreamJournal.slice(-5),
      carryover: this.getCarryover(),
      _lastDreamAt: this._lastDreamAt,
      _sleepPhase: this._sleepPhase,
      _dreamCount: this._dreamCount,
    };
  }
}

module.exports = { DreamEngine };
