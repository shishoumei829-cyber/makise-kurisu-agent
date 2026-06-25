'use strict';

const { SPECIAL_DATES } = require('./constants');

/**
 * 时间感知：节律学习、时段语义、行为影响。
 */
class TimeSense {
  constructor() {
    this.routinePatterns = {};
    this._lastLearnAt = 0;
    this.tempo = 'normal';
  }

  learnFromUserModel(userModel) {
    const hours = userModel?.model?.patterns?.active_hours || {};
    this.routinePatterns = { ...hours };
    this._lastLearnAt = Date.now();
    return this.routinePatterns;
  }

  understandTime(now = new Date()) {
    const hour = now.getHours();
    const day = now.getDay();
    const md = `${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}`;
    const period = hour < 6 ? '深夜' : hour < 11 ? '上午' : hour < 14 ? '中午' : hour < 18 ? '下午' : hour < 23 ? '晚上' : '深夜';
    const weekday = ['日', '一', '二', '三', '四', '五', '六'][day];
    const special = SPECIAL_DATES.get(md) || '';

    let routineHint = '';
    const peak = Object.entries(this.routinePatterns).sort((a, b) => b[1] - a[1])[0];
    if (peak) routineHint = `他常在 ${peak[0]} 点左右比较活跃`;

    const isOffRoutine = peak && Math.abs(Number(peak[0]) - hour) > 4;

    return {
      hour,
      period,
      weekday,
      special,
      routineHint,
      isWeekend: day === 0 || day === 6,
      isOffRoutine,
    };
  }

  influenceBehavior(timeCtx, idleMs = 0) {
    const lines = [];
    if (timeCtx.special) lines.push(`今天是${timeCtx.special}，语气可略不同但别刻意`);
    if (timeCtx.period === '深夜') {
      lines.push('夜深了，话少一点也正常');
      this.tempo = 'slow';
    } else if (timeCtx.period === '上午') {
      this.tempo = 'crisp';
    } else {
      this.tempo = 'normal';
    }
    if (timeCtx.isWeekend) lines.push('周末节奏更松');
    if (timeCtx.routineHint) lines.push(timeCtx.routineHint);
    if (timeCtx.isOffRoutine) lines.push('这个点不常聊，可能有事或睡不着');
    if (idleMs > 60 * 60 * 1000) lines.push('隔了很久才出现，时间感要拉长');
    return lines.join('；');
  }

  toPromptLine(userModel, idleMs = 0) {
    this.learnFromUserModel(userModel);
    const t = this.understandTime();
    const behavior = this.influenceBehavior(t, idleMs);
    const parts = [`${t.period}·周${t.weekday}`];
    if (behavior) parts.push(behavior);
    return parts.join('；');
  }

  load(data) {
    if (!data) return;
    if (data.routinePatterns) this.routinePatterns = data.routinePatterns;
    if (data.tempo) this.tempo = data.tempo;
  }

  snapshot() {
    return {
      routinePatterns: this.routinePatterns,
      tempo: this.tempo,
      now: this.understandTime(),
    };
  }
}

module.exports = { TimeSense };
