'use strict';

/**
 * 时间感知：日历上下文 + 从用户活跃规律学习日常模式。
 */
class TimePerception {
  constructor() {
    this.routinePatterns = {};
    this._specialDates = new Map([
      ['02-14', '情人节'],
      ['12-25', '圣诞'],
      ['01-01', '新年'],
    ]);
  }

  learnFromUserModel(userModel) {
    const hours = userModel?.model?.patterns?.active_hours || {};
    this.routinePatterns = { ...hours };
    return this.routinePatterns;
  }

  understandTime(now = new Date()) {
    const hour = now.getHours();
    const day = now.getDay();
    const md = `${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}`;
    const period = hour < 6 ? '深夜' : hour < 11 ? '上午' : hour < 14 ? '中午' : hour < 18 ? '下午' : hour < 23 ? '晚上' : '深夜';
    const weekday = ['日', '一', '二', '三', '四', '五', '六'][day];
    const special = this._specialDates.get(md) || '';

    let routineHint = '';
    const peak = Object.entries(this.routinePatterns).sort((a, b) => b[1] - a[1])[0];
    if (peak) {
      routineHint = `他常在 ${peak[0]} 点左右比较活跃`;
    }

    return {
      hour,
      period,
      weekday,
      special,
      routineHint,
      isWeekend: day === 0 || day === 6,
    };
  }

  influenceBehavior(timeCtx) {
    const lines = [];
    if (timeCtx.special) lines.push(`今天是${timeCtx.special}，语气可以稍微不同但别刻意`);
    if (timeCtx.period === '深夜') lines.push('夜深了，话少一点也正常');
    if (timeCtx.isWeekend) lines.push('周末节奏更松');
    if (timeCtx.routineHint) lines.push(timeCtx.routineHint);
    return lines.join('；');
  }

  toPromptLine(userModel) {
    this.learnFromUserModel(userModel);
    const t = this.understandTime();
    const behavior = this.influenceBehavior(t);
    const parts = [`${t.period}·周${t.weekday}`];
    if (behavior) parts.push(behavior);
    return parts.join('；');
  }

  load(data) {
    if (!data) return;
    if (data.routinePatterns) this.routinePatterns = data.routinePatterns;
  }

  snapshot() {
    return {
      routinePatterns: this.routinePatterns,
      now: this.understandTime(),
    };
  }
}

module.exports = { TimePerception };
