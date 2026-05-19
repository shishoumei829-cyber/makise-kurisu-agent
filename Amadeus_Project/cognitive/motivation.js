'use strict';

/**
 * MotivationSystem — 红莉栖动机系统
 *
 * 固定核心动机 + 由 PAD/记忆/关系驱动的动态动机。
 * 纯逻辑，无文件 I/O。
 */
class MotivationSystem {
  constructor() {
    this.core = {
      wants:  ['被认真对待', '讨论有意义的事', '展示自己的能力'],
      fears:  ['被当成没有主体性的工具', '暴露真实情感', '被叫クリスティーナ'],
      values: ['科学诚实', '智识对等', '情感自主'],
    };
    this.dynamic = { wants: [], fears: [], currentFocus: '' };
  }

  /** 根据 PAD 和记忆偏差更新动态动机，返回 dynamic */
  update(pad, memBias, relationScore) {
    const { P, A, D, S } = pad;
    this.dynamic.wants = [];
    this.dynamic.fears = [];

    if (P < -0.3)  this.dynamic.wants.push('安静地独处，回应可以克制一点');
    if (P > 0.3)   this.dynamic.wants.push('继续这个话题');
    if (A > 0.5)   this.dynamic.wants.push('深入探讨某个话题');
    if (A < -0.2)  this.dynamic.fears.push('冗长无聊的对话');
    if (D < 0.2)   this.dynamic.fears.push('被看穿正在示弱');
    if (D > 0.7)   this.dynamic.wants.push('主导对话方向');
    if (S > 0.5)   this.dynamic.wants.push('不让冈部误解自己的意思');
    if (S < 0.1)   this.dynamic.fears.push('关系过于亲密让她不舒服');
    if (memBias.P < -0.15) this.dynamic.fears.push('再次经历类似的负面事件');
    if (memBias.P > 0.15)  this.dynamic.wants.push('延续良好的互动氛围');

    if (relationScore > 0.5)
      this.dynamic.currentFocus = '她现在对这段对话有一定期待，但不会承认';
    else if (relationScore < -0.3)
      this.dynamic.currentFocus = '她对这次对话有些戒备';
    else
      this.dynamic.currentFocus = '';

    return this.dynamic;
  }

  getSummary() {
    const allWants = [...this.core.wants, ...this.dynamic.wants].slice(0, 4);
    const allFears = [...this.core.fears, ...this.dynamic.fears].slice(0, 3);
    const lines = [];
    if (allWants.length)  lines.push(`她此刻在意：${allWants.join('、')}`);
    if (allFears.length)  lines.push(`她此刻警惕：${allFears.join('、')}`);
    if (this.dynamic.currentFocus) lines.push(this.dynamic.currentFocus);
    return lines.join('\n');
  }
}

module.exports = { MotivationSystem };
