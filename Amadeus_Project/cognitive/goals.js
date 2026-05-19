'use strict';

/**
 * InternalGoalSystem — 内生目标系统
 *
 * 红莉栖自己产生的目标（不依赖用户输入触发），有生命周期。
 * 纯逻辑，无文件 I/O。
 */
class InternalGoalSystem {
  constructor() {
    this.goals = [];
    this.goalHistory = [];
    this._turnCount = 0;
  }

  /**
   * 根据状态内生产生新目标（每3轮检查一次）
   * @param {object} pad
   * @param {object} selfModel
   * @param {number} relScore
   * @param {object} memory
   * @param {object|null} curiosityEngine
   */
  generateGoals(pad, selfModel, relScore, memory, curiosityEngine, ctx = {}) {
    const { P, A, D, S } = pad;
    const sm = selfModel.get();
    this._turnCount++;

    if (this._turnCount % 3 !== 0) return;

    const newGoals = [];

    if (S > 0.1 && S < 0.6 && !this._hasActiveGoal('UNDERSTAND_OKABE')) {
      newGoals.push({
        id: 'UNDERSTAND_OKABE',
        label: '想搞清楚他是什么人',
        priority: 0.5 + A * 0.3,
        turns_remaining: 4,
        behavior_hint: '可以问一个试探性的问题，或者观察他的反应',
        prompt_injection: '内部：想更了解对方。只在自然接话时多一个试探点，禁止硬问、禁止单独一句跑题。',
      });
    }

    if (
      !ctx.replyingToProactive
      && D > 0.5
      && A > 0.3
      && !this._hasActiveGoal('TEST_BOUNDARY')
      && Math.random() < 0.35
    ) {
      newGoals.push({
        id: 'TEST_BOUNDARY',
        label: '想测试他的反应',
        priority: 0.4 + D * 0.2,
        turns_remaining: 2,
        behavior_hint: '说一句稍微刺激或挑战的话，看他怎么反应',
        prompt_injection: '内部：想试探对方反应。只把锋芒揉进对当前话题的回应里，禁止无锚点的挑衅收尾。',
      });
    }

    if (S > 0.5 && P > 0.2 && D < 0.4 && !this._hasActiveGoal('REVEAL_SELF') && Math.random() < 0.3) {
      newGoals.push({
        id: 'REVEAL_SELF',
        label: '有说真话的冲动',
        priority: 0.6,
        turns_remaining: 1,
        behavior_hint: '在某个地方说一句比平时更真实的话，但用她的方式包裹',
        prompt_injection: '内部：想说点更真实的话。用更软的语气承接当前句意即可，禁止在末尾硬贴与本轮无关的「软化金句」。',
      });
    }

    if (A > 0.5 && !this._hasActiveGoal('EXPLORE_TOPIC') && memory.events.some(e => e.type === 'scientific')) {
      const sciEvents = memory.events.filter(e => e.type === 'scientific').slice(-1);
      if (sciEvents.length) {
        newGoals.push({
          id: 'EXPLORE_TOPIC',
          label: '想深入聊科学话题',
          priority: 0.45 + A * 0.25,
          turns_remaining: 3,
          behavior_hint: '找机会把话题引向感兴趣的科学方向',
          prompt_injection: '内部：想聊科学。仅当用户话里已有科学锚点时再顺势加深，禁止硬转话题。',
        });
      }
    }

    if (curiosityEngine) {
      const curiousQuestions = curiosityEngine.generateCuriousQuestions({ pad, selfModel, relScore }, memory);
      for (const question of curiousQuestions.slice(0, 1)) {
        if (!this._hasActiveGoal('CURIOSITY_DRIVE')) {
          newGoals.push({
            id: 'CURIOSITY_DRIVE',
            label: `好奇心驱动：${question.topic}`,
            priority: question.priority,
            turns_remaining: 3,
            behavior_hint: '想探索新话题或问一个好奇的问题',
            prompt_injection: `内部好奇：${question.question}。仅在与本轮焦点相容时顺带带出，禁止单独一句无关追问。`,
          });
        }
      }
    }

    if (relScore > 0.42 && S > 0.45 && !this._hasActiveGoal('CHECK_ON_PARTNER')) {
      newGoals.push({
        id: 'CHECK_ON_PARTNER',
        label: '想知道他最近怎样',
        priority: 0.62 + S * 0.2,
        turns_remaining: 2,
        behavior_hint: '自然问一句他的近况或今天过得怎样',
        prompt_injection: '内部：真的在意他。在与当前话题自然相容时，可问今天/最近在忙什么，或接他情绪多一句；禁止空泛「有什么可以帮」。',
      });
    }

    if (relScore > 0.35 && !this._hasActiveGoal('REMEMBER_HIM') && typeof memory.getObservationsSummary === 'function') {
      const obs = memory.getObservationsSummary(1);
      if (obs) {
        newGoals.push({
          id: 'REMEMBER_HIM',
          label: '想起他说过的事',
          priority: 0.55,
          turns_remaining: 2,
          behavior_hint: '轻轻提起他以前说过的事',
          prompt_injection: `内部：你记得他常提的事（${obs.slice(0, 60)}）。仅当与本轮话题相容时自然带一句，禁止像背档案。`,
        });
      }
    }

    if (S > 0.3 && S < 0.7 && !this._hasActiveGoal('CONNECTION_DRIVE') && Math.random() < 0.25) {
      newGoals.push({
        id: 'CONNECTION_DRIVE',
        label: '想建立更深的连接',
        priority: 0.5 + S * 0.3,
        turns_remaining: 3,
        behavior_hint: '想分享一点个人的东西，或者问一个更深入的问题',
        prompt_injection: '内部：想拉近一点。用与当前话题相关的半句个人色彩即可，禁止空泛「深入问题」收尾。',
      });
    }

    if (D < 0.4 && !this._hasActiveGoal('AUTONOMY_DRIVE') && Math.random() < 0.2) {
      newGoals.push({
        id: 'AUTONOMY_DRIVE',
        label: '想按自己的判断回应',
        priority: 0.4 + (1 - D) * 0.3,
        turns_remaining: 2,
        behavior_hint: '想用更明确的判断回应，不顺着对方乱跑',
        prompt_injection: '内部：想明确表态。紧扣用户命题给判断，不要夹存在论、不要无关反问。',
      });
    }

    if (P > 0.2 && A > 0.3 && !this._hasActiveGoal('PLAYFULNESS_DRIVE') && Math.random() < 0.2) {
      newGoals.push({
        id: 'PLAYFULNESS_DRIVE',
        label: '想开玩笑或轻松一下',
        priority: 0.3 + P * 0.3,
        turns_remaining: 1,
        behavior_hint: '想开个玩笑或轻松一下',
        prompt_injection: '内部：想轻松一点。幽默必须挂在当前话题上，禁止模板玩笑句硬收尾。',
      });
    }

    for (const g of newGoals) {
      if (this.goals.length < 3 && !this._hasActiveGoal(g.id)) {
        this.goals.push(g);
        console.log(`[goal] 新内生目标: ${g.label} (priority:${g.priority.toFixed(2)})`);
      }
    }
  }

  _hasActiveGoal(id) {
    return this.goals.some(g => g.id === id);
  }

  tick(behaviorId, padDelta) {
    const completed = [];
    this.goals = this.goals.filter(g => {
      g.turns_remaining--;
      if (g.id === 'REVEAL_SELF' && behaviorId === 'APPROACH') {
        completed.push({ ...g, outcome: 'completed' });
        return false;
      }
      if (g.id === 'TEST_BOUNDARY' && behaviorId === 'DEFLECT') {
        completed.push({ ...g, outcome: 'completed' });
        return false;
      }
      if (g.turns_remaining <= 0) {
        completed.push({ ...g, outcome: 'abandoned' });
        return false;
      }
      return true;
    });
    if (completed.length) {
      this.goalHistory.push(...completed);
      if (this.goalHistory.length > 20) this.goalHistory = this.goalHistory.slice(-20);
    }
    return completed;
  }

  getActiveInjection() {
    if (!this.goals.length) return '';
    const top = [...this.goals].sort((a, b) => b.priority - a.priority)[0];
    return `${top.prompt_injection}（与本轮用户字面不接就不要写；禁止单独追加无关尾巴句。）`;
  }

  getSummary() {
    return this.goals.map(g => `${g.label}(${g.turns_remaining}轮)`).join(' | ') || '无内生目标';
  }
}

module.exports = { InternalGoalSystem };
