'use strict';

const { IntentionStore } = require('./intentionStore');
const { judgeRequest } = require('./judge');
const { extractHerPromise } = require('./promiseExtract');
const {
  buildCapabilitySnapshot,
  formatSnapshotForPrompt,
} = require('./capabilityRegistry');
const { principleLines, createDefaultAxioms } = require('../../brain/axioms/fromSoul');

/**
 * Agency Loop：感知 → 判决 → 承诺（→ 可选投影为 butler task）
 */
class AgencyLoop {
  constructor(options = {}) {
    this.dataDir = options.dataDir;
    this.journal = options.journal || options.butlerKernel?.journal || null;
    this.butlerKernel = options.butlerKernel || null;
    this.intentions = options.intentions
      || new IntentionStore(options.dataDir, this.journal);
  }

  capabilitySnapshot() {
    const toolCaps = this.butlerKernel?.capabilitySnapshot?.() || [];
    return buildCapabilitySnapshot({ toolCaps });
  }

  promptBlock(axioms = createDefaultAxioms()) {
    const snap = this.capabilitySnapshot();
    return [
      ...principleLines(axioms),
      formatSnapshotForPrompt(snap),
    ].join('\n');
  }

  openIntentionsPrompt(limit = 5) {
    const open = this.intentions.list({ openOnly: true }).slice(0, limit);
    if (!open.length) return '';
    const lines = open.map((item) => {
      const when = item.trigger?.kind === 'at_time' && item.trigger.dueAt
        ? ` @${new Date(item.trigger.dueAt).toLocaleString('zh-CN', { hour12: false })}`
        : '';
      return `- [${item.status}] ${item.goal}${when}`;
    });
    return `【未兑现承诺】\n${lines.join('\n')}`;
  }

  /**
   * 入口 A：用户请求
   */
  perceiveUser(text, options = {}) {
    const judgment = judgeRequest(text, { now: options.now });
    if (judgment.verdict === 'forbidden') {
      this.journal?.append('agency.forbidden', {
        text: String(text || '').slice(0, 200),
        alternative: judgment.alternative,
      }, { actor: 'agency', source: options.source || 'chat' });
      return {
        judgment,
        intention: null,
        task: null,
        created: false,
        action: 'agency_forbidden',
        promptHint: judgment.alternative,
      };
    }

    if (judgment.verdict === 'none' || judgment.verdict === 'clarify') {
      return {
        judgment,
        intention: null,
        task: null,
        created: false,
        action: judgment.verdict === 'clarify' ? 'agency_clarify' : 'agency_none',
      };
    }

    if (judgment.verdict === 'augmented') {
      return this._commitAugmented(text, judgment, options);
    }

    // native
    return this._commitNative(text, judgment, options);
  }

  /**
   * 入口 B：她的口头承诺
   */
  perceiveHerReply(input = {}) {
    const promise = extractHerPromise(input);
    if (!promise) {
      return { judgment: null, intention: null, task: null, created: false, action: 'no_promise' };
    }

    if (promise.trigger?.kind === 'at_time' || (promise.effectors || []).includes('speech.deferred')) {
      return this._commitNative(input.userText || promise.goal, {
        verdict: 'native',
        goal: promise.goal,
        trigger: promise.trigger,
        effectors: promise.effectors,
        speakHint: promise.speakHint,
        category: promise.category || 'deferred_speak',
      }, {
        ...input,
        source: 'her_promise',
        herReplyExcerpt: promise.herReplyExcerpt,
        now: input.now,
      });
    }

    if (promise.plan?.length) {
      return this._commitAugmented(input.userText || promise.goal, {
        verdict: 'augmented',
        goal: promise.goal,
        trigger: promise.trigger || { kind: 'immediate' },
        effectors: promise.effectors,
        plan: promise.plan,
        category: promise.category || 'general',
      }, { ...input, source: 'her_promise' });
    }

    const intention = this.intentions.commit({
      goal: promise.goal,
      trigger: promise.trigger || { kind: 'immediate' },
      effectors: promise.effectors || ['commitment.track'],
      source: 'her_promise',
      userText: input.userText,
      herReplyExcerpt: promise.herReplyExcerpt,
      dedupeKey: `promise:${String(input.userText || promise.goal).slice(0, 80)}`,
    });
    return {
      judgment: { verdict: 'native', reason: 'her_promise' },
      intention,
      task: null,
      created: true,
      action: 'intention_committed',
    };
  }

  _commitNative(text, judgment, options = {}) {
    const dueAt = judgment.trigger?.dueAt;
    const dedupeKey = judgment.trigger?.kind === 'at_time' && dueAt
      ? `at_time:${dueAt}:${String(judgment.goal || '').slice(0, 40)}`
      : `native:${String(judgment.goal || text).slice(0, 60)}`;

    // 先落 Intention；到期开口再写 reminder（兼容前端轮询），不走 butler task
    const intention = this.intentions.commit({
      goal: judgment.goal || String(text).slice(0, 200),
      trigger: judgment.trigger || { kind: 'immediate' },
      effectors: judgment.effectors || ['commitment.track'],
      speakHint: judgment.speakHint || judgment.goal,
      source: options.source || 'user_request',
      userText: String(text || '').slice(0, 400),
      herReplyExcerpt: options.herReplyExcerpt || '',
      dedupeKey,
    });

    if (
      judgment.trigger?.kind === 'at_time'
      && this.butlerKernel?.reminders
      && dueAt
      && !intention.reminderId
    ) {
      try {
        const reminder = this.butlerKernel.reminders.create({
          content: judgment.goal || judgment.speakHint || String(text).slice(0, 200),
          dueAt,
        });
        this.intentions.update(intention.id, { reminderId: reminder.id });
        intention.reminderId = reminder.id;
      } catch {
        /* dueAt 已过等 */
      }
    }

    return {
      judgment,
      intention,
      task: null,
      created: true,
      deferred: judgment.trigger?.kind === 'at_time'
        ? { dueAt: judgment.trigger.dueAt, content: judgment.goal }
        : null,
      action: 'intention_committed',
      promptBlock: '',
    };
  }

  _commitAugmented(text, judgment, options = {}) {
    if (!this.butlerKernel?.proposeTask) {
      const intention = this.intentions.commit({
        goal: judgment.goal || String(text).slice(0, 200),
        trigger: judgment.trigger || { kind: 'immediate' },
        effectors: judgment.effectors || [],
        plan: judgment.plan,
        source: options.source || 'user_request',
        userText: String(text || '').slice(0, 400),
        dedupeKey: `aug:${String(judgment.goal || text).slice(0, 60)}`,
      });
      return {
        judgment,
        intention,
        task: null,
        created: true,
        action: 'intention_committed_no_kernel',
      };
    }

    const accepted = this.butlerKernel.proposeTask({
      text: judgment.goal || text,
      title: String(judgment.goal || text).slice(0, 40),
      category: judgment.category || 'system',
      source: options.source || 'agency_judge',
      turnId: options.turnId,
      requestKey: options.requestKey || options.turnId,
      conversationId: options.conversationId,
    });

    const intention = this.intentions.commit({
      goal: judgment.goal || String(text).slice(0, 200),
      trigger: judgment.trigger || { kind: 'immediate' },
      effectors: judgment.effectors || [],
      plan: judgment.plan,
      source: options.source || 'user_request',
      userText: String(text || '').slice(0, 400),
      dedupeKey: `aug:${accepted.task?.id || String(judgment.goal || text).slice(0, 60)}`,
      taskId: accepted.task?.id || '',
    });

    return {
      judgment,
      intention,
      task: accepted.task,
      created: accepted.created,
      action: accepted.created ? 'agency_task_proposed' : 'agency_task_existing',
      promptBlock: accepted.promptBlock || '',
    };
  }

  dueIntentions(at = Date.now()) {
    return this.intentions.due(at).map((item) => {
      this.intentions.markDue(item.id);
      return this.intentions.get(item.id);
    });
  }
}

module.exports = { AgencyLoop };
