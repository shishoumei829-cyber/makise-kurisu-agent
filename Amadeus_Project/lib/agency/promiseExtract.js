'use strict';

const { parseDueAt } = require('../butler/planner');
const { extractDeferredSpeak, scheduleContentFromRequest } = require('../../cognitive/deferredSpeak');
const { judgeRequest } = require('./judge');

const ACCEPT_RE = /(?:好的?|行|可以|没问题|交给我|我记(?:住|下)?了|到点(?:叫|喊|提醒|找)|我会(?:叫|喊|提醒|说|记)|别担心|包在我身上)/;
const REJECT_RE = /(?:才不要|我才不管|办不到|做不到|别指望|不可能|滚)/;

/**
 * 从她的回复抽取承诺；优先绑定用户请求里的时间/目标。
 */
function extractHerPromise(input = {}) {
  const userText = String(input.userText || '').trim();
  const replyText = String(input.replyText || '').trim();
  const now = Number(input.now) || Date.now();
  if (!replyText || REJECT_RE.test(replyText)) return null;
  if (!ACCEPT_RE.test(replyText)) return null;

  const userJudge = userText ? judgeRequest(userText, { now }) : { verdict: 'none' };
  if (userJudge.verdict === 'forbidden') return null;

  if (userJudge.verdict === 'native' || userJudge.verdict === 'augmented') {
    return {
      goal: userJudge.goal || scheduleContentFromRequest(userText),
      trigger: userJudge.trigger || { kind: 'immediate' },
      effectors: userJudge.effectors || ['commitment.track'],
      plan: userJudge.plan,
      speakHint: userJudge.speakHint || userJudge.goal,
      category: userJudge.category,
      source: 'her_promise',
      userText,
      herReplyExcerpt: replyText.slice(0, 160),
    };
  }

  const fromReply = extractDeferredSpeak(replyText, now);
  if (fromReply) {
    return {
      goal: fromReply.content,
      trigger: { kind: 'at_time', dueAt: fromReply.dueAt, windowMs: 15 * 60 * 1000 },
      effectors: ['speech.deferred', 'time.schedule', 'commitment.track'],
      speakHint: fromReply.content,
      source: 'her_promise',
      userText,
      herReplyExcerpt: replyText.slice(0, 160),
    };
  }

  const dueFromUser = parseDueAt(userText, now);
  const dueFromReply = parseDueAt(replyText, now);
  const dueAt = (dueFromReply && dueFromReply > now) ? dueFromReply
    : (dueFromUser && dueFromUser > now) ? dueFromUser
      : null;

  if (dueAt && /叫|喊|提醒|找|说|开口|起床|闹钟/.test(`${userText} ${replyText}`)) {
    return {
      goal: scheduleContentFromRequest(userText || replyText),
      trigger: { kind: 'at_time', dueAt, windowMs: 15 * 60 * 1000 },
      effectors: ['speech.deferred', 'time.schedule', 'commitment.track'],
      speakHint: scheduleContentFromRequest(userText || replyText),
      source: 'her_promise',
      userText,
      herReplyExcerpt: replyText.slice(0, 160),
    };
  }

  if (/记住|记下|我记/.test(replyText) && userText) {
    return {
      goal: userText.slice(0, 200),
      trigger: { kind: 'immediate' },
      effectors: ['memory.retain', 'commitment.track'],
      source: 'her_promise',
      userText,
      herReplyExcerpt: replyText.slice(0, 160),
    };
  }

  return null;
}

module.exports = { extractHerPromise, ACCEPT_RE, REJECT_RE };
