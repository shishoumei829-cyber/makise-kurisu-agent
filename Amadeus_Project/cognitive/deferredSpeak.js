'use strict';

const { parseDueAt } = require('../lib/butler/planner');

/**
 * 到点开口：她的原生时间能力，不依赖「有没有电脑操作权限」。
 * 只要 Amadeus 还在跑，到期就可以主动说话。
 */

const FUTURE_SPEAK_HINT = /(?:叫|喊|提醒|告诉|通知|到点|到时候|到时|记得|别忘|闹钟|起床|叫醒|回来找我|再说|问我|盯着|盯一下|喊我|叫我)/;

function looksLikeFutureSpeakRequest(text) {
  const t = String(text || '').trim();
  if (!t || t.length < 2) return false;
  if (!FUTURE_SPEAK_HINT.test(t)) return false;
  // 需要能解析出时间；解析不到就不当自动登记（她可以追问几点）
  return Boolean(parseDueAt(t));
}

function scheduleContentFromRequest(text) {
  const raw = String(text || '').trim();
  return raw
    .replace(/(?:\d+\s*分钟后|\d+(?:\.\d+)?\s*小时后|\d+\s*天后|(?:今天|明天|后天)?\s*(?:上午|早上|中午|下午|晚上)?\s*\d{1,2}\s*[点时](?:\d{1,2}\s*分?)?)/g, '')
    .replace(/^(?:请|麻烦|帮我|替我|给我)?\s*/, '')
    .replace(/能(?:不能|够)?|可以吗|好吗|行吗|吗|嘛|呢/g, '')
    .replace(/^[，,。？?\s]+|[，,。？?\s]+$/g, '')
    .trim()
    .slice(0, 200) || '到点开口';
}

/**
 * @returns {{ dueAt: number, content: string } | null}
 */
function extractDeferredSpeak(text, now = Date.now()) {
  const t = String(text || '').trim();
  if (!looksLikeFutureSpeakRequest(t)) return null;
  const dueAt = parseDueAt(t, now);
  if (!dueAt || dueAt <= now) return null;
  return { dueAt, content: scheduleContentFromRequest(t) };
}

module.exports = {
  looksLikeFutureSpeakRequest,
  scheduleContentFromRequest,
  extractDeferredSpeak,
  FUTURE_SPEAK_HINT,
};
