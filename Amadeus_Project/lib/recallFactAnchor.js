'use strict';

/**
 * 回忆锚定：从实录 + 宫殿抽用户亲口事实，压张冠李戴。
 * 结构抽取，不是黑名单样本堆。
 */

const { needsConversationRecall } = require('./unifiedDialogueLog');

const FACTISH_USER = /请记住|记住|喜欢|不喝|讨厌|叫我|以后叫|过敏|通常.{0,8}睡|凌晨|明天.{0,12}去|实习|前端|养狗|养猫|没有猫|小黑|周五|面试|显卡|秋叶原|不喜欢被叫/;

function needsFactRecall(userText) {
  const t = String(userText || '').trim();
  if (!t) return false;
  if (needsConversationRecall(t)) return true;
  return /喜欢喝|喝什么|几点睡|怎么叫我|叫我什么|养的是什么|养什么|过敏|周五.{0,6}(?:安排|有什么|干嘛|面试)|明天去哪|干什么的|做什么的|记成|是不是说|还记得|总结一下/.test(t);
}

function collectPinnedFacts(dialogueLog, limit = 10) {
  const entries = typeof dialogueLog?.getRecent === 'function'
    ? dialogueLog.getRecent(120)
    : (dialogueLog?.entries || []);
  const out = [];
  const seen = new Set();
  for (let i = entries.length - 1; i >= 0 && out.length < limit; i--) {
    const e = entries[i];
    if (!e || e.role !== 'user') continue;
    const t = String(e.text || '').trim();
    if (!t || !FACTISH_USER.test(t)) continue;
    const key = t.slice(0, 64);
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(t.slice(0, 100));
  }
  return out.reverse();
}

/**
 * @returns {string} prompt block or ''
 */
function buildRecallFactAnchor(opts = {}) {
  const userText = String(opts.userText || '').trim();
  if (!needsFactRecall(userText)) return '';

  const facts = collectPinnedFacts(opts.dialogueLog, 10);
  let palaceBit = '';
  if (opts.memoryPalace && typeof opts.memoryPalace.navigate === 'function') {
    try {
      const nav = opts.memoryPalace.navigate(userText, { topK: 5, minScore: 1.5 });
      palaceBit = String(nav?.excerpt || '').trim().slice(0, 600);
    } catch (_) { /* ignore */ }
  }

  if (!facts.length && !palaceBit) return '';

  const lines = [];
  lines.push('【核对事实（以此为准，禁止张冠李戴）】');
  lines.push('他亲口说过的要点（有记录就承认；没有就说不确定，禁止编造相反偏好）：');
  if (facts.length) {
    for (const f of facts) lines.push(`- ${f}`);
  } else {
    lines.push('- （实录里暂无清晰偏好句，以宫殿摘录为准）');
  }
  if (palaceBit) {
    lines.push('宫殿相关摘录：');
    lines.push(palaceBit);
  }
  lines.push('若他用反问试探（例如「我是不是爱喝咖啡」），以实录为准纠正，不要顺着错记点头。');
  return lines.join('\n');
}

module.exports = {
  needsFactRecall,
  collectPinnedFacts,
  buildRecallFactAnchor,
};
