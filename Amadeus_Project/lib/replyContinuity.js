'use strict';

// 这不是“固定回答库”。它只阻止刚刚已经说过的同一句再次写入实录；
// 语义上的下一句仍由当前的主体状态和模型生成。
function normalizeReply(text) {
  return String(text || '')
    .toLowerCase()
    .replace(/[\s\u3000]/g, '')
    .replace(/[，。！？、；：,.!?;:'"“”‘’…—-]/g, '')
    .trim();
}

function isRecentAssistantDuplicate(reply, entries = []) {
  const candidate = normalizeReply(reply);
  if (candidate.length < 4) return false;
  return entries
    .filter((entry) => entry?.role === 'assistant')
    .slice(-8)
    .some((entry) => normalizeReply(entry.text || entry.content) === candidate);
}

module.exports = { normalizeReply, isRecentAssistantDuplicate };
