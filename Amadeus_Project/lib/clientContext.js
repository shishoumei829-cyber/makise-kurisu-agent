'use strict';

/**
 * 前端 situational 上下文 → prompt 块（视觉/冲动/宫殿/情境）。
 * 与 server buildPrompt 单一源配合：前端不再构建 system，只上报结构化片段。
 */

function _clip(s, max) {
  const t = String(s || '').trim().replace(/\s+/g, ' ');
  if (!t) return '';
  return t.length <= max ? t : `${t.slice(0, max)}…`;
}

function normalizeClientContext(raw = {}) {
  if (!raw || typeof raw !== 'object') return {};
  return {
    situation: _clip(raw.situation, 200),
    impulse: _clip(raw.impulse, 160),
    vision: _clip(raw.vision, 320),
    palace: _clip(raw.palace, 1200),
    skipTopic: raw.skipTopic === true,
    wantLongMemory: raw.wantLongMemory === true,
  };
}

function buildClientContextBlock(ctx = {}) {
  const c = normalizeClientContext(ctx);
  const parts = [];

  if (c.situation) {
    parts.push(`【当下情境】${_clip(c.situation, 200)}`);
  }
  if (c.impulse) {
    parts.push(`【念头碎片 · 勿当剧本】${_clip(c.impulse, 160)}`);
  }
  if (c.vision) {
    parts.push(`【视线里 · 仅视觉问题才提】${_clip(c.vision, 320)}`);
  }
  if (c.wantLongMemory && c.palace) {
    parts.push(`【记忆宫殿摘录】\n${_clip(c.palace, 1200)}`);
  }
  if (c.skipTopic) {
    parts.push('【换题意图】她觉得当前话题没意思或聊不下去，想自然换话题或收束，不要硬续上一轮。');
  }

  return parts.join('\n');
}

module.exports = {
  normalizeClientContext,
  buildClientContextBlock,
};
