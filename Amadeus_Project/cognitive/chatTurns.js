'use strict';

const { _fitPromptToBudget } = require('./prompts');

/**
 * 解析前端发来的 messages，构造与 OpenAI/Ollama chat 对齐的多轮对话。
 * 根因修复：不能把整段会话塞进一条 user，也不能只给模型「一条 user」却不给 assistant 轮次。
 */

function stripJpBlock(text) {
  const s = String(text || '');
  const idx = s.search(/\n\s*JP\s*[:：]/i);
  return (idx >= 0 ? s.slice(0, idx) : s).trim();
}

/**
 * @param {{ messages?: Array<{ role?: string, content?: string }>, userMsg?: string, message?: string }} body
 */
function parseIncomingChat(body) {
  const raw = Array.isArray(body && body.messages) ? body.messages : [];
  const systemChunks = [];
  const dialogue = [];

  for (const m of raw) {
    if (!m || typeof m.content !== 'string') continue;
    const role = m.role;
    const c = m.content;
    if (role === 'system') {
      const t = c.trim();
      if (t) systemChunks.push(t);
      continue;
    }
    if (role === 'user') {
      const t = c.trim();
      if (t) dialogue.push({ role: 'user', content: t });
    } else if (role === 'assistant') {
      const t = stripJpBlock(c).trim();
      if (t) dialogue.push({ role: 'assistant', content: t });
    }
  }

  const clientSystem = systemChunks.join('\n\n').trim();
  let lastUser = '';
  for (let i = dialogue.length - 1; i >= 0; i--) {
    if (dialogue[i].role === 'user') {
      lastUser = dialogue[i].content;
      break;
    }
  }

  const fallbackUser = String(body.userMsg || body.message || '').trim();
  if (!lastUser && fallbackUser) {
    lastUser = fallbackUser;
    dialogue.push({ role: 'user', content: fallbackUser });
  }

  const userLines = dialogue.filter((x) => x.role === 'user').map((x) => x.content);

  return { clientSystem, dialogue, lastUser, userLines };
}

function capDialogue(dialogue, maxMsgs) {
  const cap = Number(maxMsgs);
  const n = Number.isFinite(cap) && cap >= 2 ? Math.floor(cap) : 24;
  if (dialogue.length <= n) return dialogue.slice();
  return dialogue.slice(dialogue.length - n);
}

function estimateMessageChars(messages) {
  if (!Array.isArray(messages)) return 0;
  return messages.reduce((s, m) => s + String(m?.content || '').length, 0);
}

/**
 * system + 多轮 user/assistant，超长时从最早轮次裁切（保留至少 minTurns 条）
 * @param {string} systemPrompt
 * @param {Array<{ role: string, content: string }>} dialogue
 * @param {number} maxChars
 * @param {number} [minTurns]
 */
function buildOllamaMessages(systemPrompt, dialogue, maxChars, minTurns = 4) {
  const sysContent = String(systemPrompt || '').trim();
  let turns = (Array.isArray(dialogue) ? dialogue : []).filter(
    (m) => m && (m.role === 'user' || m.role === 'assistant') && String(m.content || '').trim()
  );
  const floor = Math.max(2, Math.min(minTurns, turns.length));

  while (turns.length > floor) {
    const messages = [{ role: 'system', content: sysContent }, ...turns];
    if (estimateMessageChars(messages) <= maxChars) break;
    turns = turns.slice(2);
  }

  return [{ role: 'system', content: sysContent }, ...turns];
}

/** 为多轮对话预留字符后，压缩 system 侧人格 prompt */
function fitSystemForDialogue(systemPrompt, dialogue, maxChars) {
  const budget = Math.max(1200, Number(maxChars) || 6000);
  const turns = Array.isArray(dialogue) ? dialogue : [];
  const histEst = turns.reduce((s, m) => s + String(m?.content || '').length, 0);
  const reserve = Math.min(Math.max(400, Math.floor(budget * 0.42)), histEst + 280);
  const sysCap = Math.max(900, budget - reserve);
  return _fitPromptToBudget(systemPrompt, '', sysCap);
}

module.exports = {
  stripJpBlock,
  parseIncomingChat,
  capDialogue,
  estimateMessageChars,
  buildOllamaMessages,
  fitSystemForDialogue,
};
