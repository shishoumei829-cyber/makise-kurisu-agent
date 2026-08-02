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
 * 按 Ollama num_ctx 反推安全字符预算（中日文 prompt 约 1.45 字/token）。
 * 避免 AMADEUS_MAX_PROMPT_CHARS 过大导致 exceed_context_size。
 */
function resolvePromptCharBudget({ numCtx, maxTok, envMaxChars }) {
  const ctx = Math.max(512, Number(numCtx) || 2048);
  const predict = Math.max(64, Number(maxTok) || 384);
  const configured = Number(envMaxChars);
  const envMax = Number.isFinite(configured) && configured > 0 ? configured : 8000;
  const reserve = Math.max(96, Math.min(256, Math.floor(ctx * 0.05)));
  const promptTokens = Math.max(384, ctx - predict - reserve);
  const cptEnv = Number(process.env.AMADEUS_CHARS_PER_TOKEN);
  const charsPerToken = Number.isFinite(cptEnv) && cptEnv > 0 ? cptEnv : 1.45;
  const ctxCap = Math.floor(promptTokens * charsPerToken);
  return Math.min(envMax, Math.max(900, ctxCap));
}

function calculatePromptCharBudget(numCtx, maxTokens, configuredCap = 8000) {
  return resolvePromptCharBudget({ numCtx, maxTok: maxTokens, envMaxChars: configuredCap });
}

/**
 * A short social acknowledgement should not pay the latency of the full
 * research/task prompt. This is deliberately conservative: anything that
 * looks like a request, analysis, or multi-part question stays on the full
 * path so capability and reasoning quality are never traded for speed.
 */
function isFastConversationTurn(text, options = {}) {
  if (options.autonomy || options.useLongTermMemory || options.hasTask) return false;
  const value = String(text || '').trim();
  if (value.length < 1 || value.length > 42 || /[\r\n]/.test(value)) return false;
  const needsFullContext = /(?:为什么|為什麼|怎么|怎麼|如何|帮我|幫我|提醒|创建|創建|删除|刪除|打开|打開|搜索|搜尋|查一下|计划|計劃|任务|任務|文件|邮件|郵件|日历|日曆|代码|代碼|bug|debug|分析|比较|比較|计算|計算|证明|證明|方案|设计|設計|research|search|create|delete|open|remind|plan|code|debug)/i;
  if (needsFullContext.test(value)) return false;
  // Long compound questions need continuity even when they do not contain a
  // keyword above. Simple greetings, thanks, reactions and short follow-ups do not.
  return (value.match(/[？?]/g) || []).length <= 1;
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
  // 短社交轮也要保住人格锚点；旧的 42% 固定预留会在 2600 字预算下
  // 把 system 压到约 1500 字，日语微调模型只看到半截人格而退回客服腔。
  const reserve = Math.min(Math.max(300, Math.floor(budget * 0.24)), histEst + 280);
  const sysCap = Math.max(900, budget - reserve);
  return _fitPromptToBudget(systemPrompt, '', sysCap);
}

module.exports = {
  stripJpBlock,
  parseIncomingChat,
  capDialogue,
  estimateMessageChars,
  resolvePromptCharBudget,
  calculatePromptCharBudget,
  isFastConversationTurn,
  buildOllamaMessages,
  fitSystemForDialogue,
};
