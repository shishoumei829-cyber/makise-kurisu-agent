'use strict';

/**
 * 阀门 drop / 管道清空后：先抠脏稿里的口语尾巴，再约束重生，避免空白气泡。
 */

const { gateAssistantReply, looksLikeCotLeak } = require('./generationGate');

function extractSpeechCandidate(dirty) {
  const raw = String(dirty || '').trim();
  if (!raw) return '';

  const chunks = raw
    .split(/\n+/)
    .flatMap((line) => line.split(/(?<=[。！？!?])/))
    .map((s) => s.trim())
    .filter((s) => s.length >= 2 && s.length <= 80);

  for (let i = chunks.length - 1; i >= 0; i--) {
    const c = chunks[i]
      .replace(/^[（(][^）)]*[）)]\s*/g, '')
      .replace(/[（(][^）)]*[）)]/g, '')
      .trim();
    if (!c || looksLikeCotLeak(c)) continue;
    if (/用户|对话历史|角色设定|数据库|首先需要|当前查询/.test(c)) continue;
    const g = gateAssistantReply(c);
    if (g.action === 'drop') continue;
    return g.text || c;
  }
  return '';
}

async function salvageAssistantReply(deps, {
  model,
  userText,
  factAnchor = '',
  reasons = [],
  previousDraft = '',
} = {}) {
  const fromDirty = extractSpeechCandidate(previousDraft);
  if (fromDirty) return fromDirty;

  const chatOnce = deps.ollamaChatOnce;
  if (typeof chatOnce !== 'function') return '';
  const u = String(userText || '').trim().slice(0, 280);
  if (!u) return '';

  const system = [
    '你是牧濑红莉栖。只回一句中文口语，像给熟人发消息。',
    '对话对象是冈部伦太郎（凤凰院凶真），男的，你很熟。他说凶真/冈部是在说他自己，别把凶真说成第三人称「她」。',
    '禁止分析、禁止「用户/对话历史/角色设定/作为AI/人工智能/根据你提供的信息」、禁止括号旁白。',
    reasons?.length ? `上一稿被拦：${reasons.slice(0, 3).join('+')}。` : '',
    factAnchor ? String(factAnchor).slice(0, 500) : '',
  ].filter(Boolean).join('\n');

  try {
    const raw = await chatOnce(
      model,
      [
        { role: 'system', content: system },
        { role: 'user', content: u },
      ],
      { temperature: 0.4, num_predict: 64, num_ctx: 1536 },
    );
    let text = String(raw || '').trim();
    if (typeof deps.stripModelThinkingAll === 'function') {
      text = deps.stripModelThinkingAll(text);
    }
    text = text.replace(/<[^>]+>/g, '').trim();
    const extracted = extractSpeechCandidate(text) || text;
    const gated = gateAssistantReply(extracted);
    if (gated.action === 'drop') return '';
    return gated.text || extracted;
  } catch (e) {
    console.warn('[salvage] skip:', e.message);
    return '';
  }
}

module.exports = { salvageAssistantReply, extractSpeechCandidate };
