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
  // 无依据地判断对方外表/状态时，旧稿不能“抽一句还能说的”再放行；
  // 必须丢掉整稿并在原话范围内重生，否则同一个臆测会循环回来。
  const factUnsafe = reasons.includes('invented_user_state');
  const fromDirty = factUnsafe ? '' : extractSpeechCandidate(previousDraft);
  if (fromDirty) return fromDirty;

  const chatOnce = deps.ollamaChatOnce;
  if (typeof chatOnce !== 'function') return '';
  const u = String(userText || '').trim().slice(0, 280);
  if (!u) return '';

  const system = [
    '你是牧濑红莉栖。只回一句中文口语，像给熟人发消息。',
    '对话对象是冈部伦太郎（凤凰院凶真），男的，你很熟。他说凶真/冈部是在说他自己，别把凶真说成第三人称「她」。',
    '禁止分析、禁止站在对话外面说明、禁止括号旁白。',
    factUnsafe
      ? '上一稿猜测了对方没有明确说出的外表或状态，不能沿用。只回应他实际说出的内容；不描述他的脸色、表情、坐姿、疲惫或任何看见的画面。'
      : '',
    reasons?.length ? '上一稿不能使用。直接回到他这一句的具体内容。' : '',
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
