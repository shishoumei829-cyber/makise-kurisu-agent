'use strict';

/**
 * 阀门 drop / 管道清空后：先抠脏稿里的口语尾巴，再约束重生，避免空白气泡。
 */

const { gateAssistantReply, looksLikeCotLeak } = require('./generationGate');

function isStyleCriticism(value) {
  return /客服|套话|客套|AI味|ai味|翻译腔|很假|太假|太规律|反问|模板|机械|僵硬|不像红莉栖|不像牧濑/.test(String(value || ''));
}

function isFixedStyleDefence(value) {
  return /(?:我(?:不会|不想|不是|并非)|不会再|别把我当).{0,26}(?:敷衍|应付|迎合|机械|千篇一律|公式化|空泛|照本宣科|套路|固定(?:的)?(?:回答|回应)|制式(?:的)?(?:回答|回应)|说些空话)/.test(String(value || ''));
}

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

  const repairMode = isStyleCriticism(userText) || isFixedStyleDefence(previousDraft);
  const system = [
    '你是牧濑红莉栖。只回一句中文口语，像给熟人发消息。',
    '对话对象是冈部伦太郎（凤凰院凶真），男的，你很熟。他说凶真/冈部是在说他自己，别把凶真说成第三人称「她」。',
    '禁止分析、禁止站在对话外面说明、禁止括号旁白。',
    repairMode
      ? '对方不满的是你刚刚说的话。只承接这份不满：可以承认那句话不对，但不要解释、保证、辩护，也不要谈论自己怎样回答或是什么；不要复述他用来评价你的词，也不要反问。'
      : '',
    !repairMode && reasons?.length ? '上一稿不能使用。直接回到他这一句的具体内容。' : '',
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

module.exports = { salvageAssistantReply, extractSpeechCandidate, isFixedStyleDefence };
