'use strict';

function getReplyLanguageMode() {
  const env = String(process.env.AMADEUS_REPLY_LANGUAGE || '').trim().toLowerCase();
  if (['ja', 'jp', 'japanese'].includes(env)) return 'ja';
  if (['zh', 'cn', 'chinese'].includes(env)) return 'zh';
  const model = String(process.env.AMADEUS_CHAT_MODEL || '').toLowerCase();
  return /kurisu(?:-v\d+|:v\d+)?/.test(model) ? 'ja' : 'zh';
}

function countScriptChars(text) {
  const value = String(text || '');
  return {
    kana: (value.match(/[\u3040-\u30ff\uff65-\uff9f]/g) || []).length,
    han: (value.match(/[\u3400-\u9fff]/g) || []).length,
    latin: (value.match(/[A-Za-z]/g) || []).length,
    cyrillic: (value.match(/[\u0400-\u04ff]/g) || []).length,
    total: value.replace(/\s/g, '').length,
  };
}

function validateJapaneseOutput(text) {
  const value = String(text || '').trim();
  const counts = countScriptChars(value);
  const issues = [];
  if (!value) issues.push('empty');
  if (counts.cyrillic > 0) issues.push('cyrillic');
  if (counts.kana < 1) issues.push('missing-kana');
  if (/(?:作为|我是一个|人工智能|语言模型|无法满足|抱歉，我)/.test(value)) {
    issues.push('chinese-template');
  }
  return { ok: issues.length === 0, issues, counts, text: value };
}

function isPrimarilyJapanese(text) {
  const result = validateJapaneseOutput(text);
  if (!result.ok) return false;
  const { kana, han, total } = result.counts;
  return kana >= 2 || (kana >= 1 && han <= Math.max(8, total * 0.7));
}

function stripModelDecorations(text) {
  return String(text || '')
    .replace(/<redacted_thinking>[\s\S]*?<\/redacted_thinking>/gi, '')
    .replace(/<think>[\s\S]*?<\/think>/gi, '')
    .replace(/<(?:redacted_thinking|think)>[\s\S]*$/gi, '')
    .replace(/^(?:JP|日本語|日文)\s*[:：]\s*/gim, '')
    .replace(/^(?:CN|中文)\s*[:：][\s\S]*/gim, '')
    .trim();
}

/**
 * 剥掉模型误把「意识广播」清单复述进对白的痕迹。
 * 例如：[打算] 我会先找话题——然后就闲聊。
 */
function stripConsciousnessEcho(text) {
  let t = String(text || '');
  if (!t) return '';
  t = t.replace(/【意识广播[\s\S]*?(?:】|$)/g, '');
  t = t.replace(/^\s*\d+\.\s*\[(?:打算|注意到|感受|想要|自我|关系|想起|自检)\][^\n]*/gm, '');
  t = t.replace(/\[(?:打算|注意到|感受|想要|自我|关系|想起|自检)\]\s*/g, '');
  t = t.replace(/我会先找话题[—\-~～]*然后就闲聊。?/g, '');
  t = t.replace(/本轮由内驱进入意识而开口。?/g, '');
  t = t.replace(/驱动说话但勿向用户复述清单/g, '');
  // 产品/运维元话语：曾被硬编码进 pipeline，模型也会复读
  t = t.replace(/[…\.．]*\s*刚才那句不算[，,]?\s*(?:我)?重新说[。.!！]?/g, '');
  t = t.replace(/刚才那句不算[，,]?/g, '');
  t = t.replace(/(?:那句)?不算[，,]?\s*我重新说[。.!！]?/g, '');
  t = t.replace(/^(?:说重点|怎么了|有事|听着呢|讲)[。.!！?？]?$/g, '');
  t = t.replace(/又是[这那]个话题吗[。.!！?？]?/g, '');
  t = t.replace(/停止指定话题吧[，,]?[^\n]*/g, '');
  return t.replace(/\n{3,}/g, '\n\n').trim();
}

function extractJapaneseBody(text) {
  const raw = stripModelDecorations(text);
  if (!raw) return '';
  const cnIndex = raw.search(/\n?\s*CN\s*[:：]/i);
  const candidate = cnIndex >= 0 ? raw.slice(0, cnIndex).trim() : raw;
  const lines = candidate.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
  const accepted = [];
  for (const line of lines) {
    const cleaned = line.replace(/^(?:JP|日本語|日文)\s*[:：]\s*/i, '');
    if (validateJapaneseOutput(cleaned).ok) {
      accepted.push(cleaned);
    } else if (accepted.length) {
      break;
    }
  }
  const joined = accepted.join('。').replace(/。{2,}/g, '。').trim();
  return validateJapaneseOutput(joined).ok ? joined : '';
}

/**
 * 日→中显示译文：忠实于原文意思，但必须是人能读懂的中文。
 * 禁止剧情脑补；也禁止为「硬翻」产出断句乱码。
 * TTS 仍念日语原文，字幕跟这句译文对齐。
 */
const LITERAL_JP_TO_CN_SYSTEM = [
  '你是日译中翻译器。把下面角色说的话从日语翻成简体中文。',
  '硬性规则：',
  '1. 忠实：不增删事实、不脑补未说的意思、不改成另一套剧情。',
  '2. 可读：译文必须是正常人能一眼看懂的完整中文句子；不要逐词死译成谜语或断句乱码。',
  '3. 语气：保留冷淡/吐槽/傲娇等口气，但用自然中文表达，不要夹日语助词残骸。',
  '4. 原文里已经是中文的片段原样保留。',
  '5. 去掉动作旁白括号（如（歪头）（推眼镜）），只留对白。',
  '6. 只输出译文正文，不要解释、不要 JP:/CN: 前缀、不要用引号包整句。',
  '7. 高频词：もう≠再三；もう？→又？/已经？；まだ→还/还在；もう一度→再一次。',
  '示例：もう？ まだ心配してるの → 又？还在担心吗？',
].join('\n');

function buildLiteralJpToCnMessages(jp) {
  const body = String(jp || '').trim().slice(0, 800);
  return [
    { role: 'system', content: LITERAL_JP_TO_CN_SYSTEM },
    { role: 'user', content: body },
  ];
}

/**
 * 小模型硬翻常把口语助词翻歪（如 もう→再三）。
 * 用日语原文做确定性纠偏，保证界面中文与 TTS 日语同一句。
 */
function alignLiteralCnToJapanese(jp, cn) {
  const src = String(jp || '').trim();
  let out = String(cn || '').trim();
  if (!src || !out) return out;

  // 句首「もう？」绝不是「再三」
  if (/^もう[？?]/.test(src)) {
    out = out.replace(/^(再三|再次|又一次|再三再四)/, '又');
    if (/再三/.test(out)) out = out.replace(/再三/g, '又');
  }
  // 「もう一度」→ 再一次（若模型写成「又一次」可保留；若写成再三则修）
  if (/もう一度/.test(src) && /再三/.test(out)) {
    out = out.replace(/再三/g, '再一次');
  }
  // 「まだ」在担心/害怕类句里应对「还/还在」
  if (/まだ/.test(src) && /心配|心配して/.test(src)) {
    if (/^(不再|不在)担心/.test(out.replace(/\s/g, ''))) {
      // leave alone — unlikely
    } else if (!/还/.test(out) && /担心/.test(out)) {
      out = out.replace(/担心/, '还在担心');
    }
  }
  return out.replace(/\s{2,}/g, ' ').trim();
}

module.exports = {
  getReplyLanguageMode,
  isPrimarilyJapanese,
  extractJapaneseBody,
  stripModelDecorations,
  stripConsciousnessEcho,
  countScriptChars,
  validateJapaneseOutput,
  LITERAL_JP_TO_CN_SYSTEM,
  buildLiteralJpToCnMessages,
  alignLiteralCnToJapanese,
};
