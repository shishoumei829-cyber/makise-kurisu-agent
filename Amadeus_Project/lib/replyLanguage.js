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
  return t.replace(/\n{3,}/g, '\n').trim();
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
 * 日→中硬翻系统提示：禁止通顺化、禁止脑补未说的意思。
 * 原文有多乱，译文就多直；原文里的中文碎片原样保留。
 */
const LITERAL_JP_TO_CN_SYSTEM = [
  '你是逐字翻译器，不是润色器。',
  '把下面角色说的话从日语翻成简体中文。',
  '硬性规则：',
  '1. 只做直译/硬翻，不要改写成通顺对白，不要补全、推断或美化未说出的意思。',
  '2. 原文不通顺、残缺、中日夹杂时，译文必须同等残缺或夹杂；不要圆成一句“合理的话”。',
  '3. 原文里已经是中文的片段原样保留，不要改写。',
  '4. 保留数字、公式、专有名词；不要夹英文解释。',
  '5. 只输出译文正文，不要解释、不要加 JP:/CN: 前缀、不要用引号包整句。',
  '6. 高频词勿乱套：もう≠再三；もう？→又？/已经？；まだ→还/还在；もう一度→再一次。',
  '示例（学对应，勿照抄无关句）：もう？ まだ心配してるの → 又？还在担心吗？',
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
