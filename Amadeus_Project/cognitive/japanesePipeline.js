'use strict';

/**
 * 日语阶段校验：说话对象是否正确、是否引用实录不存在的内容。
 * 通过后再译中文（或由调用方决定是否修复中文草稿）。
 */

function extractFactClaims(jpText) {
  const t = String(jpText || '');
  const claims = [];
  const patterns = [
    /(?:昨日|昨天|今朝|今天|今夜|昨晚|午前|午後|午後|先ほど|さっき|之前|刚才).{0,24}/g,
    /(?:言った|言ってた|说过|提到过|聊过|約束|約束した|答应)/g,
    /(?:実験|論文|データ|实验|论文|数据).{0,20}/g,
  ];
  for (const re of patterns) {
    const m = t.match(re);
    if (m) claims.push(...m.slice(0, 3));
  }
  return [...new Set(claims)].slice(0, 8);
}

function checkAddressee(jpText, partnerName = '') {
  const t = String(jpText || '');
  const issues = [];
  if (/あなたは誰|你是谁|誰ですか|どなた/.test(t) && partnerName) {
    issues.push('对熟人不应问「你是谁」');
  }
  if (/岡部さんは誰|岡部是谁|凤凰院是谁/.test(t)) {
    issues.push('不应问冈部是谁');
  }
  if (/AI|アシスタント|助手|程序|チャットボット/.test(t)) {
    issues.push('不应自称或暗示 AI/助手');
  }
  return issues;
}

function checkLogConsistency(jpText, conversationLog = '') {
  const log = String(conversationLog || '');
  const issues = [];
  const t = String(jpText || '');

  const memoryDeny = /覚えてない|没有说过|没说过|記録がない|ないはず/.test(t);
  const memoryAssert = /言ったよね|说过吧|約束した|聊过|提到过|记得/.test(t);

  if (memoryAssert && log.length < 40) {
    issues.push('实录为空却断言「说过/聊过」');
  }

  if (memoryDeny && /\[.*\].*(他|Kurisu):/.test(log)) {
    issues.push('实录有记录却否认记得');
  }

  const claims = extractFactClaims(t);
  for (const c of claims) {
    const core = c.replace(/[^\u4e00-\u9fa5A-Za-z0-9]/g, '').slice(0, 6);
    if (core.length >= 3 && memoryAssert && log && !log.includes(core.slice(0, 4))) {
      issues.push(`断言「${c}」但实录未找到对应片段`);
    }
  }

  return issues;
}

function validateJapaneseLine(jpText, ctx = {}) {
  const issues = [
    ...checkAddressee(jpText, ctx.partnerName),
    ...checkLogConsistency(jpText, ctx.conversationLog),
  ];
  return {
    ok: issues.length === 0,
    issues,
    jp: String(jpText || '').trim(),
  };
}

/**
 * 构建 LLM 自检 prompt（同一模型二次调用）
 */
function buildSelfCheckPrompt(jpText, conversationLog) {
  return {
    system: '你是台词质检员。只输出 JSON：{"ok":true} 或 {"ok":false,"issues":["..."]}。不要解释。',
    user: `日语台词：
${jpText}

对话实录摘录：
${String(conversationLog || '').slice(-1200)}

只验两件事：
1) 说话对象是否正确（对熟人别问「你是谁」、别称 AI）
2) 是否引用了实录里不存在的内容（别说「我们聊过X」若实录无X）

输出 JSON。`,
  };
}

function parseSelfCheckJson(raw) {
  const t = String(raw || '').trim();
  const m = t.match(/\{[\s\S]*\}/);
  if (!m) return null;
  try {
    return JSON.parse(m[0]);
  } catch {
    return null;
  }
}

/**
 * 完整管道：中文草稿 → 日语 → 规则+可选LLM自检 → 必要时修复 → 最终中文
 */
async function runJapaneseValidationPipeline(ctx = {}) {
  const {
    chineseDraft,
    conversationLog,
    partnerName,
    translateToJapanese,
    translateToChinese,
    llmSelfCheck,
    repairChinese,
  } = ctx;

  let cn = String(chineseDraft || '').trim();
  if (!cn) return { ok: false, chinese: '', japanese: '', issues: ['空回复'] };

  let jp = '';
  if (typeof translateToJapanese === 'function') {
    jp = await translateToJapanese(cn);
  }
  if (!jp) {
    return { ok: true, chinese: cn, japanese: '', issues: [], skipped: 'no_jp' };
  }

  let validation = validateJapaneseLine(jp, { conversationLog, partnerName });

  if (validation.ok && typeof llmSelfCheck === 'function') {
    const llmResult = await llmSelfCheck(jp, conversationLog);
    if (llmResult && llmResult.ok === false) {
      validation = { ok: false, issues: llmResult.issues || ['LLM自检未通过'], jp };
    }
  }

  if (!validation.ok && typeof repairChinese === 'function') {
    cn = await repairChinese(cn, validation.issues);
    if (typeof translateToJapanese === 'function') {
      jp = await translateToJapanese(cn);
      validation = validateJapaneseLine(jp, { conversationLog, partnerName });
    }
  }

  if (validation.ok && typeof translateToChinese === 'function' && jp) {
    const cnFromJp = await translateToChinese(jp);
    if (cnFromJp && cnFromJp.length >= Math.min(cn.length, 4)) {
      cn = cnFromJp;
    }
  }

  return {
    ok: validation.ok,
    chinese: cn,
    japanese: jp,
    issues: validation.issues || [],
  };
}

function validateChineseReply(text, conversationLog, partnerName) {
  const issues = [
    ...checkAddressee(text, partnerName),
    ...checkLogConsistency(text, conversationLog),
  ];
  if (/那还能是谁|你是哪位|你是什么模型|作为AI|作为人工智能/.test(String(text || ''))) {
    issues.push('语用错误：对熟人不应问身份');
  }
  return { ok: issues.length === 0, issues };
}

module.exports = {
  validateJapaneseLine,
  validateChineseReply,
  buildSelfCheckPrompt,
  parseSelfCheckJson,
  runJapaneseValidationPipeline,
  checkAddressee,
  checkLogConsistency,
};
