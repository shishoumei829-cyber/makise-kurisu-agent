'use strict';

/**
 * OOC 检测与兜底：口吻像、少出戏。
 */

function ix() {
  if (typeof window !== 'undefined' && window.AmadeusInteractionContext) {
    return window.AmadeusInteractionContext;
  }
  try {
    // eslint-disable-next-line global-require
    return require('./interactionContext');
  } catch {
    return null;
  }
}

function oocPatterns() {
  return [
    { id: 'ai_identity', re: /作为(?:一个)?AI|我是人工智能|语言模型|大语言模型|ChatGPT|助手程序|智能助手|很高兴为您服务|有什么可以帮您|请问还有什么需要|希望能帮到您/, weight: 3 },
    { id: 'ai_tone', re: /亲爱的用户|尊敬的用户|温馨提示|建议您呢|没问题呢[！!]|好的呢[！!~～]|当然可以呢/, weight: 2 },
    { id: 'lecture', re: /首先[，,].*其次|综上所述|需要注意的是|具体来说[，,]我可以为你解释/, weight: 2 },
    { id: 'meta', re: /角色设定|人设|OOC|出戏|扮演|剧本|台词表|几个梗|治标不治本|像她的/, weight: 3 },
    { id: 'third_person', re: /牧濑红莉栖(?:认为|表示|说道|觉得)|她(?:突然|轻声|冷冷)/, weight: 2 },
    { id: 'roleplay', re: /（[^）]{0,40}）|\([^)]{0,40}\)|挑眉|叹气|转过头|耸肩/, weight: 3 },
    { id: 'over_soft', re: /别难过啦|抱抱|乖[，,]|我会一直陪着你|母亲般|温柔地安慰/, weight: 2 },
    { id: 'perform_tsun', re: /才不是担心你|才不是关心|别误会[^。]{0,12}我只是顺便/, weight: 1 },
  ];
}

function scoreOoc(reply) {
  const o = String(reply || '');
  let score = 0;
  const hits = [];
  for (const p of oocPatterns()) {
    if (p.re.test(o)) {
      score += p.weight;
      hits.push(p.id);
    }
  }
  return { score, hits };
}

function autonomyIx() {
  try {
    return require('../cognitive/turnContinuity');
  } catch {
    return null;
  }
}

function replyNeedsOocRepair(userText, reply, opts = {}) {
  const o = String(reply || '').trim();
  if (!o) return { needs: true, reason: 'empty', category: 'empty' };

  const tc = autonomyIx();
  const anchor = String(opts.userAnchor || userText || '');
  if (
    opts.autonomy
    && tc
    && typeof tc.replyLooksLikeAutonomyFabrication === 'function'
    && tc.replyLooksLikeAutonomyFabrication(anchor, o)
  ) {
    return { needs: true, reason: 'autonomy_fabrication', category: 'autonomy_fabrication' };
  }

  const ic = ix();
  if (ic && typeof ic.replyNeedsNicknameRepair === 'function' && ic.replyNeedsNicknameRepair(userText, o)) {
    return { needs: true, reason: 'nickname', category: 'nickname' };
  }

  const { score, hits } = scoreOoc(o);
  if (score >= 2) return { needs: true, reason: hits.join(','), category: hits[0] || 'ooc' };

  const u = String(userText || '');
  if (o.length > 220 && !/论文|实验|代码|报错|原理|证明|为什么|怎么/.test(u) && /首先|其次|总之|综上/.test(o)) {
    return { needs: true, reason: 'lecture_long', category: 'lecture' };
  }

  return { needs: false, reason: '', category: '' };
}

const FALLBACK_POOL = {
  default: ['说重点。', '怎么了。', '有事？', '听着呢。', '讲。'],
  ai: ['……你把我当成什么客服了？有事直说。', '别来这套，我又不是助手热线。'],
  lecture: ['太长。你到底想问哪一句？', '别上课，讲重点。'],
  meta: ['……你在说什么？有话正常说。'],
};

function oocRepairFallback(userText, category = 'default') {
  const ic = ix();
  if (category === 'nickname' && ic && typeof ic.nicknameRepairFallback === 'function') {
    return ic.nicknameRepairFallback(userText);
  }
  const cat = category === 'ai_identity' ? 'ai' : category;
  const pool = FALLBACK_POOL[cat] || FALLBACK_POOL.default;
  return pool[Math.floor(Math.random() * pool.length)];
}

function sanitizeOocSurface(reply) {
  let o = String(reply || '').trim();
  o = o.replace(/作为(?:一个)?AI[^。！？!?]{0,40}[。！？!?]?/g, '');
  o = o.replace(/很高兴为您服务[^。！？!?]*[。！？!?]?/g, '');
  o = o.replace(/\*[^*]{1,30}\*/g, '');
  return o.replace(/\s+/g, ' ').trim();
}

function repairKurisuReply(userText, reply, opts = {}) {
  let out = String(reply || '').trim();
  if (!out) return oocRepairFallback(userText, 'default');

  const ic = ix();
  if (ic && typeof ic.replyNeedsNicknameRepair === 'function' && ic.replyNeedsNicknameRepair(userText, out)) {
    return ic.nicknameRepairFallback(userText);
  }

  const tc = autonomyIx();
  if (
    opts.autonomy
    && tc
    && typeof tc.replyLooksLikeAutonomyFabrication === 'function'
    && tc.replyLooksLikeAutonomyFabrication(opts.userAnchor || userText, out)
  ) {
    return tc.autonomyFabricationFallback(opts.userAnchor || userText);
  }

  const check = replyNeedsOocRepair(userText, out, opts);
  if (!check.needs) return out;

  const { score } = scoreOoc(out);
  if (score >= 3 || check.category === 'ai_identity') {
    return oocRepairFallback(userText, check.category);
  }

  const cleaned = sanitizeOocSurface(out);
  if (cleaned.length >= 6 && !replyNeedsOocRepair(userText, cleaned).needs) return cleaned;

  return oocRepairFallbackSafe(userText, check.category);
}

function oocRepairFallbackSafe(userText, category) {
  const ic = ix();
  if (ic && typeof ic.nicknameRepairFallback === 'function') {
    return ic.nicknameRepairFallback(userText);
  }
  return oocRepairFallback(userText, category);
}

/**
 * 流式已显示内容与定稿兜底不一致时：若兜底句与本轮话题无关而流式内容更贴题，保留流式（仅做轻清洗）
 */
function extractTopicKeys(userText) {
  const t = String(userText || '');
  const keys = new Set();
  const fixed = t.match(/实验室|秋叶原|未来道具|我是谁|你是谁|无聊|克里斯|助手|变态|冈部|凤凰院/g) || [];
  fixed.forEach((k) => keys.add(k));
  if (/实验室/.test(t)) {
    keys.add('研究所');
    keys.add('未来道具');
  }
  const han = t.match(/[\u4e00-\u9fff]{2,4}/g) || [];
  han.forEach((k) => { if (k.length >= 2) keys.add(k); });
  return [...keys];
}

function reconcileFinalReply(streamed, final, userText) {
  const s = String(streamed || '').trim();
  const f = String(final || '').trim();
  if (!s || !f || s === f) return f;

  const keys = extractTopicKeys(userText);
  const hitS = keys.length ? keys.some((k) => s.includes(k)) : s.length > 6;
  const hitF = keys.length ? keys.some((k) => f.includes(k)) : false;

  const fillerOnly = /^(?:说重点|怎么了|有事|听着呢|讲)[。.!！?？]?$/.test(f.replace(/\s+/g, ''));
  const genericFallback = fillerOnly
    || /变态的是你的脑子|天才倒是没说错|你到底想说什么|有话正常说|我又不是助手热线/.test(f);
  const ic = ix();
  const nickCtx = ic && typeof ic.nicknameContextActive === 'function' && ic.nicknameContextActive(userText);
  if (genericFallback && hitS && !hitF && !nickCtx) {
    const cleaned = sanitizeOocSurface(s);
    const nickOk = !ic || typeof ic.replyNeedsNicknameRepair !== 'function'
      || !ic.replyNeedsNicknameRepair(userText, cleaned);
    if (nickOk && !replyNeedsOocRepair(userText, cleaned).needs) return cleaned;
  }
  if (s.length >= 8 && f.length < s.length * 0.55 && hitS && !nickCtx) {
    const cleaned = sanitizeOocSurface(s);
    if (cleaned.length >= 6 && !replyNeedsOocRepair(userText, cleaned).needs) return cleaned;
  }
  return f;
}

const api = {
  oocPatterns,
  scoreOoc,
  replyNeedsOocRepair,
  sanitizeOocSurface,
  oocRepairFallback,
  repairKurisuReply,
  reconcileFinalReply,
  extractTopicKeys,
};

if (typeof module !== 'undefined' && module.exports) {
  module.exports = api;
}
if (typeof window !== 'undefined') {
  window.AmadeusOocGuard = api;
}
