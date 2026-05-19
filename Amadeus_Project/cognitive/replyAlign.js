'use strict';

/**
 * 对话对齐：本轮焦点、RAG 门控、尾部跑句修剪、相处向轻提示
 * 供 server /chat 使用（浏览器侧有精简镜像逻辑）。
 */

/** @param {string} t */
function extractTokens(t) {
  const s = String(t || '').trim();
  if (!s) return [];
  const han = s.match(/[\u4e00-\u9fff]{2,}/g) || [];
  const alnum = s.match(/[A-Za-z0-9]{3,}/g) || [];
  const merged = [...han, ...alnum.map((x) => x.toLowerCase())];
  const seen = new Set();
  const out = [];
  for (const x of merged) {
    if (seen.has(x)) continue;
    seen.add(x);
    out.push(x);
    if (out.length >= 16) break;
  }
  return out;
}

/**
 * @param {string} userText
 * @param {{ replyingToProactive?: boolean, proactiveAnchor?: string }} [opts]
 * @returns {string} 单行焦点提示（写入 system，不替代用户原话）
 */
function utteranceFocusLine(userText, opts = {}) {
  const t = String(userText || '').trim().replace(/\s+/g, ' ');
  if (!t) return '';
  if (opts.replyingToProactive && opts.proactiveAnchor) {
    const { buildProactiveReplyFocus } = require('./turnContinuity');
    return buildProactiveReplyFocus(t, opts.proactiveAnchor);
  }
  if (/什么模型|什么意思|做什么的|有什么用|最终目的|哪个实验|什么实验|哪个模型|具体.*什么|收敛|线性回归|你刚才|之前说过|提到过/.test(t)) {
    return `【本轮焦点】对方在追问你先前说法的具体所指——禁止编造「我们聊过」「你提过」的课题；若你只是随口让对方来帮忙，应承认或说清楚范围；举例贴近神经科学/认知与实验室，禁止硬套无关机器学习推销话术；回答必须是完整句子。`;
  }
  if (/[？?]|吗$|么$|呢$|什么|怎么|为什么|为啥|如何|是不是|要不要|能不能|可以吗/i.test(t)) {
    return `【本轮焦点】对方主要在提问或确认——先正面回答命题，少岔到无关记忆或内心独白。`;
  }
  if (/难受|烦|累|无聊|郁闷|伤心|害怕|焦虑|压力|睡不着|想哭|孤独|寂寞|心情不好/i.test(t)) {
    return `【本轮焦点】对方在倒情绪或喊无聊——短接话即可；禁止硬推销「来实验室」「无限杯Dr Pepper」广告串。`;
  }
  if (/实验室.{0,10}(?:在哪|哪里|哪儿|地址)|未来道具.{0,8}(?:在哪|哪里)/i.test(t)) {
    return `【本轮焦点】问实验室位置——答秋叶原上管电器对面破楼二楼（未来道具研究所）；不要答非所问，不要扯变态外号。`;
  }
  if (/我是谁|我叫什么|你还记得我是谁/i.test(t)) {
    return `【本轮焦点】对方问自己的名字——有档案则说出，没有则说没告诉过你；禁止空喊「我当然知道」却不报名字。`;
  }
  if (/你是谁/.test(t) && !/我是谁/.test(t)) {
    return `【本轮焦点】对方问你是谁——牧濑红莉栖，研究者；简短自介，禁止「实验室最有趣的人」等广告腔。`;
  }
  if (/克里斯蒂?娜|クリスティーナ|Christina/i.test(t)) {
    return '【本轮焦点】冈部在叫你「克里斯蒂娜」整段外号——否定整段称呼、要求叫牧濑红莉栖，点出是冈部乱起的；禁止说像中二病起的名字、禁止「又是谁起的」装不认识；禁止「才不是蒂娜」。';
  }
  if (/助手/.test(t) && /叫|喊|称/.test(t)) {
    return '【本轮焦点】「助手」是在叫你——反驳外号，不是认助手这个身份。';
  }
  if (/变态|hentai/i.test(t) && /叫|喊|称/.test(t)) {
    return '【本轮焦点】变态类外号——怼回去，别当夸奖；对冈部用熟人语气，别像对陌生人。';
  }
  if (/你好|嗨|在吗|干嘛呢|做什么|吃了吗|早|晚安|睡了|起了/i.test(t)) {
    return `【本轮焦点】寒暄/近况——接话自然简短，别突然切入沉重或无关实验话题。`;
  }
  if (/谢谢|感谢|辛苦了|对不起|抱歉|我错了/i.test(t)) {
    return `【本轮焦点】致谢或道歉——先回应这份态度，别用反问把气氛顶回去。`;
  }
  if (t.length <= 10 && !/[。！？!?]/.test(t)) {
    return `【本轮焦点】对方话很短——短句接住；若你们够熟，可多追半句或轻轻问一句，不要只回一个字。`;
  }
  return `【本轮焦点】先紧扣对方本轮字面话题；背景记忆与内心念头只能当佐料，不能盖过主题。`;
}

/**
 * 伪相似度：兼容 cosine(0~1) 与 distance（越大越不像）
 * @param {{ score?: number }} hit
 */
function ragPseudoSimilarity(hit) {
  const s = hit && Number.isFinite(hit.score) ? hit.score : NaN;
  if (!Number.isFinite(s)) return 0.5;
  if (s >= 0 && s <= 1.02) return s;
  return 1 / (1 + Math.max(0, s));
}

/**
 * @param {string[]} userToks
 * @param {string} doc
 */
function lexicalHitRatio(userToks, doc) {
  const d = String(doc || '');
  if (!userToks.length || !d) return 0;
  let hit = 0;
  for (const tok of userToks) {
    if (d.includes(tok)) hit++;
  }
  return hit / userToks.length;
}

/**
 * @param {Array<{ text?: string, score?: number, source?: string }>} hits
 * @param {string} userText
 * @param {{ minPseudoSim?: number, minLex?: number }} [opts]
 */
const AUTONOMY_SCIENCE_RE = /量子|睡眠剥夺|神经认知|咖啡因.{0,12}剥夺|论文|假说|世界线|时间机器|SERN|实验数据|研究.{0,6}影响/;

/**
 * 主动轮 RAG：保留口吻相关片段，剔除用户未提过的「科学讲义」类命中
 * @param {Array<{ text?: string }>} hits
 * @param {string} userAnchor
 */
function filterAutonomyRagHits(hits, userAnchor = '') {
  const u = String(userAnchor || '');
  const allowScience = AUTONOMY_SCIENCE_RE.test(u);
  if (allowScience) return hits || [];
  return (hits || []).filter((h) => {
    const text = String(h.text || '');
    if (!AUTONOMY_SCIENCE_RE.test(text)) return true;
    return lexicalHitRatio(extractTokens(u), text) >= 0.2;
  });
}

/**
 * @param {string} memCtx
 * @param {string} userAnchor
 */
function filterAutonomyMemCtx(memCtx, userAnchor = '') {
  const u = String(userAnchor || '');
  if (AUTONOMY_SCIENCE_RE.test(u)) return memCtx;
  const lines = String(memCtx || '').split('\n');
  return lines
    .filter((line) => {
      if (!AUTONOMY_SCIENCE_RE.test(line)) return true;
      return line.length < 36;
    })
    .join('\n')
    .trim();
}

function filterRagHits(hits, userText, opts = {}) {
  const minPseudo = opts.minPseudoSim != null ? opts.minPseudoSim : 0.34;
  const minLex = opts.minLex != null ? opts.minLex : 0.11;
  const userToks = extractTokens(userText);
  const list = Array.isArray(hits) ? hits : [];
  if (!list.length) return [];
  if (!userToks.length) return list.slice(0, 2);

  return list.filter((h) => {
    const text = String(h.text || '');
    const ps = ragPseudoSimilarity(h);
    const lex = lexicalHitRatio(userToks, text);
    if (lex >= minLex) return true;
    if (ps >= minPseudo && lex >= 0.04) return true;
    return false;
  });
}

/**
 * @param {object} userModelInst UserModel 实例
 * @param {string} userText
 * @param {number} relScore
 */
function buildEngagementHint(userModelInst, userText, relScore) {
  if (!userModelInst || relScore < 0.22) return '';
  const topics = userModelInst.model && userModelInst.model.preferences
    ? userModelInst.model.preferences.topics
    : {};
  const top = Object.entries(topics || {})
    .sort((a, b) => b[1] - a[1])
    .slice(0, 2)
    .map(([k]) => k)
    .filter(Boolean);
  if (!top.length) return '';
  const u = String(userText || '');
  const already = top.some((k) => u.includes(k));
  if (already) return '';
  return `【相处学习】他平时更容易接住的话题：${top.join('、')}。仅当与本轮焦点自然相容时，可用半句轻轻勾一下，禁止硬转或复读人设金句。`;
}

/** 去掉全角/半角括号内的动作旁白（挑眉、叹气等） */
function stripRoleplayActions(text) {
  let t = String(text || '');
  t = t.replace(/（[^）\n]{0,80}）/g, '');
  t = t.replace(/\([^)\n]{0,80}\)/g, '');
  t = t.replace(/(?:沉默|停顿|叹气|挑眉|转头|耸肩|轻笑|冷笑|苦笑|注视|凝视)/g, '');
  return t.replace(/\s{2,}/g, ' ').trim();
}

function hasRoleplayActions(text) {
  return /（[^）\n]{1,80}）|\([^)\n]{1,80}\)/.test(String(text || ''));
}

function isGenericFillerLine(text) {
  const t = String(text || '').trim();
  return /^(?:说重点|怎么了|有事？|听着呢|讲。)$/.test(t);
}

/**
 * 去掉与本轮用户/前文明显不接的最后一句（常见「硬模仿尾巴」）
 * @param {string} reply
 * @param {string} userText
 * @param {string} [recentCorpus] 最近对话里用户侧文本拼串
 */
function stripOrphanClosingSentence(reply, userText, recentCorpus = '') {
  const raw = String(reply || '').trim();
  if (!raw) return raw;

  let jpBlock = '';
  let body = raw;
  const jpIdx = raw.search(/\n\s*JP\s*[:：]/i);
  if (jpIdx >= 0) {
    body = raw.slice(0, jpIdx).trim();
    jpBlock = raw.slice(jpIdx);
  }

  body = stripRoleplayActions(body);

  const anchor = `${String(userText || '')}\n${String(recentCorpus || '')}`;
  const anchorToks = extractTokens(anchor);
  if (body.length < 10) return (body + jpBlock).trim() || raw;

  const splitKeepDelim = (s) => {
    const out = [];
    let buf = '';
    for (let i = 0; i < s.length; i++) {
      const ch = s[i];
      buf += ch;
      if (/[。！？!?]/.test(ch)) {
        out.push(buf.trim());
        buf = '';
      }
    }
    if (buf.trim()) out.push(buf.trim());
    return out.filter(Boolean);
  };

  const sents = splitKeepDelim(body);
  if (sents.length < 2) return raw;

  const last = sents[sents.length - 1];
  const rest = sents.slice(0, -1).join('');
  if (last.length > 52) return raw;

  const overlapLastUser = lexicalHitRatio(anchorToks, last);
  const overlapLastBody = lexicalHitRatio(extractTokens(rest), last);
  const templateTail = /^(?:话说回来|说起来|对了|顺便(?:问|说)一下|所以说|你知道(?:吗)?|不过呢|其实吧)/.test(last);

  const orphanWeak =
    overlapLastUser < 0.12
    && overlapLastBody < 0.18
    && last.length <= 48;

  const orphanTemplate = templateTail && overlapLastUser < 0.18 && last.length <= 44;

  if ((orphanWeak || orphanTemplate) && rest.replace(/[^\u4e00-\u9fff\w]/g, '').length >= 4) {
    const newBody = stripRoleplayActions(sents.slice(0, -1).join('').trim());
    const lastClean = stripRoleplayActions(last);
    // 若删掉末句后只剩碎片/旁白，保留末句或合并为更完整的一句
    if (!newBody || newBody.length < 4 || hasRoleplayActions(newBody) || isGenericFillerLine(newBody)) {
      const merged = stripRoleplayActions([newBody, lastClean].filter(Boolean).join(''));
      if (merged.length >= 4) return (merged + jpBlock).trim();
      return (lastClean || newBody || body) + jpBlock;
    }
    return (newBody + jpBlock).trim();
  }

  return (body + jpBlock).trim();
}

/**
 * 定稿是否应覆盖流式：避免用更短的旁白碎片或通用兜底替换完整流式
 */
function shouldReplaceStreamText(streamed, final) {
  const s = stripRoleplayActions(String(streamed || '').trim());
  const f = stripRoleplayActions(String(final || '').trim());
  if (!f) return false;
  if (!s || s === f) return false;
  if (hasRoleplayActions(s) && !hasRoleplayActions(f)) return true;
  if (isGenericFillerLine(f) && s.length > 12 && !isGenericFillerLine(s)) return false;
  if (f.length < s.length * 0.45 && s.length > 16) return false;
  if (hasRoleplayActions(f) && !hasRoleplayActions(s)) return false;
  return true;
}

/**
 * 去掉模型偶发的 Markdown 标记（**粗体**、井号标题等），保留 JP: 行
 * @param {string} reply
 */
function stripChatMarkdown(reply) {
  const raw = String(reply || '');
  const jpIdx = raw.search(/\n\s*JP\s*[:：]/i);
  let body = raw;
  let jpBlock = '';
  if (jpIdx >= 0) {
    body = raw.slice(0, jpIdx).trimEnd();
    jpBlock = raw.slice(jpIdx);
  }
  let t = body;
  t = t.replace(/\*\*([^*]*)\*\*/g, '$1');
  t = t.replace(/\*\*/g, '');
  t = t.replace(/^#{1,6}\s*/gm, '');
  t = t.replace(/^[\-*•]\s+/gm, '');
  t = t.replace(/`([^`]+)`/g, '$1');
  return (t.trim() + jpBlock).trim();
}

module.exports = {
  extractTokens,
  utteranceFocusLine,
  filterRagHits,
  filterAutonomyRagHits,
  filterAutonomyMemCtx,
  buildEngagementHint,
  stripRoleplayActions,
  hasRoleplayActions,
  isGenericFillerLine,
  stripOrphanClosingSentence,
  shouldReplaceStreamText,
  stripChatMarkdown,
  lexicalHitRatio,
};
