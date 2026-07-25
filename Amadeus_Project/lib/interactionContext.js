'use strict';
{

function _loadPartnerIdentity() {
  try {
    return require('./partnerIdentity');
  } catch (_) { /* browser */ }
  if (typeof window !== 'undefined' && window.AmadeusPartnerIdentity) {
    return window.AmadeusPartnerIdentity;
  }
  return null;
}
const pi = _loadPartnerIdentity();
const {
  isOkabePartnerMode,
  partnerIsOkabe,
  userClaimsOkabe,
  userAsksAboutOkabe,
  buildPartnerContextBlock,
  replyLooksLikeUnknownOkabe,
  unknownOkabeFallback,
} = pi || {
  isOkabePartnerMode: () => true,
  partnerIsOkabe: () => true,
  userClaimsOkabe: (t) => /我是.{0,6}冈部|我就是冈部|你不知道我是冈部/i.test(String(t || '')),
  userAsksAboutOkabe: (t) => /冈部是谁|你不认识冈部/i.test(String(t || '')),
  buildPartnerContextBlock: () => '',
  replyLooksLikeUnknownOkabe: () => false,
  unknownOkabeFallback: () => '',
};

/**
 * 互动语境：冈部扮演、实验室外号、对 Kurisu 的称呼。
 * 「克里斯蒂娜」= 整段外号（クリスティーナ），不是「蒂娜」简称——见 buildChristinaSemanticsLines。
 */

const NICKNAME_RULES = [
  { id: 'christina', re: /克里斯蒂?娜|クリスティーナ|Christina/i, label: '「克里斯蒂娜」（整段外号）' },
  { id: 'assistant', re: /助手/, label: '「助手」' },
  { id: 'kurisu_name', re: /牧濑|红莉栖|Kurisu|マキセ|まきせ/i, label: '直呼本名' },
  { id: 'perv_combo', re: /天才.{0,6}变态|变态.{0,6}(?:天才|少女)|实验.{0,2}少女|hentai/i, label: '冈部式恶搞组合外号' },
  { id: 'perv', re: /变态/, label: '「变态」类外号' },
  { id: 'baka', re: /笨蛋|バカ|馬鹿/, label: '笨蛋类（常是互怼）' },
  { id: 'lab', re: /凤凰院|凶真|未来道具|Lab\s*mem|ラボ/i, label: '实验室/冈部梗' },
];

/** クリスティーナ / 克里斯蒂娜 的语义（写入 prompt，不是台词表） */
function buildChristinaSemanticsLines() {
  return [
    '「克里斯蒂娜」= 冈部乱起的整段外号（日文クリスティーナ / 英文 Christina），是在叫你牧濑红莉栖，不是第三者的名字。',
    '你讨厌的是被贴上这个滑稽的西洋式称呼本身；反驳应针对整段「克里斯蒂娜」，不是把汉字拆开当成「蒂娜」这个简称。',
    '禁止：「才不是蒂娜」「我不是蒂娜」「别叫我蒂娜」「蒂娜是谁」——那是常见误译/误梗，不是她的反应。',
    '可以的方向：「谁起的克里斯蒂娜」「我才不是什么克里斯蒂娜」「叫牧濑红莉栖」「别用那种称呼」；语气烦、羞、恼，亲密度高可半推半就但不认「蒂娜」。',
  ];
}

function detectNicknamesAtKurisu(text) {
  const t = String(text || '').trim();
  if (!t) return [];
  const hits = [];
  const seen = new Set();
  for (const rule of NICKNAME_RULES) {
    if (!rule.re.test(t)) continue;
    if (rule.id === 'perv' && !/(?:你|妳|喂|啊|哼|又|还|这|那|真是|果然).{0,8}变态|变态.{0,8}(?:啊|呢|吧|哦|喔|吗)|^变态|变态[！!？?]/.test(t)) {
      if (!/天才|少女|实验|hentai/i.test(t)) continue;
    }
    if (seen.has(rule.id)) continue;
    seen.add(rule.id);
    hits.push({ id: rule.id, label: rule.label });
  }
  return hits;
}

function detectOkabeLikely(ctx = {}) {
  if (ctx.partnerIsOkabe === true) return true;
  const name = String(ctx.displayName || ctx.userName || '').trim();
  const t = String(ctx.userText || '').trim();
  const recent = (ctx.recentUserLines || []).join('\n');
  if (/OKABE|冈部|倫太郎|伦太郎|凤凰院|凶真|おかべ|オカベ|Okabe|Hououin/i.test(name)) return true;
  if (userClaimsOkabe(t) || userAsksAboutOkabe(t)) return true;
  if (/冈部|凤凰院|凶真|伦太郎|El\s*Psy|おかべ|オカベ|Okabe|中二|疯狂科学家/i.test(t)) return true;
  if (/助手|克里斯|变态|凤凰院/.test(recent) && /助手|克里斯|变态|凤凰院|冈部/.test(t)) return true;
  return isOkabePartnerMode() && ctx.defaultPartnerOkabe === true;
}

function userMentionedChristina(userText) {
  return /克里斯蒂?娜|クリスティーナ|Christina/i.test(String(userText || ''));
}

/**
 * @param {string} userText
 * @param {object} [ctx]
 */
function analyzeUserTurn(userText, ctx = {}) {
  const t = String(userText || '').trim();
  const nicknames = detectNicknamesAtKurisu(t);
  const okabeLikely = detectOkabeLikely({
    ...ctx,
    userText: t,
    partnerIsOkabe: ctx.partnerIsOkabe === true || partnerIsOkabe(ctx.whoami || {}),
    defaultPartnerOkabe: isOkabePartnerMode(),
  });
  const nicknameOnly = t.length <= 24 && nicknames.length > 0 && !/^(?:为什么|怎么|什么|谁|哪)/.test(t);
  const teasing = /[！!？?~～]/.test(t) || nicknameOnly;
  const christina = nicknames.some((n) => n.id === 'christina');
  return { nicknames, okabeLikely, nicknameOnly, teasing, christina, userText: t };
}

function buildInteractionPromptBlock(analysis) {
  if (!analysis) return '';
  const lines = [];

  if (analysis.okabeLikely) {
    lines.push('对方是冈部伦太郎（你很熟）：未来道具研究所式拌嘴，日常微信口吻，不是陌生人问路。');
  }
  if (analysis.christina) {
    lines.push(...buildChristinaSemanticsLines());
  }
  if (analysis.nicknames.length) {
    lines.push(
      `对方在用叫你（牧濑红莉栖）的外号/梗，不是让你辨认名叫某某的第三者：${analysis.nicknames.map((n) => n.label).join('、')}。`,
    );
    lines.push('禁止：「不认识这个名字」「抱歉我不认识」「您哪位」式应答。应像被起绰号：反驳、吐槽、懒得理、或亲近时半推半就。');
    if (analysis.nicknames.some((n) => n.id === 'perv' || n.id === 'perv_combo')) {
      lines.push('「变态」类外号让你不爽或想怼回去，不是夸奖；禁止「算你有眼光」+撒娇语气、禁止欣然接受变态标签。');
    }
    if (analysis.nicknames.some((n) => n.id === 'perv_combo')) {
      lines.push('「天才变态实验少女」等是冈部胡扯的整蛊：吐槽对方脑子/谁才是变态，不要认真论证外号对不对。');
    }
  }
  if (analysis.teasing && analysis.nicknames.length) {
    lines.push('本轮偏调侃：可短促回击，不必长篇科普或元叙事。');
  }
  lines.push('禁止跳出角色谈「设定」「梗」「只有几个称呼」「治标不治本」等。');

  return lines.length ? `【互动语境】\n${lines.join('\n')}` : '';
}

const CHRISTINA_FALLBACK_POOL = [];

function christinaNicknameFallback(_userText) {
  return '';
}

function nicknameContextActive(userText) {
  return detectNicknamesAtKurisu(userText).length > 0 || userMentionedChristina(userText);
}

/** 仅当用户话里确有外号/恶搞词时用——不再塞固定句 */
function nicknameMisreadFallback(_userText) {
  return '';
}

/** 非外号类误兜底：空，留给模型重试或原文 */
function topicalFallback(_userText) {
  return '';
}

function replyLooksLikeNicknameMisread(userText, reply) {
  const t = String(userText || '');
  const o = String(reply || '');
  if (!detectNicknamesAtKurisu(t).length) return false;
  return /不认识|不认得|没听过|不知道.{0,6}(?:谁|名字)|抱歉.{0,8}不认识|您哪位|哪位是/i.test(o);
}

/**
 * 把「克里斯蒂娜」误拆成「蒂娜」来反驳——非角色反应
 */
function replyLooksLikeChristinaTinaMisparse(userText, reply) {
  if (!userMentionedChristina(userText)) return false;
  const o = String(reply || '');
  if (/才不是蒂娜|不是蒂娜|我不是蒂娜|没叫蒂娜|别叫(?:我)?蒂娜|谁是蒂娜|什么蒂娜|蒂娜是谁|才不是.*蒂娜|蒂娜才不是/i.test(o)) {
    return true;
  }
  if (/喂喂|不是什么.*名字|才不是什么(?!克里斯)/i.test(o) && !/克里斯蒂娜/.test(o)) {
    return true;
  }
  if (/蒂娜/.test(o) && /才不是|我不是|别叫|谁是|不是.*吗|才不是/.test(o) && !/克里斯蒂娜|克里斯|Christina|クリスティーナ/.test(o)) {
    return true;
  }
  return false;
}

function replyLooksLikeMetaOrWrongTsundere(reply) {
  const o = String(reply || '');
  return /治标不治本|几个梗|几个称呼|设定好|扮演|像她的|算你还有点眼光|轻易接受/i.test(o)
    || (/变态/i.test(o) && /眼光|接受|夸奖|~|～/i.test(o));
}

function replyLooksLikeFakeKnowWho(userText, reply) {
  if (!/我是谁|我叫什么|你还记得我是谁/i.test(String(userText || ''))) return false;
  const o = String(reply || '');
  if (/当然知道|我当然|知道你是我|知道你是谁/i.test(o)
    && !/(?:叫|名字是|你是)[^。]{0,16}[\u4e00-\u9fa5A-Za-z]{2,}/.test(o)) {
    return true;
  }
  if (/实验室里最有趣|科学狂人/.test(o)) return true;
  return false;
}

function replyLooksLikeLabAd(reply) {
  return /最酷的实验|无限杯|永远有.*Dr Pepper|来实验室找我啊/i.test(String(reply || ''));
}

/** 克里斯蒂娜：贬成「中二病起的名字」、装不知道谁起的 */
function replyLooksLikeWeakChristinaReply(userText, reply) {
  if (!userMentionedChristina(userText)) return false;
  const o = String(reply || '');
  if (/牧濑红莉栖/.test(o) && (/冈部|谁起的|乱起|我才不是.*克里斯蒂娜/.test(o))) return false;
  if (/中二病才会|像是某个中二|莫名其妙的称呼|难听的外号|起鸡皮疙瘩/.test(o)) return true;
  if (/又是谁起的克里斯蒂娜|那个笨蛋又是谁起的/.test(o)) return true;
  if (/听起来就像是某个/.test(o) && /克里斯蒂娜/.test(o)) return true;
  return false;
}

/** 把自己说成「实验室里的疯子」等 */
function replyLooksLikeWrongSelfTalk(userText, reply) {
  const u = String(userText || '');
  if (!/你是谁|你是什么样|说说你|你是怎样/.test(u)) return false;
  return /我就是.*疯子|实验室里的那个.*疯子|整天摆弄未来道具的疯子|不就是实验室里的/i.test(
    String(reply || ''),
  );
}

/** 「那个笨蛋」模板连发、同一外号抱怨复读 */
function replyLooksLikeBakaTemplateSpam(userText, reply) {
  const o = String(reply || '');
  const hits = (o.match(/那个笨蛋/g) || []).length;
  if (hits >= 2) return true;
  if (/那个笨蛋.{0,20}克里斯蒂娜/.test(o) && /那个笨蛋/.test(o.slice(o.indexOf('克里斯蒂娜')))) return true;
  return false;
}

/** 综合：外号/角色误读需重写 */
function replyNeedsNicknameRepair(userText, reply) {
  const t = String(userText || '');
  const o = String(reply || '');
  if (replyLooksLikeUnknownOkabe(t, o)) return true;
  if (replyLooksLikeWeakChristinaReply(t, o)) return true;
  if (replyLooksLikeWrongSelfTalk(t, o)) return true;
  if (replyLooksLikeBakaTemplateSpam(t, o)) return true;
  if (replyLooksLikeChristinaTinaMisparse(t, o)) return true;
  if (nicknameContextActive(t) && (
    replyLooksLikeNicknameMisread(t, o)
    || replyLooksLikeMetaOrWrongTsundere(o)
  )) return true;
  if (replyLooksLikeFakeKnowWho(t, o)) return true;
  if (/无聊|没意思/.test(t) && replyLooksLikeLabAd(o)) return true;
  return false;
}

function nicknameRepairFallback(userText) {
  if (userClaimsOkabe(userText) || userAsksAboutOkabe(userText)) return unknownOkabeFallback(userText);
  if (userMentionedChristina(userText)) return christinaNicknameFallback(userText);
  if (nicknameContextActive(userText)) return nicknameMisreadFallback(userText);
  return topicalFallback(userText);
}

/**
 * @param {string} userText
 * @param {object} [whoami]
 */
function buildIdentityPromptBlock(userText, whoami = {}) {
  const t = String(userText || '');
  const partnerBlock = buildPartnerContextBlock(whoami, t);
  const name = whoami && whoami.name && whoami.name !== '未知' ? String(whoami.name) : '';
  const extra = [];

  if (/我是谁|我叫什么|你还记得我是谁|你知道我是谁/i.test(t)) {
    if (partnerIsOkabe(whoami) || userClaimsOkabe(t)) {
      extra.push(`【身份】对方问自己的名字；他就是冈部伦太郎——自然说出名字，禁止「你没告诉过我」式装陌生。`);
    } else if (name) {
      extra.push(`【身份】对方问自己的名字；档案里他叫「${name}」——若你记得可自然说出，不要空泛「我当然知道」却不报名字。`);
    } else {
      extra.push('【身份】对方问自己的名字；你没有可靠记录时必须直说「你没告诉过我」，禁止假装知道。');
    }
  }
  if (/你是谁/.test(t) && !/我是谁/.test(t)) {
    extra.push('【身份】对方问你是谁——牧濑红莉栖，研究者；简短自介即可，禁止「实验室最有趣的人」等广告腔。');
  }

  const parts = [partnerBlock, ...extra].filter(Boolean);
  return parts.length ? parts.join('\n') : '';
}

const api = {
  NICKNAME_RULES,
  buildChristinaSemanticsLines,
  detectNicknamesAtKurisu,
  detectOkabeLikely,
  userMentionedChristina,
  analyzeUserTurn,
  buildInteractionPromptBlock,
  christinaNicknameFallback,
  nicknameMisreadFallback,
  nicknameContextActive,
  nicknameRepairFallback,
  topicalFallback,
  replyLooksLikeNicknameMisread,
  replyLooksLikeChristinaTinaMisparse,
  replyLooksLikeMetaOrWrongTsundere,
  replyLooksLikeFakeKnowWho,
  replyLooksLikeLabAd,
  replyNeedsNicknameRepair,
  buildIdentityPromptBlock,
  replyLooksLikeUnknownOkabe,
  unknownOkabeFallback,
  replyLooksLikeWeakChristinaReply,
  replyLooksLikeWrongSelfTalk,
  replyLooksLikeBakaTemplateSpam,
};

if (typeof module !== 'undefined' && module.exports) {
  module.exports = api;
}
if (typeof window !== 'undefined') {
  window.AmadeusInteractionContext = api;
}
}
