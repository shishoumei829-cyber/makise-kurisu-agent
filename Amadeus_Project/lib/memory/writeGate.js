'use strict';

/**
 * 记忆写入闸门：决定本轮能否晋升进宫殿 / 身份 / 事件。
 * 宫殿是长期记忆中枢；毒句只是拒绝原因之一。
 */

const { isDialoguePoison, isInternalControlLeak, admitDialogueText } = require('../unifiedDialogueLog');

const EXPLICIT_PROFILE = /(?:我(?:很|最|一直|通常|平时){0,2}(?:喜欢|讨厌|不喜欢|偏爱|习惯|从不|总是)|我的(?:名字|工作|职业|生日|家乡|专业|爱好)|我叫|请记住|记住我|以后叫我)/;
const SUBSTANTIVE = /(?:我觉得|我认为|我发现|我最近|其实我|因为|所以|但是|不过|为什么|怎么回事|本质|原理|如果|假如|担心|害怕|难过|焦虑|计划|打算|决定|记得|约定|明天|后天|闹钟|提醒|好困|想睡|睡觉|累了|好饿|开心|无聊|论文|实验|量子)/;
const CASUAL = /^(?:嗯+|哦+|喔+|哼+|哈+|额+|啊+|在吗|在|好|行|随便|没事|知道了|收到|ok|OK|嗯嗯|呵呵|enn)[。！？.!?～~]*$/i;
const LIGHT_CHAT = /^(?:好无聊|无聊啊?|有点无聊|没事了|还好|还行|累了|困了|好困|哈哈哈?|呵呵)[啊呀吧呢嘛]?[。！？!?～~]*$/;
const META = /刚才那句不算|我重新说|对话模式|我们这边|作为(?:一个)?AI|语言模型|智能助手|信息压缩器|矛盾检测器|NO_CONFLICT|压[缩縮]为/;

const ROOMS = Object.freeze(['hall', 'lab', 'cafe', 'forbidden']);

function clip(s, n = 80) {
  const t = String(s || '').replace(/\s+/g, ' ').trim();
  if (!t) return '';
  return t.length <= n ? t : `${t.slice(0, n)}…`;
}

function classifyRoom(text) {
  const t = String(text || '');
  if (/爱|喜欢你|在意你|时间线|命运|牺牲|记得你|思念/.test(t)) return 'forbidden';
  if (/科学|量子|神经|论文|实验|研究|物理|数学|编程|算法|闹钟|提醒|设定/.test(t)) return 'lab';
  if (/今天|睡觉|吃|喝|累|无聊|开心|难过|天气|玩|游戏|饿|困|好困/.test(t)) return 'cafe';
  return 'hall';
}

function isCasualUtterance(text) {
  const t = String(text || '').trim();
  if (!t) return true;
  if (CASUAL.test(t) || LIGHT_CHAT.test(t)) return true;
  const compact = t.replace(/\s+/g, '');
  return compact.length > 0 && compact.length <= 2;
}

function assistantRejectReason(assistantText, opts = {}) {
  const a = String(assistantText || '').trim();
  if (!a) return 'empty_assistant';
  if (isInternalControlLeak(a)) return 'poison_or_internal';
  const gated = admitDialogueText(a, {
    role: 'assistant',
    autonomy: opts.proactive === true,
    proactive: opts.proactive === true,
  });
  if (!gated.ok || gated.action === 'drop') {
    const ids = gated.reasons || [];
    if (ids.includes('identity_collapse') || ids.includes('physical_effector') || ids.includes('cot_leak')) {
      return 'identity_collapse';
    }
    if (ids.includes('product_meta')) return 'meta_pollution';
    return 'poison_or_internal';
  }
  if (META.test(a)) return 'meta_pollution';
  // 兼容：旧标签式 META 已覆盖大部分；结构闸仍是主判
  if (isDialoguePoison(a, { autonomy: opts.proactive === true })) return 'poison_or_internal';
  return '';
}

/**
 * @param {object} input
 * @param {string} input.userText
 * @param {string} input.assistantText
 * @param {boolean} [input.proactive]
 * @param {boolean} [input.userRepliedToProactive] 主动开口后他是否已接话
 * @param {object} [input.userAdmission] memoryAdmission.assessUserText 结果
 */
function assessArchive(input = {}) {
  const userText = String(input.userText || '').trim();
  const assistantText = String(input.assistantText || '').trim();
  const proactive = input.proactive === true;
  const userReplied = input.userRepliedToProactive === true;
  const admission = input.userAdmission || null;

  if (proactive && !userReplied) {
    return {
      admit: false,
      action: 'reject',
      reason: 'proactive_unanswered',
      allowPalace: false,
      allowProfile: false,
      allowEvent: false,
      room: null,
    };
  }

  if (isInternalControlLeak(userText) || META.test(userText)) {
    return {
      admit: false,
      action: 'reject',
      reason: 'user_internal_or_meta',
      allowPalace: false,
      allowProfile: false,
      allowEvent: false,
      room: null,
    };
  }

  const asstBad = assistantRejectReason(assistantText, { proactive });
  if (asstBad) {
    return {
      admit: false,
      action: 'reject',
      reason: asstBad,
      allowPalace: false,
      allowProfile: false,
      allowEvent: false,
      room: null,
    };
  }

  if (isCasualUtterance(userText) && !EXPLICIT_PROFILE.test(userText)) {
    return {
      admit: false,
      action: 'working',
      reason: 'short_casual',
      allowPalace: false,
      allowProfile: false,
      allowEvent: false,
      room: null,
    };
  }

  if (admission && admission.tier === 'reject') {
    return {
      admit: false,
      action: 'reject',
      reason: admission.reason || 'admission_reject',
      allowPalace: false,
      allowProfile: false,
      allowEvent: false,
      room: null,
    };
  }

  if (admission && admission.tier === 'working' && admission.reason === 'topic_quarantined') {
    return {
      admit: false,
      action: 'working',
      reason: 'topic_quarantined',
      allowPalace: false,
      allowProfile: false,
      allowEvent: false,
      room: null,
    };
  }

  const explicit = EXPLICIT_PROFILE.test(userText)
    || (admission && admission.allowProfile === true);
  const substantive = SUBSTANTIVE.test(userText)
    || userText.replace(/\s+/g, '').length >= 16
    || (admission && (admission.tier === 'episodic' || admission.tier === 'durable'));

  if (!explicit && !substantive) {
    return {
      admit: false,
      action: 'working',
      reason: 'insufficient_evidence',
      allowPalace: false,
      allowProfile: false,
      allowEvent: false,
      room: null,
    };
  }

  const room = classifyRoom(`${userText} ${assistantText}`);
  return {
    admit: true,
    action: 'archive',
    reason: explicit ? 'explicit_or_profile' : 'substantive_turn',
    allowPalace: true,
    allowProfile: !!explicit,
    allowEvent: !!(admission && admission.allowEvent) || substantive,
    room,
  };
}

/** 短标签（UI / 列表） */
function compressDeterministic(userText, assistantText) {
  const u = clip(userText, 14);
  const a = clip(assistantText, 14);
  if (!u && !a) return '';
  if (!a) return u;
  if (!u) return a;
  return `${u}→${a}`;
}

/**
 * 可检索详情：保留足够事实，供召回命中（不是储物标签）
 */
function buildRecallableDetail(userText, assistantText) {
  const u = clip(userText, 72);
  const a = clip(assistantText, 72);
  if (!u && !a) return '';
  if (!a) return `他提到：${u}`;
  if (!u) return `她说：${a}`;
  return `他：${u}｜她：${a}`;
}

function buildTopic(userText, assistantText) {
  const { extractKeywords } = require('./palaceRetrieve');
  const kws = extractKeywords(userText, assistantText);
  return kws.slice(0, 4).join('·') || clip(userText, 16);
}

module.exports = {
  ROOMS,
  classifyRoom,
  isCasualUtterance,
  assistantRejectReason,
  assessArchive,
  compressDeterministic,
  buildRecallableDetail,
  buildTopic,
  EXPLICIT_PROFILE,
};
