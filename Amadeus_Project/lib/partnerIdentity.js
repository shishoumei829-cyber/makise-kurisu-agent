'use strict';

/**
 * 对话对象身份（默认：冈部伦太郎）。AMADEUS_PARTNER_ID=custom 可改为仅依赖 whoami 档案。
 */

const OKABE_PROFILE = {
  id: 'okabe',
  name: '冈部伦太郎',
  aliases: ['凤凰院凶真', '凶真', '冈伦', '冈部', '伦太郎', 'Okabe', '凤凰院'],
  traits: ['中二', '不按常理', '其实很在意同伴'],
  relationship_note:
    '他就是冈部伦太郎——未来道具研究所那个很烦、很中二、但你很熟的人。不是陌生人，禁止装不认识。',
  basics: { role: '未来道具研究所 · 凤凰院凶真' },
};

function getPartnerId() {
  if (typeof window !== 'undefined' && window.__AMADEUS_PARTNER_ID) {
    return String(window.__AMADEUS_PARTNER_ID).trim().toLowerCase();
  }
  const v = process.env.AMADEUS_PARTNER_ID;
  if (v === undefined || v === null || String(v).trim() === '') return 'okabe';
  return String(v).trim().toLowerCase();
}

function isOkabePartnerMode() {
  const id = getPartnerId();
  return id === 'okabe' || id === 'okarin' || id === 'hououin';
}

function shouldBootstrapOkabeWhoami() {
  if (!isOkabePartnerMode()) return false;
  const v = process.env.AMADEUS_PARTNER_BOOTSTRAP;
  if (v === '0' || v === 'false') return false;
  return true;
}

/**
 * @param {object} w
 * @returns {object}
 */
function bootstrapWhoamiRecord(w = {}) {
  if (!shouldBootstrapOkabeWhoami()) return { ...w };
  const out = { ...w };
  if (!out.name || out.name === '未知') out.name = OKABE_PROFILE.name;
  if (!String(out.relationship_note || '').trim()) {
    out.relationship_note = OKABE_PROFILE.relationship_note;
  }
  const traits = new Set([...(out.traits || []), ...OKABE_PROFILE.traits]);
  out.traits = [...traits];
  out.basics = { ...(out.basics || {}), ...OKABE_PROFILE.basics };
  if (!out.basics.role) out.basics.role = OKABE_PROFILE.basics.role;
  out.partner_id = 'okabe';
  out.last_updated = Date.now();
  return out;
}

/**
 * 读盘后确保 whoami 含冈部档案（有变更则写回）
 * @param {string} filePath
 * @returns {object}
 */
function ensureWhoamiOnDisk(filePath) {
  let data = {
    name: '未知',
    traits: [],
    preferences: [],
    basics: {},
    relationship_note: '',
    last_updated: Date.now(),
  };
  try {
    const fs = require('fs');
    if (fs.existsSync(filePath)) {
      data = { ...data, ...JSON.parse(fs.readFileSync(filePath, 'utf8')) };
    }
  } catch (_) { /* ignore */ }
  const next = bootstrapWhoamiRecord(data);
  const changed = JSON.stringify(next) !== JSON.stringify(data);
  if (changed && filePath) {
    try {
      const fs = require('fs');
      fs.writeFileSync(filePath, JSON.stringify(next, null, 2));
    } catch (_) { /* ignore */ }
  }
  return next;
}

/**
 * @param {object} [whoami]
 * @returns {string}
 */
function resolvePartnerDisplayName(whoami = {}) {
  if (whoami.name && whoami.name !== '未知') return String(whoami.name);
  if (isOkabePartnerMode()) return OKABE_PROFILE.name;
  return '';
}

/**
 * @param {object} [whoami]
 */
function partnerIsOkabe(whoami = {}) {
  if (!isOkabePartnerMode()) return false;
  if (whoami.partner_id === 'okabe') return true;
  const n = String(whoami.name || '');
  if (/冈部|Okabe|凤凰院|凶真|伦太郎/i.test(n)) return true;
  return shouldBootstrapOkabeWhoami();
}

function userClaimsOkabe(text) {
  return /我是.{0,6}冈部|我就是冈部|我当然是冈部|你不知道我是冈部|不记得我是冈部|我不是冈部吗|我明明就是冈部/i.test(
    String(text || ''),
  );
}

function userAsksAboutOkabe(text) {
  return /冈部是谁|谁是冈部|你不认识冈部|不知道冈部|冈部是什么人|冈部是什么/.test(String(text || ''));
}

/**
 * @param {object} [whoami]
 * @param {string} [userText]
 * @returns {string}
 */
function buildPartnerContextBlock(whoami = {}, userText = '') {
  const t = String(userText || '');
  if (!partnerIsOkabe(whoami) && !userClaimsOkabe(t)) return '';

  const name = resolvePartnerDisplayName(whoami);
  const lines = [
    `【对话对象 · 每轮生效】对方是${name}（冈部伦太郎 / 凤凰院凶真）。你们很熟悉，拌过无数次嘴，不是第一次见面。`,
    '禁止：「冈部是谁」「我不认识冈部」「您哪位」「抱歉不认识这个名字」——对熟人的日常反应。',
    '口吻：像微信里和熟人——短、上口、会吐槽他中二；别讲经、别客服、别动不动论文/实验开场。',
    '称呼：平时叫「你」；吐槽时可用冈部/凶真/那个笨蛋，不必每句报名字。',
  ];

  if (userClaimsOkabe(t)) {
    lines.push('他在强调自己就是冈部：别装傻，用你们一贯的互怼/熟人口吻接话。');
  }
  if (userAsksAboutOkabe(t)) {
    lines.push('他问冈部是谁：多半在试探你记不记得——承认你们很熟，别百科式介绍陌生人。');
  }
  if (/我是谁|你还记得我是谁|我叫什么|你知道我是谁/.test(t)) {
    lines.push(`他问身份：你记得他是${name}（冈部）；自然说出名字，禁止空泛「我当然知道」却不报名字。`);
  }
  return lines.join('\n');
}

function replyLooksLikeUnknownOkabe(userText, reply) {
  if (!isOkabePartnerMode()) return false;
  const o = String(reply || '');
  const u = String(userText || '');
  if (
    /不认识.{0,8}冈部|冈部是谁[？?]?$|没听过冈部|不知道.{0,6}冈部|哪位是冈部|谁是冈部[^伦太郎]|冈部.{0,4}哪位/i.test(o)
  ) {
    return true;
  }
  if (/陌生人|第一次见面|您哪位|没听说过你/.test(o) && /冈部|凶真|凤凰院/.test(u)) {
    return true;
  }
  if (/我是谁|我叫什么|你还记得我是谁/.test(u) && /你没告诉|不知道你叫|不认识你|没说过名字/.test(o)) {
    return true;
  }
  return false;
}

function unknownOkabeFallback(userText) {
  const t = String(userText || '');
  if (userClaimsOkabe(t)) return '……你不就是冈部吗？还想让我走一遍认亲流程？';
  if (userAsksAboutOkabe(t)) return '哈？凤凰院凶真，你今天又中二到连自己都不认识了？';
  if (/我是谁|你还记得我是谁/.test(t)) return '你是冈部伦太郎啊……除非你又想听一遍全名才满意？';
  return '……冈部，你今天怎么回事？';
}

const api = {
  OKABE_PROFILE,
  getPartnerId,
  isOkabePartnerMode,
  shouldBootstrapOkabeWhoami,
  bootstrapWhoamiRecord,
  ensureWhoamiOnDisk,
  resolvePartnerDisplayName,
  partnerIsOkabe,
  userClaimsOkabe,
  userAsksAboutOkabe,
  buildPartnerContextBlock,
  replyLooksLikeUnknownOkabe,
  unknownOkabeFallback,
};

if (typeof module !== 'undefined' && module.exports) {
  module.exports = api;
}
if (typeof window !== 'undefined') {
  window.AmadeusPartnerIdentity = api;
}
