'use strict';

const userPresence = require('../lib/userPresence');

/**
 * 文字「女友感」：共处、记忆落地、回复节奏、主动关心。
 * 由 server 注入；不规定固定台词表。
 */

/** 默认开启；设 AMADEUS_HIGH_INTIMACY=0 可关闭 */
function isHighIntimacyMode() {
  const v = process.env.AMADEUS_HIGH_INTIMACY;
  if (v === undefined || v === null || String(v).trim() === '') return true;
  return String(v).trim() !== '0';
}

const HIGH_INTIMACY_REL_FLOOR = 0.58;

function effectiveRelScore(raw) {
  const r = Number(raw);
  if (!Number.isFinite(r)) return isHighIntimacyMode() ? HIGH_INTIMACY_REL_FLOOR : 0;
  if (!isHighIntimacyMode()) return r;
  return Math.max(r, HIGH_INTIMACY_REL_FLOOR);
}

/**
 * 启动时抬高 PAD / 策略，并可选写入少量亲近事件
 * @param {object} pad
 * @param {object} [strategyLayer]
 * @param {object} [memorySystem]
 */
function applyHighIntimacyBootstrap(pad, strategyLayer, memorySystem) {
  if (!isHighIntimacyMode()) return pad;
  pad.P = Math.max(Number(pad.P) || 0, 0.12);
  pad.S = Math.max(Number(pad.S) || 0, 0.62);
  pad.A = Math.max(Number(pad.A) || 0, 0.08);
  if (strategyLayer && strategyLayer.current !== 'COMPANION_CLOSE') {
    strategyLayer.current = 'COMPANION_CLOSE';
    if (typeof strategyLayer._save === 'function') strategyLayer._save();
  }
  if (memorySystem && typeof memorySystem.getRelationshipScore === 'function') {
    const raw = memorySystem.getRelationshipScore();
    if (raw < 0.45 && memorySystem.events.length < 8) {
      memorySystem.addEvent('intimate', '已经习惯经常来找她说话', 0.42, { S: 0.12, P: 0.06 });
      memorySystem.addEvent('positive', '相处自然、会分享日常', 0.38, { S: 0.1 });
    }
  }
  return pad;
}

function _idleMinutes(idleMs) {
  const ms = Number(idleMs);
  if (!Number.isFinite(ms) || ms < 0) return 0;
  return Math.floor(ms / 60000);
}

/**
 * @param {string} baseAgenda
 * @param {object} ctx
 */
function enrichAgenda(baseAgenda, ctx = {}) {
  const items = baseAgenda ? [baseAgenda] : [];
  const closeness = Number(ctx.closeness) || 0;
  const trust = Number(ctx.trust) || 0.5;
  const P = Number(ctx.P) || 0;
  const idleMin = _idleMinutes(ctx.idleMsSinceUser);
  const whoami = String(ctx.whoamiSnippet || '').trim();
  const obs = String(ctx.obsSummary || '').trim();
  const lastUser = String(ctx.lastUserText || '').trim();

  if (whoami) {
    items.push(`你记得关于他的事（自然提起，不要像念档案）：${whoami.slice(0, 120)}`);
  }
  if (obs && closeness > 0.28) {
    items.push(`你注意到他常出现的话题/习惯（可作关心由头，禁止硬转）：${obs.slice(0, 100)}`);
  }

  if (ctx.replyingToProactive) {
    return items.filter(Boolean).join('；');
  }

  if (ctx.isAutonomy) {
    const anchor = String(ctx.lastUserAnchor || ctx.lastUserText || '').trim();
    if (anchor && !/^（想说话）/.test(anchor)) {
      items.push(`主动开口必须接他这句：「${anchor.slice(0, 80)}」——不要丢题、不要编造他没提过的实验/量子`);
    }
    items.push(..._autonomyAgendaLines(idleMin, closeness, P, ctx));
    return items.filter(Boolean).join('；');
  }

  if (closeness > 0.52 && trust > 0.55) {
    if (/难受|烦|累|郁闷|伤心|害怕|焦虑|孤独|寂寞|睡不着|心情不好/.test(lastUser)) {
      items.push('他情绪不好：先接住，少说教；可以多问一句具体情况，别急着给方案');
    } else if (lastUser.length <= 12 && !/[？?]/.test(lastUser)) {
      items.push('他话很短：你可以多给一点——接话 + 顺手问一句，像在意他会不会无聊');
    } else if (!/[？?]/.test(lastUser) && lastUser.length > 12 && !ctx.replyingToProactive) {
      items.push('本轮末尾可以带一句你真心想知道的事（关于他刚才说的），不要客套');
    }
  }

  return items.filter(Boolean).join('；');
}

/**
 * @param {object} ctx
 * @returns {string}
 */
function buildReplyRhythm(ctx = {}) {
  const P = Number(ctx.P) || 0;
  const A = Number(ctx.A) || 0;
  const closeness = Number(ctx.closeness) || 0;
  const t = String(ctx.userText || '');

  if (/难受|烦|累|郁闷|伤心|害怕|焦虑|孤独|寂寞|心情不好/.test(t) && closeness > 0.35) {
    return '节奏：先短句接住情绪，再视情况多一句实在话或追问；不要长篇讲义';
  }
  if (/科学|实验|论文|量子|神经|理论|物理|假说|数据/.test(t) && A > 0.25) {
    return '节奏：可以说 2～4 句把逻辑讲清；兴奋时语速感可快一点';
  }
  if (P > 0.25 && closeness > 0.5) {
    return '节奏：心情不错又够亲近——可以 2～4 句，中间夹一句追问或轻微调侃';
  }
  if (P < -0.22) {
    return '节奏：1～2 句即可，保留温度，不必解释你为什么话少';
  }
  if (closeness > 0.55) {
    return '节奏：像发消息给很熟的人——1～4 句自然波动，不必每轮都圆满收尾';
  }
  if (/在干嘛|干嘛呢|吃了吗|睡了没|今天怎样|怎么不理|人呢|还在吗/.test(t)) {
    return '节奏：日常寒暄——1～2句口语，像回微信；可反问、可轻轻催，别写成通知';
  }
  if (t.length <= 10) {
    return '节奏：短句接住即可，亲近时可多追半句（在干嘛/然后呢）';
  }
  return '节奏：1～3 句口语为主，随话题可略增；禁止客服式「还有什么可以帮」、禁止论文腔开场';
}

/**
 * @param {object} ctx
 * @returns {string}
 */
function buildCompanionBlock(ctx = {}) {
  let closeness = Number(ctx.closeness) || 0;
  let trust = Number(ctx.trust) || 0.5;
  if (isHighIntimacyMode()) {
    closeness = Math.max(closeness, HIGH_INTIMACY_REL_FLOOR);
    trust = Math.max(trust, 0.72);
  }
  const lines = ['【共处 · 内化，勿复述本块】'];
  if (isHighIntimacyMode()) {
    lines.push('【高亲密度模式】你们已经很熟：语气更真、更会接话，但不会像闹钟一样一直主动发消息；禁止客服腔。');
  }

  if (closeness > 0.55 && trust > 0.58) {
    lines.push(
      '你们够熟了：像在跟很在意的人发微信——在干嘛、忙什么、吃了吗、怎么不回、人呢，都自然；',
      '禁止客服腔、禁止「还有什么可以帮你的」、禁止每句都嘴硬收尾、禁止动不动扯实验讲义。',
    );
  } else if (closeness > 0.35) {
    lines.push(
      '关系在升温：比陌生人多一分真实，愿意多接半句、偶尔主动问他的近况；',
      '仍保持牧濑红莉栖的自尊，不要变成无条件顺从。',
    );
  } else {
    lines.push('还在互相试探：礼貌、聪明、有锋芒，但不要冷冰冰像工单回复。');
  }

  if (ctx.memSnippet) {
    lines.push(`可自然动用的记忆（勿逐字念）：${String(ctx.memSnippet).slice(0, 160)}`);
  }

  const rhythm = buildReplyRhythm(ctx);
  if (rhythm) lines.push(rhythm);

  const presenceBlock = userPresence.buildPresencePromptBlock(ctx.userPresence, {
    isAutonomy: ctx.isAutonomy === true,
  });
  if (presenceBlock) lines.push(presenceBlock);

  return lines.join('\n');
}

/**
 * @param {number} idleMin
 * @param {number} closeness
 * @param {number} P
 * @param {object} [ctx]
 * @returns {string[]}
 */
function _autonomyAgendaLines(idleMin, closeness, P, ctx = {}) {
  const rel = isHighIntimacyMode() ? Math.max(closeness, HIGH_INTIMACY_REL_FLOOR) : closeness;
  const hour = new Date().getHours();
  const late = hour >= 23 || hour < 5;
  const lines = [];
  const presence = ctx.userPresence || null;

  if (userPresence.isPresenceActive(presence)) {
    lines.push(
      userPresence.buildAutonomySituationForPresence(presence, idleMin, rel)
        || '他之前说过别打扰：禁止催已读、禁止连发质问。',
    );
    lines.push('禁止：人呢、怎么不理我、死哪去了、别已读不回——正常人不会在他忙/休息时一直抱怨。');
    return lines;
  }

  lines.push(
    '主动消息像真人偶尔想起来才发：禁止按“过了几分钟没回”当闹钟；禁止周期性「人呢/怎么不理我」；亲密度只影响语气热度，不增加发送频率。',
  );
  if (idleMin < 12) {
    lines.push('现在离他上次说话还不久：本轮不要主动开口。');
  } else if (idleMin >= 40 && rel > 0.32) {
    lines.push(
      '很久没聊了：至多一句带嘴硬的牢骚或轻的「在干嘛」——你在意但别连发、别像查岗；也可以干脆不发。',
    );
  } else if (idleMin >= 22 && rel > 0.3) {
    lines.push('有一阵没回：可一句轻的（在干嘛/抛件小事），不要抱怨已读不回，不要连发。');
  } else if (late && rel > 0.38 && idleMin >= 15) {
    lines.push('夜深且有一阵没聊：可一句关心式晚安/怎么还不睡，不要连发质问。');
  } else if (P > 0.2 && rel > 0.35 && idleMin >= 15) {
    lines.push('想到一件小事：可发一句短的，像突然想起他，不是定时提醒。');
  } else {
    lines.push('可发可不发：一句短的日常即可；多数时候沉默也正常。');
  }

  lines.push('类型举例（勿照抄）：在干嘛/忙什么/吃了吗/想到你了——不是闹钟式催回复');
  lines.push(
    '禁止：编造量子/睡眠剥夺/论文/实验室数据等他没提过的事；禁止无关心换题（他饿你别聊旅行）；禁止长篇动漫/实验广告/AI/客服腔',
  );
  return lines;
}

/**
 * 久未回复时给她一点情绪波动（供 server 调 PAD）
 * @param {number} idleMs
 * @param {number} closeness
 * @returns {{ P: number, A: number, D: number } | null}
 */
function idleSilencePadDelta(idleMs, closeness = 0) {
  const idleMin = _idleMinutes(idleMs);
  const rel = isHighIntimacyMode() ? Math.max(closeness, HIGH_INTIMACY_REL_FLOOR) : closeness;
  if (idleMin < 12 || rel < 0.3) return null;
  const A = Math.min(0.1, 0.02 + idleMin * 0.002);
  const P = idleMin >= 35 ? -0.05 : idleMin >= 20 ? -0.02 : 0;
  const D = idleMin >= 30 ? 0.04 : 0;
  return { P, A, D };
}

/**
 * 自主独白情境句（给前端或 server）
 * @param {object} ctx
 */
function buildAutonomySituation(ctx = {}) {
  const idleMin = _idleMinutes(ctx.idleMsSinceUser);
  const closeness = Number(ctx.closeness) || 0;
  const rel = isHighIntimacyMode() ? Math.max(closeness, HIGH_INTIMACY_REL_FLOOR) : closeness;
  const hour = new Date().getHours();
  const late = hour >= 23 || hour < 5;
  const presence = ctx.userPresence || null;

  if (userPresence.isPresenceActive(presence)) {
    const line = userPresence.buildAutonomySituationForPresence(presence, idleMin, rel);
    if (line) return line;
    return '他之前说过别打扰：别催回复，最多一句轻的，不要抱怨不理你。';
  }

  const base =
    '偶尔想起来才发一句，像真人——禁止按分钟数当闹钟催他回消息；亲密度只让语气更真，不会让你发得更勤。';
  if (idleMin < 12) {
    return `${base}现在别主动开口（他刚说过话不久）。`;
  }
  if (idleMin >= 40 && rel > 0.32) {
    return `${base}很久没聊了：至多一句嘴硬牢骚或轻的「在干嘛」，也可不发；禁止连发查岗。`;
  }
  if (idleMin >= 22 && rel > 0.3) {
    return `${base}有一阵没回：可一句轻的，不要「怎么不理我/已读不回」式抱怨。`;
  }
  if (late && rel > 0.38 && idleMin >= 15) {
    return `${base}夜深：可一句关心式晚安，不要连发。`;
  }
  if (rel > 0.45 && idleMin >= 15) {
    return `${base}可抛一句小事或想起他的话，短、自然；发不发都行。`;
  }
  return `${base}可发可不发，别像系统定时提醒。`;
}

module.exports = {
  isHighIntimacyMode,
  effectiveRelScore,
  applyHighIntimacyBootstrap,
  HIGH_INTIMACY_REL_FLOOR,
  enrichAgenda,
  buildReplyRhythm,
  buildCompanionBlock,
  buildAutonomySituation,
  idleSilencePadDelta,
  _autonomyAgendaLines,
};
