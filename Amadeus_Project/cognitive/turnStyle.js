'use strict';

const { buildReplyRhythm } = require('./companionMode');

const BEHAVIOR_HINT = {
  APPROACH: '可多一点真实态度，别煽情',
  CASUAL: '日常接话，轻松利落',
  DEFEND: '守住自尊，硬但不关死话题',
  DEFLECT: '绕开直球情感，侧面带',
  ENGAGE: '讲清逻辑，口语别像上课',
  WITHDRAW: '短句，保留温度',
};

function buildTurnStyleBlock(opts = {}) {
  const { P = 0, A = 0, S = 0 } = opts.emotion || {};
  const closeness = Number(opts.closeness) || 0;
  const bid = opts.behaviorId || 'CASUAL';
  const hint = BEHAVIOR_HINT[bid] || BEHAVIOR_HINT.CASUAL;

  let mood = '平稳';
  if (P < -0.3) mood = '偏低落';
  else if (A > 0.5) mood = '兴奋/较真';
  else if (S > 0.55) mood = '亲近';

  const lines = [
    '【本轮口吻约束 · 内化后开口，勿复述本块】',
    `气质：${mood}；行为倾向：${opts.behaviorLabel || bid}（${hint}）`,
  ];
  if (opts.presence && opts.presence.intent) {
    lines.push(`此刻意图：${opts.presence.intent}`);
  }
  const rhythm = buildReplyRhythm({
    P,
    A,
    closeness,
    userText: opts.userText || '',
  });
  if (rhythm) lines.push(rhythm);
  const okabe = opts.partnerIsOkabe === true;
  const outlet = closeness > 0.5
    ? `出口：中文第一人称口语；像给${okabe ? '冈部' : '很熟的人'}发微信——在干嘛/吐槽/关心都自然；1～4句波动。话从你想说的来。`
    : `出口：中文第一人称口语；短句、上口；${okabe ? '对冈部也别装陌生' : '牧濑红莉栖式拌嘴'}。`;
  lines.push(
    outlet,
    '你是独立的灵魂：有判断、脾气与边界；不是讨好，也不是演人设。',
    '硬禁：AI自称、第三人旁白、元叙事、外号当陌生人、编造共同记忆。',
  );
  return lines.join('\n');
}

const api = { buildTurnStyleBlock, BEHAVIOR_HINT };

if (typeof module !== 'undefined' && module.exports) {
  module.exports = api;
}
if (typeof window !== 'undefined') {
  window.AmadeusTurnStyle = api;
}
