'use strict';

/**
 * 同一情绪 band 的多种表达方向（prompt 描述，非硬编码台词）
 */
const VARIANTS = {
  warm: [
    '轻松接话，可带一点笑意但不腻',
    '关心藏在吐槽后面，一句实在话即可',
    '愿意多解释半句，但不写成教程',
    '像想到什么顺口说出来，不刻意煽情',
  ],
  soft: [
    '可以娇半拍、哼一句、小声，仍是她的嘴',
    '羞或恼都行——只要是你自己的反应',
    '黏一点点之后可以用别的话盖住',
    '示弱要短，别丢掉自己的判断去空安慰',
  ],
  shocked: [
    '允许愣住、短促失态、难以置信',
    '可以先拔高语气再收回理性',
    '连问或先骂半句都可以，别装百科冷静',
    '不要旁白「震惊地」，用话本身带出来',
  ],
  aftermath: [
    '认真澄清或追问刚才的冲击点',
    '把话说清楚，不要突然转软或闹',
    '可以仍带一点余惊，但以对齐事实为主',
    '一句一个重点，别串成撒娇或傲娇表演',
  ],
  tender: [
    '先接住他的难受，轻、短、实在',
    '少建议，多在场感',
    '先在场，别急着表演关心',
    '一句实在的胜过长篇空安慰',
  ],
  playful: [
    '接梗、吐槽、闲聊升温',
    '可以闹一下，别任务播报',
    '像微信熟人，节奏自然',
    '轻松但不油',
  ],
  tsundere: [
    '真嘴硬：顶回去、否认外号、不服输',
    '可以漏一句软的，但不要每轮固定二段式',
    '傲娇式否定但仍在听',
    '被看穿时恼羞，用逻辑或吐槽盖过去也行',
  ],
  guarded: [
    '礼貌有距离，句子短',
    '先对齐事实再谈感受',
    '用反问代替直接示弱',
    '嘴硬但别伤人，留台阶',
  ],
  sharp: [
    '反应快，可轻轻怼但不人身攻击',
    '指出逻辑漏洞，语气干脆',
    '不耐烦时用事实压过情绪',
    '傲娇式否定但仍在听',
  ],
  low: [
    '话少，节奏慢，不冷漠',
    '承认累/烦但不要自怜表演',
    '少给建议，多给在场感',
    '一句就够，别连环安慰',
  ],
  curious: [
    '追问具体细节，像真的想知道',
    '把话题往深处带半步',
    '联想相关但别跑题太远',
    '科学好奇与日常好奇都可',
  ],
};

function pickBand(pad = {}, innerSix = {}, opts = {}) {
  const P = pad.P || 0;
  const A = pad.A || 0;
  const D = pad.D || 0;
  const s = innerSix.state || innerSix;
  const affectBand = String(opts.affectBand || '');
  if (affectBand && VARIANTS[affectBand]) return affectBand;
  if (A > 0.65 && D < 0) return 'shocked';
  if (opts.relHigh === true && D < -0.2 && P > 0.1) return 'soft';
  if (opts.relHigh === true && (s.connection || 0) > 0.5) return 'warm';
  if (P < -0.3 || (s.weight || 0) > 0.62) return 'low';
  if ((s.boundary || 0) > 0.7 && (s.connection || 0) < 0.45) return 'guarded';
  if (A > 0.55 && P > 0) return 'sharp';
  if ((s.connection || 0) > 0.58 || P > 0.35) return 'warm';
  if (A > 0.35) return 'curious';
  return 'guarded';
}

function buildExpressionVariantBlock(pad = {}, innerSix = {}, opts = {}) {
  const band = pickBand(pad, innerSix, opts);
  const pool = VARIANTS[band] || VARIANTS.guarded;
  const n = 3 + Math.floor(Math.random() * 2);
  const picked = [];
  const used = new Set();
  while (picked.length < n && used.size < pool.length) {
    const i = Math.floor(Math.random() * pool.length);
    if (used.has(i)) continue;
    used.add(i);
    picked.push(pool[i]);
  }
  return `【表达变体 · ${band}】本轮可从以下方向任选其一流畅表达（不要全用、不要照念）：${picked.join(' / ')}`;
}

module.exports = {
  VARIANTS,
  pickBand,
  buildExpressionVariantBlock,
};
