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
  const s = innerSix.state || innerSix;
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
