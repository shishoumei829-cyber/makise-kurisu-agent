'use strict';

const fs   = require('fs');
const path = require('path');

/**
 * PAD 状态管理 + 事件推断
 *
 * 导出：
 *   PAD_BASE, PAD_DECAY_LAMBDA
 *   loadPAD(filePath)
 *   savePAD(filePath, pad)
 *   updatePAD(pad, delta, eventImportance?)
 *   inferMainEventFromInput(userInput, currentPAD)
 */

const PAD_BASE         = { P: -0.1, A: 0.2, D: 0.6, S: 0.0 };
const PAD_DECAY_LAMBDA = 0.0001;
const BOND_DELTA_S     = 0.006;

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

function loadPAD(filePath) {
  try {
    if (fs.existsSync(filePath)) {
      const raw = JSON.parse(fs.readFileSync(filePath, 'utf8'));
      const lastOnline = raw.lastOnline || 0;
      const dt = (Date.now() - lastOnline) / 1000;
      const decay = Math.exp(-PAD_DECAY_LAMBDA * dt);
      return {
        P: PAD_BASE.P + ((raw.P || PAD_BASE.P) - PAD_BASE.P) * decay,
        A: PAD_BASE.A + ((raw.A || PAD_BASE.A) - PAD_BASE.A) * decay,
        D: PAD_BASE.D + ((raw.D || PAD_BASE.D) - PAD_BASE.D) * decay,
        S: Math.min(1, (raw.S || PAD_BASE.S) + 0.001),
      };
    }
  } catch {}
  return { ...PAD_BASE };
}

function savePAD(filePath, pad) {
  fs.writeFile(filePath, JSON.stringify({ ...pad, lastOnline: Date.now() }, null, 2), (err) => {
    if (err) console.error('[pad] Save error:', err.message);
  });
}

/**
 * 非线性 PAD 更新（情绪惯性 + 刺激评价 + 性格底色 + 微扰噪声 + 羁绊钝化）
 */
function updatePAD(pad, delta, eventImportance = 0.5) {
  const alpha   = 0.86;
  const beta    = clamp(0.2 + eventImportance * 0.6, 0.2, 0.8);
  const gamma   = 0.18;
  const noise   = 0.015;
  const S       = pad.S;
  const negDamp = 1 - S * 0.5;

  const next = (key) => {
    const current      = pad[key] || 0;
    const stimulus     = delta[key] || 0;
    const dampedStim   = stimulus < 0 ? stimulus * negDamp : stimulus;
    const epsilon      = (Math.random() * 2 - 1) * noise;
    return clamp(Math.tanh(
      alpha * current + beta * dampedStim + gamma * PAD_BASE[key] + epsilon
    ), -1, 1);
  };

  return {
    P: next('P'),
    A: next('A'),
    D: next('D'),
    S: clamp(pad.S + (delta.S || 0), 0, 1),
  };
}

/**
 * 从用户输入推断「单一主事件」
 * @param {string} userInput
 * @param {{ S: number }} currentPAD  — 仅需 S 字段用于动态判断
 */
function inferMainEventFromInput(userInput, currentPAD) {
  const t   = userInput;
  const pad = currentPAD || { S: 0 };
  const candidates = [];

  if (/科学|量子|神经|实验|时间机器|Dr\.?Pepper|胡椒/.test(t))
    candidates.push({ type: 'scientific', importance: 0.6, delta: { P: 0.12, A: 0.18, D: 0.05 }, content: `讨论科学：${t.substring(0, 30)}` });

  if (/クリスティーナ|克里斯蒂?娜|助手|Christina|变态|天才.*变态|实验.*少女/i.test(t)) {
    const pDelta = pad.S > 0.5 ? 0.05 : -0.15;
    candidates.push({ type: 'complex', importance: 0.85, delta: { P: pDelta, A: 0.32, D: 0.05 }, content: `实验室外号/拌嘴：${t.substring(0, 30)}` });
  }

  if (/关心|温柔|在乎|谢谢|感谢|辛苦/.test(t))
    candidates.push({ type: 'intimate', importance: 0.55, delta: { P: 0.14, A: 0.05, D: -0.2, S: 0.015 }, content: `表达关心：${t.substring(0, 30)}` });

  if (/笨蛋|蠢|闭嘴|滚|烦死/.test(t))
    candidates.push({ type: 'negative', importance: 0.5, delta: { P: -0.2, A: 0.1 }, content: `粗鲁言辞：${t.substring(0, 30)}` });

  if (/喜欢你|爱你|爱上/.test(t))
    candidates.push({ type: 'intimate', importance: 0.85, delta: { P: 0.2, D: -0.28, S: 0.02 }, content: `情感表白：${t.substring(0, 30)}` });

  if (/孤独|一个人|没人/.test(t))
    candidates.push({ type: 'neutral', importance: 0.4, delta: { P: -0.05, A: -0.05 }, content: `提及孤独：${t.substring(0, 30)}` });

  if (/红莉栖|牧濑|Kurisu/.test(t) && /笨蛋|蠢|废物/.test(t))
    candidates.push({ type: 'insulted', importance: 1.0, delta: { P: -0.22, A: 0.3, D: 0.1 }, content: `被骂：${t.substring(0, 30)}` });

  if (/好累|累死|睡不着|熬夜/.test(t))
    candidates.push({ type: 'user_tired', importance: 0.5, delta: { P: 0.04 }, content: `用户疲惫：${t.substring(0, 30)}` });

  if (/睡觉|去睡|睡了|晚安|补觉|先睡/.test(t))
    candidates.push({ type: 'neutral', importance: 0.62, delta: { P: 0.03, A: -0.06 }, content: `他说要去睡觉：${t.substring(0, 40)}` });

  if (!candidates.length)
    return { type: 'neutral', importance: 0.1, delta: { S: BOND_DELTA_S }, content: '正常对话积累' };

  candidates.sort((a, b) => b.importance - a.importance);
  const main = { ...candidates[0], delta: { ...(candidates[0].delta || {}) } };
  main.delta.S = (main.delta.S || 0) + BOND_DELTA_S;
  return main;
}

module.exports = {
  PAD_BASE,
  PAD_DECAY_LAMBDA,
  BOND_DELTA_S,
  clamp,
  loadPAD,
  savePAD,
  updatePAD,
  inferMainEventFromInput,
};
