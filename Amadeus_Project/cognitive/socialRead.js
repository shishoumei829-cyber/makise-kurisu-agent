'use strict';

/**
 * 读场：像人坐在旁边时对气氛的连续判断。
 * 不负责「到点开口」，只描述此刻场子舒不舒服、能不能插话、有没有冷。
 */

function clamp(value, min = 0, max = 1) {
  return Math.max(min, Math.min(max, value));
}

function compact(value) {
  return String(value || '').replace(/\s+/g, ' ').trim();
}

/**
 * @param {object} input
 * @returns {object} continuous social field
 */
function readSocialField(input = {}) {
  const lastUser = compact(input.lastUserText);
  const lastReply = compact(input.lastReplyText);
  const facePresent = input.facePresent === true;
  const faceMs = Math.max(0, Number(input.faceMs) || 0);
  const idleMs = Math.max(0, Number(input.idleMs) || 0);
  const quietMs = Math.max(0, Number(input.quietMs) || idleMs);
  const pad = input.pad || {};
  const rel = clamp(Number(input.relScore) || 0);
  const isThinking = input.isThinking === true;
  const ttsPlaying = input.ttsPlaying === true;
  const awaitingReply = input.awaitingProactiveReply === true;
  const alreadyTalking = input.alreadyTalking === true || input.dialogueStarted === true;

  const cold = /^(嗯+|哦+|喔+|啊+|额+|行|好|随便|没事|在|哦|嗯|知道了|收到|ok|OK|嗯嗯|呵呵)[。！？.!]?$/i.test(lastUser)
    || /不想聊|随便你|都行|无所谓/.test(lastUser);
  const bored = /无聊|没事干|好闲|闲得|陪我|说点什么/.test(lastUser);
  const ending = /晚安|睡了|先忙|拜拜|再见|不用回|我走了/.test(lastUser);
  const asked = /[?？]\s*$/.test(lastReply);
  const shortUser = lastUser.replace(/\s/g, '').length > 0 && lastUser.replace(/\s/g, '').length <= 4;

  // 舒适沉默：人在旁边安静待着也可以，不必填满空气
  let comfortableSilence = 0;
  if (facePresent && quietMs >= 8000) {
    comfortableSilence = clamp(
      0.32
        + Math.min(0.38, quietMs / 200000)
        + (alreadyTalking ? 0.08 : 0)
        + rel * 0.16
        + (Number(pad.P) || 0) * 0.1
        - (cold ? 0.4 : 0)
        - (bored ? 0.22 : 0)
        - (awaitingReply ? 0.18 : 0),
    );
  }

  let tension = 0;
  if (cold || (shortUser && quietMs > 12000)) {
    tension = clamp(0.42 + Math.min(0.4, quietMs / 90000));
  }
  if (bored) tension = Math.max(tension, 0.55);
  if (awaitingReply && quietMs > 22000) tension = Math.max(tension, 0.48);

  let warmth = clamp(
    0.4 + rel * 0.35 + (Number(pad.S) || 0) * 0.2 + (facePresent ? 0.12 : -0.18),
  );
  if (cold) warmth = clamp(warmth - 0.25);

  let engagement = alreadyTalking
    ? clamp(0.55 + (lastUser.length > 8 ? 0.15 : -0.12) - (cold ? 0.3 : 0))
    : clamp(facePresent ? 0.32 + Math.min(0.35, faceMs / 150000) : 0.08);

  let interruptCost = 0.18;
  if (isThinking || ttsPlaying) interruptCost = 0.95;
  if (asked && quietMs < 14000) interruptCost = Math.max(interruptCost, 0.78);
  if (ending) interruptCost = 0.98;
  if (input.dnd === true) interruptCost = 1;

  const floorOpen = interruptCost < 0.55;
  const atmospherePressure = clamp(
    tension * 0.72
      + (facePresent && quietMs > 50000 && comfortableSilence < 0.45 ? 0.22 : 0)
      + (bored ? 0.18 : 0)
      - comfortableSilence * 0.48,
  );

  return {
    facePresent,
    faceMs,
    quietMs,
    comfortableSilence: Number(comfortableSilence.toFixed(3)),
    tension: Number(tension.toFixed(3)),
    warmth: Number(warmth.toFixed(3)),
    engagement: Number(engagement.toFixed(3)),
    interruptCost: Number(interruptCost.toFixed(3)),
    floorOpen,
    atmospherePressure: Number(atmospherePressure.toFixed(3)),
    cold,
    bored,
    ending,
    asked,
  };
}

/**
 * 内驱强度 × 读场 → 这一刻值不值得开口。
 * 舒适沉默会抬高门槛：她可以只是坐着。
 */
function shouldSpeakNow(urgeIntensity, social, opts = {}) {
  const intensity = Number(urgeIntensity) || 0;
  if (!social) return { ok: false, reason: 'no_social', need: 1, intensity };
  if (social.ending) return { ok: false, reason: 'ending', need: 1, intensity };
  if (!social.floorOpen) return { ok: false, reason: 'floor_closed', need: 1, intensity };

  const comfortPenalty = (social.comfortableSilence || 0) * 0.38;
  const tensionBoost = (social.tension || 0) * 0.22;
  const besideBoost = social.facePresent ? 0.06 : -0.18;
  const relBoost = clamp(Number(opts.relScore) || 0) * 0.1;
  const pressureBoost = (social.atmospherePressure || 0) * 0.16;

  const need = clamp(
    0.4 + comfortPenalty - tensionBoost - besideBoost - relBoost - pressureBoost,
    0.22,
    0.85,
  );

  if (intensity < need) {
    return { ok: false, reason: 'urge_not_ripe', need, intensity };
  }
  return { ok: true, reason: 'urge_and_moment', need, intensity };
}

/** 读场建议多久后再感受一次（冲动越强越勤；舒适沉默越久越懒） */
function nextSenseMs(social, urgeIntensity = 0) {
  const comfort = social?.comfortableSilence || 0;
  const intensity = clamp(Number(urgeIntensity) || 0);
  if (!social?.facePresent) return 10000;
  if (comfort > 0.65) return Math.round(14000 + (1 - intensity) * 10000);
  if (intensity > 0.55) return Math.round(3500 + (1 - intensity) * 4000);
  return Math.round(6000 + (1 - intensity) * 7000);
}

module.exports = {
  readSocialField,
  shouldSpeakNow,
  nextSenseMs,
  clamp,
};
