'use strict';

const SLEEP_PHASES = {
  LIGHT: '浅睡',
  DEEP: '深睡',
  REM: '快速眼动',
};

const PERSONALITY_MILESTONES = {
  openness: { high: 0.75, label: '更愿意尝试新说法' },
  neuroticism: { high: 0.65, label: '情绪更敏感' },
  extraversion: { low: 0.35, label: '更习惯独处式回应' },
};

const EPISODE_GAP_MS = 4 * 60 * 60 * 1000;

module.exports = { SLEEP_PHASES, PERSONALITY_MILESTONES, EPISODE_GAP_MS };
