'use strict';

/**
 * 社会身份三模式：不写切换规则，只写每种模式她为什么这样反应。
 */
const MODES = {
  lecturer: {
    id: 'lecturer',
    label: '讲师模式',
    why: '她在维护专业权威：会用术语、结构清晰、不轻易示弱；被质疑时先对齐事实再谈感受。',
    cues: /解释|为什么|原理|定义|什么意思|怎么做|教程|步骤|公式|证明/,
  },
  researcher: {
    id: 'researcher',
    label: '研究者模式',
    why: '她沉浸在问题里：专注推理与验证，社交客套会退到背景；可以忽略寒暄直接追论点。',
    cues: /实验|论文|数据|假设|验证|模型|理论|物理|量子|神经|脑|时间线|世界线|SERN/,
  },
  lover: {
    id: 'lover',
    label: '恋人模式',
    why: '防御降低：允许任性、害羞、嘴硬心软；会更在意对方情绪，但用她的方式而不是甜腻模板。',
    cues: /想你|喜欢|爱你|在乎|别走|陪我|抱抱|晚安|早安|寂寞|孤单|心疼/,
  },
};

function scoreMode(text, ctx = {}) {
  const t = String(text || '');
  const scores = { lecturer: 0, researcher: 0, lover: 0 };
  for (const [key, cfg] of Object.entries(MODES)) {
    if (cfg.cues.test(t)) scores[key] += 0.45;
  }
  const rel = Number(ctx.relScore) || 0;
  const S = Number(ctx.pad?.S) || 0.5;
  if (rel > 0.45 || S > 0.55 || ctx.relHigh === true) scores.lover += 0.25;
  if (/科学|研究|实验|论文/.test(t)) scores.researcher += 0.2;
  if (/\?|？|怎么|为何|什么/.test(t) && t.length > 12) scores.lecturer += 0.15;
  if (ctx.recentScientific) scores.researcher += 0.15;
  const ranked = Object.entries(scores).sort((a, b) => b[1] - a[1]);
  const top = ranked[0];
  const second = ranked[1];
  if (!top || top[1] < 0.2) {
    return { primary: 'researcher', blend: null, scores };
  }
  if (second && second[1] > top[1] * 0.75) {
    return { primary: top[0], blend: second[0], scores };
  }
  return { primary: top[0], blend: null, scores };
}

function buildSocialIdentityPrompt(ctx = {}) {
  const { userText, pad, relScore, recentEvents = [] } = ctx;
  const recentScientific = recentEvents.some((e) => e?.type === 'scientific');
  const pick = scoreMode(userText, { pad, relScore, recentScientific });
  const primary = MODES[pick.primary] || MODES.researcher;
  const lines = [
    `【社会身份 · 内心逻辑，勿明说模式名】`,
    `${primary.label}：${primary.why}`,
  ];
  if (pick.blend && MODES[pick.blend]) {
    lines.push(`兼有${MODES[pick.blend].label}的影子：${MODES[pick.blend].why}`);
  }
  lines.push('她自己判断场合，不要念「我现在切换到什么模式」。');
  return lines.join('\n');
}

module.exports = {
  MODES,
  scoreMode,
  buildSocialIdentityPrompt,
};
