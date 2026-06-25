'use strict';

/** 内驱力类型与中文标签 */
const DRIVE_TYPES = {
  CURIOSITY: '好奇心',
  CREATIVITY: '创造性',
  EXPLORATION: '探索欲',
  MASTERY: '掌控欲',
  CONNECTION: '连接欲',
  AUTONOMY: '自主欲',
  MEANING: '意义欲',
  PLAYFULNESS: '玩心',
  PROTECTION: '保护欲',
  CREATION: '创造欲',
};

/** 冲动可转化的行动意图 */
const ACTION_INTENTS = {
  ASK_QUESTION: '想问清楚',
  REACH_OUT: '想主动联系',
  EXPLORE_TOPIC: '想深入话题',
  SELF_EXPRESSION: '想说点自己的话',
  HOLD_BACK: '想先憋着',
  DEEPEN_BOND: '想拉近一点',
  DEFEND_SELF: '想守住边界',
  PLAYFUL_JAB: '想轻轻怼一下',
  CREATE_IDEA: '有个新念头',
  STAY_SILENT: '宁愿先不说话',
};

/** 自主行为环输出动作 */
const AUTONOMY_ACTIONS = {
  SPEAK: '开口',
  WAIT: '等待',
  REFLECT: '内化',
};

/**
 * 各内驱力基底水平、衰减半衰期(ms)、饱和恢复率
 * basal: 静息时的趋向；halfLife: 无刺激时回落速度
 */
const DRIVE_PHYSIOLOGY = {
  CURIOSITY:    { basal: 0.55, halfLife: 45 * 60 * 1000, satiationDecay: 0.08 },
  CREATIVITY:   { basal: 0.45, halfLife: 60 * 60 * 1000, satiationDecay: 0.06 },
  EXPLORATION:  { basal: 0.40, halfLife: 50 * 60 * 1000, satiationDecay: 0.07 },
  MASTERY:      { basal: 0.50, halfLife: 90 * 60 * 1000, satiationDecay: 0.05 },
  CONNECTION:   { basal: 0.35, halfLife: 35 * 60 * 1000, satiationDecay: 0.10 },
  AUTONOMY:     { basal: 0.60, halfLife: 70 * 60 * 1000, satiationDecay: 0.04 },
  MEANING:      { basal: 0.38, halfLife: 80 * 60 * 1000, satiationDecay: 0.05 },
  PLAYFULNESS:  { basal: 0.30, halfLife: 40 * 60 * 1000, satiationDecay: 0.12 },
  PROTECTION:   { basal: 0.42, halfLife: 55 * 60 * 1000, satiationDecay: 0.09 },
  CREATION:     { basal: 0.44, halfLife: 65 * 60 * 1000, satiationDecay: 0.07 },
};

/**
 * 内驱力交叉抑制：当 A 激活高时，抑制 B 的净激活
 * 值 0~1 表示抑制强度
 */
const DRIVE_INHIBITION = {
  CONNECTION: { AUTONOMY: 0.35, PROTECTION: 0.15 },
  AUTONOMY: { CONNECTION: 0.40, MEANING: 0.10 },
  PROTECTION: { CONNECTION: 0.45, PLAYFULNESS: 0.30 },
  PLAYFULNESS: { PROTECTION: 0.20, MASTERY: 0.15 },
  CURIOSITY: { STAY_SILENT: 0.25 },
  MASTERY: { PLAYFULNESS: 0.20 },
};

/** 冲动意图 → 行为候选加分 */
const INTENT_BEHAVIOR_MAP = {
  ASK_QUESTION: { ENGAGE: 0.22, CASUAL: 0.08 },
  REACH_OUT: { APPROACH: 0.28, CASUAL: 0.12 },
  EXPLORE_TOPIC: { ENGAGE: 0.30, CASUAL: 0.05 },
  SELF_EXPRESSION: { APPROACH: 0.20, CASUAL: 0.10 },
  HOLD_BACK: { WITHDRAW: 0.25, DEFLECT: 0.12 },
  DEEPEN_BOND: { APPROACH: 0.26, CASUAL: 0.14 },
  DEFEND_SELF: { DEFEND: 0.28, WITHDRAW: 0.10 },
  PLAYFUL_JAB: { DEFLECT: 0.18, CASUAL: 0.15 },
  CREATE_IDEA: { ENGAGE: 0.18, APPROACH: 0.10 },
  STAY_SILENT: { WITHDRAW: 0.30 },
};

/** 科学/情感/日常 话题簇，供好奇心与创造性使用 */
const TOPIC_CLUSTERS = {
  science: /科学|量子|神经|时间|物理|数学|实验|理论|论文|假说|数据|脑|认知/,
  emotion: /喜欢|爱|在乎|难受|烦|累|孤独|寂寞|心情|害怕|担心/,
  daily: /吃|睡|玩|看|听|今天|明天|工作|学校|在干嘛|忙/,
  meta: /意识|存在|意义|真实|AI|程序|记忆|思考/,
  relationship: /你|我|我们|朋友|亲近|讨厌|笨蛋/,
};

module.exports = {
  DRIVE_TYPES,
  ACTION_INTENTS,
  AUTONOMY_ACTIONS,
  DRIVE_PHYSIOLOGY,
  DRIVE_INHIBITION,
  INTENT_BEHAVIOR_MAP,
  TOPIC_CLUSTERS,
};
