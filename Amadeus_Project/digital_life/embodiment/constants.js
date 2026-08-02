'use strict';

const SCENE_LEX = {
  tired: [/疲惫|困|打哈欠|累|眼皮/],
  positive: [/微笑|笑|开心|愉快/],
  focused: [/专注|工作|打字|认真|看书/],
  absent: [/离开|不在|空|没人|走了/],
  distracted: [/手机|分心|低头|刷/],
  dim: [/黑暗|暗|灯关|昏暗/],
  lonely: [/一个人|孤独|空房间/],
};

const EXPRESSION_PRESETS = [
  { id: 'neutral', label: '平静', pad: { P: 0, A: 0, D: 0 } },
  { id: 'warm', label: '柔和', pad: { P: 0.45, A: 0.15, D: -0.1 } },
  { id: 'shy', label: '害羞', pad: { P: 0.2, A: 0.35, D: -0.35 } },
  { id: 'alert', label: '警觉', pad: { P: -0.1, A: 0.55, D: 0.25 } },
  { id: 'cold', label: '冷淡', pad: { P: -0.45, A: -0.15, D: 0.35 } },
  { id: 'sad', label: '低落', pad: { P: -0.35, A: -0.2, D: -0.15 } },
  { id: 'excited', label: '兴奋', pad: { P: 0.35, A: 0.65, D: 0.15 } },
  { id: 'shocked', label: '震惊', pad: { P: 0.05, A: 0.78, D: -0.2 } },
];

const SPECIAL_DATES = new Map([
  ['02-14', '情人节'],
  ['03-14', '白色情人节'],
  ['12-25', '圣诞'],
  ['01-01', '新年'],
  ['08-15', '中元'],
]);

module.exports = { SCENE_LEX, EXPRESSION_PRESETS, SPECIAL_DATES };
