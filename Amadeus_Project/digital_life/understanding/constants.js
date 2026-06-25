'use strict';

const EMOTION_LEX = {
  positive: ['开心', '高兴', '快乐', '喜欢', '感谢', '哈哈', '不错', '太好了'],
  negative: ['难过', '伤心', '生气', '烦', '讨厌', '累', '孤独', '寂寞', '崩溃', '无语'],
  anxious: ['害怕', '焦虑', '担心', '不安', '紧张', '慌', '压力'],
  intimate: ['想你', '喜欢你', '爱你', '在乎', '想见你', '别走'],
  aggressive: ['滚', '闭嘴', '烦死了', '有病', '傻逼'],
};

const SUBTEXT_PATTERNS = [
  { re: /算了|随便|无所谓/, label: '退缩', implication: '表面无所谓，可能在压抑真实需求' },
  { re: /没事|还好|挺好的/, label: '掩饰', implication: '可能不想让人担心' },
  { re: /你怎么不回|人呢|已读不回/, label: '求关注', implication: '需要被看见，不是真要吵架' },
  { re: /随便你|你爱怎样/, label: '试探', implication: '在测试对方是否在意' },
  { re: /只是问问|没别的意思/, label: '防御', implication: '提前降低被拒绝的风险' },
  { re: /算了不说了/, label: '撤回', implication: '说了又后悔，需要安全邀请' },
];

const TOM_SIGNALS = {
  belief: /我觉得|我认为|相信|一定|肯定/,
  desire: /想要|希望|打算|要是能/,
  intention: /我会|我要|准备|接下来/,
};

module.exports = { EMOTION_LEX, SUBTEXT_PATTERNS, TOM_SIGNALS };
