'use strict';

const userPresence = require('../lib/userPresence');

/**
 * 对话线程承接：尤其她主动开口后，用户来接话时的焦点约束。
 */

function _clip(s, max = 160) {
  const t = String(s || '').trim().replace(/\s+/g, ' ');
  if (!t) return '';
  return t.length <= max ? t : `${t.slice(0, max)}…`;
}

/**
 * @param {string} userText
 * @param {string} proactiveAnchor 她上一轮主动消息正文
 * @returns {string}
 */
function buildProactiveReplyFocus(userText, proactiveAnchor) {
  const anchor = _clip(proactiveAnchor, 200);
  const u = String(userText || '').trim();
  if (!anchor) return '';
  return [
    '【承接主动话题 · 内化，勿复述】',
    `上一轮是你主动对他说的：「${anchor}」`,
    '他现在是在接你刚抛出来的话（剧情、观点、吐槽都行）。你要顺着这条线往下聊：接细节、补一句看法、或问他一句相关的。',
    '禁止：指责他敷衍、禁止「转移话题」「打算就这么了事」式质问、禁止说他没认真回——除非他明显在骂人或完全无关且你自己先冷场。',
    u.length <= 16
      ? '他这句偏短：当作接梗或附和，你可以多接半句把话题托住，不要扣帽子。'
      : '先对齐他这句里和你主动话题相关的点，再展开。',
  ].join('\n');
}

/**
 * 从多轮 dialogue 推断：当前 user 是否在接她上一句 assistant（且非 SKIP/想说话 占位）
 * @param {Array<{ role: string, content: string }>} dialogue
 */
/**
 * 从 dialogue 取最后一条真实用户话（跳过想说话/占位）
 * @param {Array<{ role: string, content: string }>} dialogue
 */
function extractLastRealUserLine(dialogue) {
  if (!Array.isArray(dialogue)) return '';
  for (let i = dialogue.length - 1; i >= 0; i--) {
    const m = dialogue[i];
    if (!m || m.role !== 'user') continue;
    const c = String(m.content || '').trim();
    if (!c || /^（想说话）|^（转移话题）|^（以下是最近对话/.test(c)) continue;
    return c;
  }
  return '';
}

/**
 * 主动开口：接他上一句、禁止编造实验/无关心换题
 * @param {object} opts
 * @param {string} [opts.lastUserText]
 * @param {string} [opts.lastKurisuLine]
 */
function buildAutonomyContinuityBlock(opts = {}) {
  const lastUser = String(opts.lastUserText || '').trim();
  const lastK = String(opts.lastKurisuLine || '').trim();
  const presence = opts.userPresence || null;
  const lines = [
    '【主动开口 · 必守】',
    '你是来续聊或日常敲他，不是来表演研究者、不是写科幻/论文开场白。',
    '禁止编造：他没提过的实验、论文、量子力学、睡眠剥夺研究、实验室乱糟糟的数据——一律不许捏造。',
    '禁止无关心换题：他刚说饿就接吃的/让他去吃饭；说头疼就接头疼或关心；说无聊就接无聊——别提旅行、别突然科普、别换全新人设剧情。',
  ];
  if (userPresence.isPresenceActive(presence)) {
    lines.push(
      userPresence.buildPresencePromptBlock(presence, { isAutonomy: true }),
    );
  } else if (lastUser) {
    lines.push(
      `他最近一句：「${_clip(lastUser, 140)}」——优先接这句话里的词和情绪；可问一句相关的，或丢一句你这边的小事，但不要丢题。`,
    );
    if (/饿|想吃|好饿|肚子/.test(lastUser)) {
      lines.push('锚点：饿——接吃饭、零食、催他去吃，不要扯旅行或实验。');
    }
    if (/要睡|睡觉|晚安|去睡|困死了|好困|别吵.*睡/.test(lastUser)) {
      lines.push('锚点：要休息——祝晚安或一句轻的，禁止催回复、禁止「人呢/怎么不理我」。');
    } else if (/忙|没空|开会|加班|上课|写代码|赶工|勿扰|别烦|别吵|晚点再说|不想聊/.test(lastUser)) {
      lines.push('锚点：他忙或不想被打扰——别催已读，最多一句轻的，禁止抱怨不理你。');
    } else if (/头疼|头痛|疼|难受|累|困|睡不着/.test(lastUser)) {
      lines.push('锚点：不舒服——接关心、休息、别硬撑，不要扯你在做什么研究。');
    }
    if (/无聊|没意思/.test(lastUser)) {
      lines.push('锚点：无聊——接闲聊、在干嘛、吐槽，不要硬推销实验室或科学话题。');
    }
  } else if (lastK) {
    lines.push(
      `你上一句：「${_clip(lastK, 120)}」——若再开口只可顺着这句追问或补半句，不要另起科学实验人设。`,
    );
  } else {
    lines.push('没有新话题时：只发日常短句（在干嘛/吃了吗/怎么不回），不要凭空编你在做什么实验。');
  }
  lines.push(
    '口吻靠传记/whoami/最近对话维持，记忆照常用；只禁止编造他没提过的课题，别把本人演成陌生人。',
  );
  lines.push(
    '可连发 1～3 条短消息（像微信连发），每条单独一行，行与行之间空一行；每条 1～2 句；同一主题连发时换说法，禁止三条都在重复抱怨同一外号。',
  );
  lines.push('禁止每条都用「那个笨蛋」起手；对冈部可直接叫「你」或冈部/凶真。');
  return lines.join('\n');
}

function detectReplyingToHerThread(dialogue) {
  if (!Array.isArray(dialogue) || dialogue.length < 2) {
    return { active: false, anchor: '' };
  }
  let lastUser = '';
  let lastAsst = '';
  for (let i = dialogue.length - 1; i >= 0; i--) {
    const m = dialogue[i];
    if (!m || !m.content) continue;
    if (m.role === 'user' && !lastUser) {
      lastUser = String(m.content).trim();
      continue;
    }
    if (m.role === 'assistant' && !lastAsst && lastUser) {
      lastAsst = String(m.content).trim();
      break;
    }
  }
  if (!lastAsst || !lastUser) return { active: false, anchor: '' };
  if (/^（想说话）|^（转移话题）|^（以下是最近对话/.test(lastUser)) {
    return { active: false, anchor: '' };
  }
  if (lastAsst.length < 4) return { active: false, anchor: '' };
  return { active: true, anchor: lastAsst };
}

/**
 * 主动消息里编造用户没提过的「研究/实验」人设
 * @param {string} userAnchor
 * @param {string} reply
 */
function replyLooksLikeAutonomyFabrication(userAnchor, reply) {
  const o = String(reply || '');
  if (
    !/量子|睡眠剥夺|神经认知|咖啡因.{0,16}剥夺|实验室.{0,12}数据|研究.{0,8}影响|拧断.{0,4}脑子|论文|假说|世界线/i.test(
      o,
    )
  ) {
    return false;
  }
  const u = String(userAnchor || '');
  if (/量子|睡眠|实验|研究|数据|论文|神经认知|剥夺|实验室|咖啡因/.test(u)) return false;
  return true;
}

function autonomyFabricationFallback(userAnchor) {
  const u = String(userAnchor || '');
  if (/饿|想吃|好饿|肚子/.test(u)) return '你都喊饿了还不去吃？别光在这打字。';
  if (/头疼|头痛|疼|难受|累|困/.test(u)) return '……不舒服就歇会儿，别硬撑。';
  if (/无聊/.test(u)) return '无聊就说话，别装死。';
  if (/在吗|人呢|不理/.test(u)) return '在。怎么了？';
  return '……所以呢？';
}

module.exports = {
  buildProactiveReplyFocus,
  buildAutonomyContinuityBlock,
  extractLastRealUserLine,
  detectReplyingToHerThread,
  replyLooksLikeAutonomyFabrication,
  autonomyFabricationFallback,
};
