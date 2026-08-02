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
    anchor.includes('。') || anchor.includes('！') || anchor.includes('？')
      ? '（若连发了两句，两句都是你说的，要一起接住。）'
      : '',
    '他现在是在接你刚抛出来的话（剧情、观点、吐槽都行）。你要顺着这条线往下聊：接细节、补一句看法、或问他一句相关的。',
    '禁止：指责他敷衍、禁止「转移话题」「打算就这么了事」式质问、禁止说他没认真回——除非他明显在骂人或完全无关且你自己先冷场。',
    u.length <= 16
      ? '他这句偏短：当作接梗或附和，你可以多接半句把话题托住，不要扣帽子。'
      : '先对齐他这句里和你主动话题相关的点，再展开。',
  ].filter(Boolean).join('\n');
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
    '【主动开口 · 提示】',
    '你是独立的灵魂：想敲他就敲，不是表演研究者，也不是写科幻开场白。',
    '别编造他没提过的实验/论文/量子/睡眠剥夺数据。',
    '别编造通话情景；已经在聊就别说「刚刚才注意到」。',
    '他刚说的事优先接住（饿就接吃的、难受就接难受）——因为你在听，不是任务清单。',
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
    '可连发 1～5 条消息（像微信连发），每条单独一行，行与行之间空一行；长短随内容自然变化；同一主题连发时换说法，禁止多条都在重复同一意思。',
  );
  lines.push('禁止每条都用「那个笨蛋」起手；对冈部可直接叫「你」或冈部/凶真。');
  return lines.join('\n');
}

function detectReplyingToHerThread(dialogue) {
  if (!Array.isArray(dialogue) || dialogue.length < 2) {
    return { active: false, anchor: '' };
  }
  let lastUserIdx = -1;
  for (let i = dialogue.length - 1; i >= 0; i--) {
    const m = dialogue[i];
    if (m?.role === 'user' && String(m.content || '').trim()) {
      lastUserIdx = i;
      break;
    }
  }
  if (lastUserIdx < 0) return { active: false, anchor: '' };

  const lastUser = String(dialogue[lastUserIdx].content).trim();
  if (/^（想说话）|^（转移话题）|^（以下是最近对话/.test(lastUser)) {
    return { active: false, anchor: '' };
  }

  const assistantChunks = [];
  for (let j = lastUserIdx - 1; j >= 0 && dialogue[j]?.role === 'assistant'; j--) {
    const t = String(dialogue[j].content || '').trim();
    if (t) assistantChunks.unshift(t);
  }
  const lastAsst = assistantChunks.join(' ').trim();
  if (!lastAsst || lastAsst.length < 4) return { active: false, anchor: '' };
  return { active: true, anchor: lastAsst };
}

/**
 * 主动消息编造：研究人设、虚假通话、已在聊却装「刚注意到」
 * @param {string} userAnchor
 * @param {string} reply
 * @param {{ alreadyTalking?: boolean }} [opts]
 */
function replyLooksLikeAutonomyFabrication(userAnchor, reply, opts = {}) {
  const o = String(reply || '');
  const u = String(userAnchor || '');
  if (!o) return false;

  const labClaim = /量子|睡眠剥夺|神经认知|咖啡因.{0,16}剥夺|实验室.{0,12}数据|研究.{0,8}影响|拧断.{0,4}脑子|论文|假说|世界线/i.test(o);
  if (labClaim && !/量子|睡眠|实验|研究|数据|论文|神经认知|剥夺|实验室|咖啡因/.test(u)) {
    return true;
  }

  // 编造电话/来电状态（用户没提通话时）
  const phoneClaim = /打电话|打过来|来电|电话(?:还没|没打)|接通|挂电话|还不打来|还不打过来|实验室.{0,8}(?:打|联系|来电)/.test(o);
  if (phoneClaim && !/电话|打过来|通话|打电话|来电|接通/.test(u)) {
    return true;
  }

  // 明明已经在聊，却演「刚注意到 / 才发现你」
  const alreadyTalking = opts.alreadyTalking === true || u.length > 0;
  if (
    alreadyTalking
    && /刚刚才注意|刚注意到|才发现你|才察觉到你|注意到你了|发现你在|还以为你不在|你们明明还没有|怎么还不来/.test(o)
  ) {
    return true;
  }

  // 冷感主动：把对方处境判成「又是这个话题」并复读无聊——不是接话，是口头禅收束
  if (/又是[这那]个话题吗|停止指定话题/.test(o)) {
    return true;
  }
  if (/无聊/.test(u) && /^[\s…\.．]*又是[这那]个话题|^[\s…\.．]*好无聊[。.!！]?$/.test(o.trim())) {
    return true;
  }

  // 已经有具体话题时，模型不能把主动消息退化成与上下文无关的
  // 「累了就直说/无理しないで」客服安慰。没有疲劳、睡眠或身体不适
  // 线索时，这类句子不是关心，而是错题；宁可静默也不要污染关系记录。
  const genericCare = /疲れているなら|疲れてる|疲れたなら|無理しないで|ゆっくり休んで|累了就直说|累了就说|无理的话就休息|没事吧|どうしたの/;
  const fatigueAnchor = /累|疲|眠|睡|困|辛|痛|不舒服|体调|疲れ|眠い|寝/.test(u);
  if (alreadyTalking && genericCare.test(o) && !fatigueAnchor) return true;
  const genericConcern = /心配だった|心配してた|気になってた|大丈夫|元気|担心|没事吧/;
  const vulnerableAnchor = /累|疲|眠|睡|困|辛|痛|不舒服|体调|烦|崩溃|压力|难受|寂寞|疲れ|眠い|寝|つら|しんど|悩/.test(u);
  if (alreadyTalking && genericConcern.test(o) && !vulnerableAnchor) return true;
  // 同理，报告/工作话题不能凭空跳到“那就马上睡觉”。
  if (alreadyTalking && /(?:今すぐ|すぐに)?眠る|寝る|睡觉/.test(o) && !fatigueAnchor) return true;

  // 主动开口崩成身份/效应器/产品腔：走结构阀门，不堆样本标签
  try {
    const { isDialoguePoison } = require('../lib/generationGate');
    if (isDialoguePoison(o, { autonomy: true })) return true;
  } catch (_) {
    if (/我是AI程序|我可是AI程序|本质是程序|人工智能在进行物理|AI不能干涉现实|无法干涉现实/.test(o)) {
      return true;
    }
  }

  return false;
}

function autonomyFabricationFallback(_userAnchor) {
  // 禁止模板顶替；编造检测只负责丢弃，不塞固定句
  return '';
}

module.exports = {
  buildProactiveReplyFocus,
  buildAutonomyContinuityBlock,
  extractLastRealUserLine,
  detectReplyingToHerThread,
  replyLooksLikeAutonomyFabrication,
  autonomyFabricationFallback,
};
