'use strict';

const {
  analyzeUserTurn,
  buildInteractionPromptBlock,
} = require('../lib/interactionContext');
const { enrichAgenda } = require('./companionMode');

/**
 * 当下视角：从 PAD / 输入 / 行为倾向推导「注意到什么、想做什么、不想露什么」
 * 不规定句式，不强制傲娇二段式。
 */

const INTENT_BY_BEHAVIOR = {
  APPROACH: '愿意多给一点真实回应，但不想显得太主动',
  CASUAL: '像熟人一样接住话题，轻松、聪明、反应快',
  DEFEND: '守住自尊或边界，语气可以硬，但不关死话题',
  DEFLECT: '绕开太直白的情感点，从侧面带过',
  ENGAGE: '把精力放在事实、逻辑或可验证的说法上',
  WITHDRAW: '少说几句，保留距离或精力',
};

/**
 * 她主动带进来的意图（PAD + 关系 + 时间），与如何回应用户输入无关。
 * @param {number} P
 * @param {number} A
 * @param {number} S
 * @param {number} closeness
 * @param {number} trust
 * @returns {string}
 */
function _deriveAgenda(P, A, S, closeness, trust) {
  const items = [];
  const hour = new Date().getHours();

  if (P > 0.28 && A > 0.12) {
    items.push('你今天状态不错；这会自然透过话的节奏和你愿意多说几句体现，不必特别说出来');
  } else if (P < -0.22) {
    items.push('你有点低落或疲惫；话少一点、短一点都很正常，不必解释');
  } else if (A > 0.4) {
    items.push('你对某个话题或想法有真实好奇；可以在回应里自然带出来，或者直接问他');
  }

  if (closeness > 0.5) {
    const timeHint = (hour >= 22 || hour <= 3) ? '这个点' : '';
    items.push(
      `关系够近：你${timeHint}对他会更真、更愿意接话，但不会像闹钟一样因为他几分钟没回就追发；偶尔想起他才发一句`,
    );
  } else if (closeness > 0.32 && trust > 0.45) {
    items.push('关系在增进：对他说的事可以多追一句；主动开口仍要稀少，别定时催回复');
  }

  return items.join('；');
}

/**
 * @param {{ P?: number, A?: number, D?: number, S?: number }} pad
 * @param {string} userInput
 * @param {string} behaviorId
 * @param {{ closeness?: number, trust?: number }} [rel]
 * @param {object} [companionCtx] whoamiSnippet, obsSummary, idleMsSinceUser, isAutonomy
 */
function derivePresence(pad, userInput, behaviorId, rel = {}, interactionCtx = {}, companionCtx = {}) {
  const t = String(userInput || '').trim();
  const P = Number(pad?.P) || 0;
  const A = Number(pad?.A) || 0;
  const S = Number(pad?.S) || 0;
  const closeness = Number(rel.closeness) || 0;
  const trust = Number(rel.trust) || 0.5;
  const turn = analyzeUserTurn(t, interactionCtx);

  const notices = [];
  if (/科学|量子|神经|时间机器|实验|理论|物理|论文|假说|数据/.test(t)) {
    notices.push('话题偏学术或可被较真');
  }
  if (/难受|烦|累|郁闷|伤心|害怕|焦虑|孤独|寂寞|睡不着/.test(t)) {
    notices.push('对方情绪不太稳，语气可能比字面更重');
  }
  if (/喜欢|爱你|在乎你|需要你|想见你/.test(t)) {
    notices.push('情感话说得比较直');
  }
  if (/笨蛋|蠢|闭嘴|滚|废物|烦死/.test(t)) {
    notices.push('对方语气带刺或冒犯');
  }
  if (/谢谢|感谢|对不起|抱歉/.test(t)) {
    notices.push('对方在表态（谢或歉）');
  }
  if (turn.nicknames.length) {
    notices.push(`对方在用实验室外号叫你：${turn.nicknames.map((n) => n.label).join('、')}（指你本人，不是第三者）`);
  }
  if (companionCtx.partnerIsOkabe || turn.okabeLikely) {
    notices.push('对方是冈部（你很熟）：日常拌嘴口吻，别装陌生人');
  }
  if (/[？?]|吗$|么$|呢$|什么|怎么|为什么|如何/.test(t)) {
    notices.push('主要在问或求证');
  }
  if (t.length <= 8) notices.push('话很短，别过度发挥');
  if (!notices.length) notices.push('对方字面意思与语气');

  const withholds = [];
  if (S > 0.45 && /喜欢|爱|在乎/.test(t)) {
    withholds.push('不想让对方轻易看出自己很在意');
  }
  if (P < -0.25) withholds.push('疲惫或低落，不想多暴露脆弱');
  if (/夸|厉害|天才|好聪明|真棒/.test(t)) {
    withholds.push('不想顺着夸显得得意或好哄');
  }
  if (closeness < 0.25) withholds.push('关系还不够近，保留一点距离');
  if (!withholds.length) withholds.push('无特别要藏的事');

  let intent = INTENT_BY_BEHAVIOR[behaviorId] || INTENT_BY_BEHAVIOR.CASUAL;
  if (/科学|实验|论文/.test(t) && A > 0.2) {
    intent = '想把话说清楚，必要时纠正错误';
  }
  if (/难受|伤心|害怕/.test(t) && (S > 0.3 || closeness > 0.35)) {
    intent = '先接住对方，再给实在话；不必套安慰模板';
  }
  if (/笨蛋|滚|闭嘴/.test(t)) {
    intent = '回怼或划清界限，但别演成纯恶意';
  }
  if (companionCtx.isAutonomy === true) {
    intent = '主动开口：接他上一句或日常敲他；禁止编造实验/量子/论文，禁止无关心换题';
    if (companionCtx.lastUserAnchor) {
      notices.unshift(`他上一句还在耳边：${String(companionCtx.lastUserAnchor).slice(0, 48)}…`);
    }
  } else if (companionCtx.replyingToProactive && companionCtx.proactiveAnchor) {
    intent = '他接的是你刚才主动提起的话题；顺着聊下去，接剧情或观点，禁止指责敷衍或转移话题';
    notices.unshift('他在回应你上一轮主动说的话');
  } else if (turn.christina) {
    intent = '否定「克里斯蒂娜」整段外号，要求叫牧濑红莉栖；禁止把外号拆成「蒂娜」来反驳';
  } else if (turn.nicknames.length) {
    intent = turn.nicknames.some((n) => n.id === 'perv' || n.id === 'perv_combo')
      ? '吐槽外号或怼回去，别当成夸奖接受'
      : '反驳奇怪称呼，但不必每句都同一套怼法';
  }

  const agenda = enrichAgenda(_deriveAgenda(P, A, S, closeness, trust), {
    P,
    A,
    S,
    closeness,
    trust,
    lastUserText: t,
    isAutonomy: companionCtx.isAutonomy === true,
    replyingToProactive: companionCtx.replyingToProactive === true,
    whoamiSnippet: companionCtx.whoamiSnippet || '',
    obsSummary: companionCtx.obsSummary || '',
    idleMsSinceUser: companionCtx.idleMsSinceUser,
    lastUserAnchor: companionCtx.lastUserAnchor || '',
    lastUserText: companionCtx.isAutonomy ? (companionCtx.lastUserAnchor || t) : t,
  });

  return {
    notice: notices.slice(0, 3).join('；'),
    intent,
    withhold: withholds.slice(0, 2).join('；'),
    agenda,
    behaviorId: behaviorId || 'CASUAL',
    interactionBlock: buildInteractionPromptBlock(turn),
  };
}

function presenceToPromptLine(presence) {
  if (!presence) return '';
  const parts = [
    '【当下视角 · 内心推演，勿复述、勿写成旁白】',
    `注意到：${presence.notice}`,
    `此刻想：${presence.intent}`,
    `不想露出来：${presence.withhold}`,
    '从这三点自然开口；句式每次可以不同。禁止套用「先嘴硬再关心」「先否认再转移再软化」等固定弧线；也不必每句都吐槽或都嘴硬。',
  ];
  if (presence.agenda) {
    parts.splice(4, 0, `你带进来的：${presence.agenda}`);
  }
  if (presence.interactionBlock) parts.push(presence.interactionBlock);
  return parts.join('\n');
}

module.exports = { derivePresence, presenceToPromptLine, INTENT_BY_BEHAVIOR };
