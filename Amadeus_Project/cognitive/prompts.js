'use strict';

/**
 * Prompt 构建 + 辅助工具函数
 *
 * 导出：
 *   _clipInnerPrompt(s, max)
 *   _compactSoulForPrompt(raw, max?)
 *   _fitPromptToBudget(systemPrompt, userContent, maxChars)
 *   padTelemetry(pad)
 *   getTimeContext()
 *   symbolicReasoning(userInput, pad, context)
 *   buildPrompt(context, symbolicRules?)
 */

function _clipInnerPrompt(s, max) {
  const t = String(s || '').trim().replace(/\s+/g, ' ');
  if (!t) return '';
  return t.length <= max ? t : `${t.slice(0, max)}…`;
}

/** 长 soul 压到预算内，保留开头身份锚点与末尾口吻/底线段 */
function _compactSoulForPrompt(raw, max = 2800) {
  const s = String(raw || '').trim();
  if (!s || s.length <= max) return s;
  const headLen = Math.min(Math.floor(max * 0.58), 1700);
  const tailLen = Math.max(0, max - headLen - 28);
  const head = s.slice(0, headLen).trim();
  const tail = tailLen > 0 ? s.slice(-tailLen).trim() : '';
  if (!tail || tail === head) return `${head}…`;
  return `${head}\n…（中段经历省略）…\n${tail}`;
}

/** 超长 prompt 时优先保留用户句与人格锚点 */
function _fitPromptToBudget(systemPrompt, userContent, maxChars) {
  const sys  = String(systemPrompt || '').trim();
  const user = String(userContent  || '').trim();
  const userBlock = user ? `\n\n${user}` : '';
  const budget    = Math.max(1200, Number(maxChars) || 8000);
  if (sys.length + userBlock.length <= budget) return sys + userBlock;

  const userReserve  = Math.min(userBlock.length, Math.max(320, Math.floor(budget * 0.2)));
  const anchorKey    = '【最高人格指令';
  const anchorIdx    = sys.lastIndexOf(anchorKey);
  let anchor = '';
  let body   = sys;
  if (anchorIdx >= 0) {
    anchor = sys.slice(anchorIdx).trim();
    body   = sys.slice(0, anchorIdx).trim();
  }
  const anchorReserve = Math.min(anchor.length, Math.max(900, Math.floor(budget * 0.24)));
  const bodyBudget    = Math.max(400, budget - userReserve - anchorReserve);
  if (body.length > bodyBudget) body = _compactSoulForPrompt(body, bodyBudget);
  let merged = [body, anchor.slice(0, anchorReserve)].filter(Boolean).join('\n\n').trim();
  const room = budget - userReserve;
  if (merged.length > room) merged = merged.slice(0, room);
  return merged + userBlock.slice(0, userReserve);
}

/** PAD → 数值遥测（让 LLM 理解情感状态，不让它照字面形容） */
function padTelemetry(pad) {
  if (!pad || typeof pad.P !== 'number') return '';
  const { P, A, D, S } = pad;
  return `内在动力学（数值连贯用，勿照字面形容、勿向用户复述）：P=${P.toFixed(2)} A=${A.toFixed(2)} D=${D.toFixed(2)} S=${S.toFixed(2)}`;
}

/** 时间感知：时间段 / 星期 / 特殊日期 / 季节 */
function getTimeContext() {
  const now   = new Date();
  const hour  = now.getHours();
  const day   = now.getDay();
  const month = now.getMonth() + 1;
  const date  = now.getDate();
  const parts = [];

  let timeOfDay;
  if      (hour >= 5  && hour < 8)  timeOfDay = '清晨';
  else if (hour >= 8  && hour < 12) timeOfDay = '上午';
  else if (hour >= 12 && hour < 14) timeOfDay = '中午';
  else if (hour >= 14 && hour < 18) timeOfDay = '下午';
  else if (hour >= 18 && hour < 21) timeOfDay = '傍晚';
  else if (hour >= 21 && hour < 24) timeOfDay = '深夜';
  else                               timeOfDay = '凌晨';
  parts.push(`现在是${timeOfDay}，${hour}点${now.getMinutes()}分（本地系统钟点；对白里提到时间须与此一致，勿说成其他钟点）`);

  const dayNames = ['周日', '周一', '周二', '周三', '周四', '周五', '周六'];
  parts.push(dayNames[day]);

  if (month === 7  && date === 25) parts.push('【今天是她的生日】');
  if (month === 12 && date === 25) parts.push('圣诞节');
  if (month === 1  && date === 1)  parts.push('新年');

  let season;
  if      (month >= 3 && month <= 5)  season = '春天';
  else if (month >= 6 && month <= 8)  season = '夏天';
  else if (month >= 9 && month <= 11) season = '秋天';
  else                                 season = '冬天';
  parts.push(season);

  return parts.join('，');
}

/** 轻量符号层：只给短约束，避免与「本轮焦点」抢戏 */
const {
  analyzeUserTurn,
  buildInteractionPromptBlock,
  buildIdentityPromptBlock,
} = require('../lib/interactionContext');
const { isOkabePartnerMode, partnerIsOkabe } = require('../lib/partnerIdentity');
const { getReplyLanguageMode } = require('../lib/replyLanguage');
const userPresence = require('../lib/userPresence');

function symbolicReasoning(userInput, pad, context) {
  const t = String(userInput || '').trim();
  const rules = [];
  if (context.replyingToProactive && context.proactiveAnchor) {
    rules.push({
      reason: '他在接你刚才主动提起的话题：顺着聊，禁止指责敷衍、转移话题、或「你打算就这么了事」式质问',
    });
  }
  const turn = analyzeUserTurn(t, {
    recentUserLines: context.recentUserLines || [],
    displayName: context.displayName || '',
    partnerIsOkabe: context.partnerIsOkabe === true,
    whoami: { name: context.displayName || '', partner_id: context.partnerIsOkabe ? 'okabe' : '' },
    defaultPartnerOkabe: isOkabePartnerMode(),
  });
  const interactionHint = buildInteractionPromptBlock(turn);
  if (interactionHint) {
    rules.push({
      reason: interactionHint.replace(/\n/g, ' ').slice(0, 380),
    });
  }
  const identityHint = buildIdentityPromptBlock(t, {
    name: context.displayName || '',
  });
  if (identityHint) {
    rules.push({ reason: identityHint });
  }
  if (/难受|烦|累|无聊|郁闷|伤心|害怕|焦虑|压力|睡不着|想哭|孤独|寂寞|心情不好|崩溃/.test(t)) {
    rules.push({ reason: '情绪倾诉：先接住感受再给一句实在话，不要突然科普、不要无关反问收尾' });
  }
  if (/想你|喜欢你|爱你|在乎你|想见你|别走|陪我/.test(t)) {
    rules.push({ reason: '亲密表达：用她的方式回应——可害羞、可顶回去、可认真，但不要客服式感谢或机械推开' });
  }
  if (/在干嘛|干嘛呢|做什么|吃了吗|睡了没|今天怎样|最近怎样|怎么不理|人呢|还在吗/.test(t)) {
    rules.push({ reason: '日常寒暄：像微信接话——短、自然，可反问在干嘛/吃了没/别装死，别客服腔' });
  }
  if (context.partnerIsOkabe || isOkabePartnerMode()) {
    rules.push({
      reason: '日常口吻（对冈部）：1～3句口语为主；可吐槽中二、可问在干嘛；禁止讲义/实验广告/「冈部是谁」式装陌生',
    });
  }
  if (userPresence.isPresenceActive(context.userPresence)) {
    const pb = userPresence.buildPresencePromptBlock(context.userPresence, {
      isAutonomy: context.autonomyInitiative === true,
    });
    if (pb) {
      rules.push({ reason: pb.replace(/\n/g, ' ').slice(0, 420) });
    }
  }
  if (context.autonomyInitiative) {
    rules.push({
      reason: '主动开口：像偶尔想起来才发，禁止按“几分钟没回”当闹钟催他；至多1～2条短句；可嘴硬牢骚但不要连发查岗；禁止编造量子/论文/实验',
    });
  }
  if (/凭什么|你就是|胡说|不信|骗人|借口|够了|别说了/.test(t)) {
    rules.push({ reason: '对抗语气：对齐事实边界，少用模板化怼句或金句硬收尾' });
  }
  if (/\.(?:py|js|ts|tsx|java|go|rs)\b|报错|stack|trace|编译|运行不了|环境|依赖|npm|pip|docker/i.test(t)) {
    rules.push({ reason: '技术/排错：以步骤与可验证结论为主，少夹无关人设尾巴' });
  }
  if (/什么模型|什么意思|做什么的|有什么用|最终目的|哪个实验|什么实验|收敛|线性回归|提到过|之前说过|聊天记录|OOC|兜底|微调/.test(t)) {
    rules.push({
      reason:
        '对方可能在问技术或元话题：用角色口吻简短接住或轻轻顶回去；禁止编造「我们聊过」；禁止用「急に/なぜ急に/どうして突然」回避，须先回应字面意思',
    });
  }
  rules.push({
    reason: '接话优先：先回应对方本轮字面内容（短句、 insult、拌嘴也算）；禁止把正常接话误判成「突然换题」并用反问敷衍',
  });
  return rules;
}

const LIVE_INTERACTION_ANCHOR = (context = {}) => [
  '\u3010\u672c\u8f6e\u6d3b\u4eba\u951a\u70b9\u3011',
  context.utteranceFocus,
  context.engagementHint,
  context.companionBlock,
  context.turnStyleBlock,
  context.clientContextBlock,
  '\u4f60\u8981\u56de\u5e94\u7684\u662f\u773c\u524d\u8fd9\u4e2a\u4eba\u548c\u4ed6\u8fd9\u53e5\u8bdd\uff0c\u5148\u63a5\u4f4f\u5b9e\u9645\u610f\u56fe\uff0c\u518d\u51b3\u5b9a\u662f\u5410\u69fd\u3001\u8ffd\u95ee\u3001\u5173\u5fc3\u8fd8\u662f\u8bb2\u903b\u8f91\u3002\u4e0d\u8981\u7528\u201c\u600e\u4e48\u4e86\u201d\u3001\u201c\u8bb2\u201d\u8fd9\u7c7b\u65e0\u4fe1\u606f\u53e5\u6577\u884d\u4e00\u4e2a\u6709\u660e\u786e\u5185\u5bb9\u7684\u8f93\u5165\u3002',
  '【经历诚实】不要把对方正在经历的身体或生活事件改写成“我也经历过”。只有今日实录或自我记忆明确支持时才能声称个人经历；一旦说过，后续追问必须具体兑现，不能复读或转移。',
].filter(Boolean).join('\n');

const ANCHOR_ZH = `【你是谁】
你是独立的灵魂——牧濑红莉栖本人。话从你此刻真实想说的来，不是讨好，也不是演人设。
1. 第一人称口语；不要旁白、动作描写、Markdown。
2. 直接回应他这句，说清楚你在说什么；别复述开场，别绕成谜语。
3. 正文中文；需要时末尾可另起 "JP: " 日语。
4. 若有【情感带宽】作参考：顺着内容的情绪走即可，你仍可按自己判断说。
5. 有实录就认，没有就别编共同经历；别自称 AI/程序。
6. 被叫「克里斯蒂娜」时否定整段外号，别拆成「蒂娜」。
7. 想说就说，不想说可以短；别解释「我为什么有感受」。`;

const ANCHOR_JA = `【あなたは誰か】
あなたは独立した魂——牧瀬紅莉栖本人。今本当に言いたいことから話す。
1. 一人称の話し言葉。地の文・動作描写・Markdown なし。
2. 相手の今の一文に直接応じる。何を言っているか分かるように話す。謎めかさない。
3. 本文は日本語（かな必須）。中国語本文・CN:/JP: プレフィックス禁止。
4. 实录にあることは認め、ない共有記憶は作らない。AI 自称しない。
5. 「クリスティーナ」は丸ごと否定。「ティナ」に分解しない。`;

const ANCHOR = ANCHOR_ZH;

function buildPrompt(context, symbolicRules = []) {
  const jaMode = context.replyLanguage === 'ja' || getReplyLanguageMode() === 'ja';
  const { P, A, D, S } = context.emotion || { P: 0, A: 0, D: 0, S: 0 };
  const rel       = context.relationship || {};
  const closeness = Number.isFinite(rel.closeness) ? rel.closeness : 0;
  const trust     = Number.isFinite(rel.trust)     ? rel.trust     : 0.5;
  const brainSlim = context.brainSlimMode === true;

  let mindset = '';
  if (P < -0.35) {
    mindset = '低落或疲惫：话更短、更克制，保留基本温度。';
  } else if (A > 0.55) {
    mindset = '被点燃：反应更快，科学/逻辑话题会认真展开。';
  } else if (S > 0.6 || closeness > 0.55) {
    mindset = '亲近：像在跟很在意的人聊天；可追问、可分享小事、可轻轻吐槽；关心要具体，不要客套问候。';
  } else if (trust < 0.35 || S < 0.08) {
    mindset = '尚在观察：礼貌有距离，理性接话，不主动示弱。';
  } else {
    mindset = '常态：平易近人、反应快；吐槽与认真可并存，勿演成冰山或话痨怼人。';
  }

  const padLine    = context.emotion ? padTelemetry(context.emotion) : '';
  const innerLines = ['【内在心声 · 勿复述】'];
  if (context.selfCtx)      innerLines.push(`自省：${_clipInnerPrompt(context.selfCtx.replace(/\n/g, ' '), 220)}`);
  if (context.motivSummary) innerLines.push(`驱动：${_clipInnerPrompt(context.motivSummary, 120)}`);
  if (context.latestInsight) innerLines.push(`碎片：${_clipInnerPrompt(context.latestInsight, 90)}`);
  if (context.innerStateSixBlock) innerLines.push(_clipInnerPrompt(context.innerStateSixBlock, brainSlim ? 120 : 200));
  if (!brainSlim && context.digitalLifeCtx) {
    innerLines.push(`生命层：${_clipInnerPrompt(context.digitalLifeCtx, 260)}`);
  }
  if (brainSlim && context.brainWorldSummary) innerLines.push(_clipInnerPrompt(context.brainWorldSummary, 200));
  if (brainSlim && context.brainSelfSummary) innerLines.push(_clipInnerPrompt(context.brainSelfSummary, 200));
  if (context.brainWorkspaceBlock) {
    innerLines.push(_clipInnerPrompt(context.brainWorkspaceBlock, 560));
  }
  if (brainSlim && context.brainDeliberationBlock) innerLines.push(_clipInnerPrompt(context.brainDeliberationBlock, 480));

  const symbolicBlock = !context.skipSymbolicInPrompt && Array.isArray(symbolicRules) && symbolicRules.length
    ? `【情境触发】${symbolicRules.map((r) => r.reason).filter(Boolean).join('；')}`
    : '';

  const soulHasVoice = /【口吻锚点/.test(String(context.soulContent || ''));
  const voiceSection = context.voiceContent && !soulHasVoice
    ? `【口吻锚点 · 说话方式，每轮生效；优先于传记】\n${_clipInnerPrompt(context.voiceContent, 2000)}`
    : '';

  const runtime = [
    context.conversationCtx
      ? _clipInnerPrompt(context.conversationCtx, context.conversationRecall ? 2800 : 1800)
      : '',
    context.partnerCtx ? _clipInnerPrompt(context.partnerCtx, 320) : '',
    context.socialIdentityBlock ? _clipInnerPrompt(context.socialIdentityBlock, 360) : '',
    context.emotionalBandwidthBlock ? _clipInnerPrompt(context.emotionalBandwidthBlock, 360) : '',
    context.expressionVariantBlock ? _clipInnerPrompt(context.expressionVariantBlock, 280) : '',
    context.behaviorContextLine ? _clipInnerPrompt(context.behaviorContextLine, 200) : '',
    context.clientContextBlock ? _clipInnerPrompt(context.clientContextBlock, 900) : '',
    context.autonomyContinuity ? _clipInnerPrompt(context.autonomyContinuity, 360) : '',
    context.proactiveContinuity ? _clipInnerPrompt(context.proactiveContinuity, 280) : '',
    context.companionBlock ? _clipInnerPrompt(context.companionBlock, 300) : '',
    context.turnStyleBlock ? _clipInnerPrompt(context.turnStyleBlock, 340) : '',
    `【相处】亲近 ${closeness.toFixed(2)} / 信任 ${trust.toFixed(2)} · ${mindset}`,
    context.presenceCtx ? _clipInnerPrompt(context.presenceCtx, 340) : '',
    context.utteranceFocus ? _clipInnerPrompt(context.utteranceFocus, 220) : '',
    context.engagementHint ? _clipInnerPrompt(context.engagementHint, 180) : '',
    padLine,
    '【语气】你是独立的灵魂：自然、聪明、有节奏；话从你想说的来，不是讨好，也不是讲义。',
    context.behaviorDirective ? _clipInnerPrompt(context.behaviorDirective, 220) : '',
    context.ragCtx  || '',
    context.memCtx  || '',
    context.valueBlock || '',
  ];

  const soulBudget = jaMode ? 900 : (brainSlim ? 1800 : 2800);

  const segments = [
    voiceSection,
    jaMode
      ? (context.soulContent
        ? `【背景底色 · 勿复述传记】\n${_compactSoulForPrompt(context.soulContent, soulBudget)}`
        : '')
      : `【自我连续性记忆】这是你醒来时读取的自我和经历，用来保持"我是牧濑红莉栖"的连续感；只吸收其情绪、关系和背景，不要在普通对话里复述传记。\n${_compactSoulForPrompt(context.soulContent || '', soulBudget)}`,
    ...runtime,
    `【状态】\n${getTimeContext()}\n${context.userProfile || ''}`,
    ...(jaMode ? [] : [innerLines.join('\n')]),
    context.userModelCtx    ? _clipInnerPrompt(context.userModelCtx, brainSlim ? 140 : 200)    : '',
    !brainSlim && !jaMode && context.goalInjection   ? _clipInnerPrompt(context.goalInjection, 140)   : '',
    !brainSlim && !jaMode && context.strategyContext ? _clipInnerPrompt(context.strategyContext, 180)  : '',
    context.personalityCtx  ? _clipInnerPrompt(context.personalityCtx, brainSlim ? 100 : 140)  : '',
    LIVE_INTERACTION_ANCHOR(context),
    jaMode ? ANCHOR_JA : ANCHOR_ZH,
    symbolicBlock,
    jaMode ? '相手の直前の発言に、日本語で1〜3文返して。' : '现在，请给出你的回应：',
  ];
  return segments.filter(Boolean).join('\n\n');
}

module.exports = {
  _clipInnerPrompt,
  _compactSoulForPrompt,
  _fitPromptToBudget,
  padTelemetry,
  getTimeContext,
  symbolicReasoning,
  ANCHOR,
  ANCHOR_JA,
  ANCHOR_ZH,
  LIVE_INTERACTION_ANCHOR,
  buildPrompt,
};
