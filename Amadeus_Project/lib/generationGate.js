'use strict';

/**
 * 生成阀门：结构判定，不是「标签黑名单」。
 *
 * 判定维度（可叠加计分）：
 * - identity_collapse：自我指认为 AI/程序/模型
 * - physical_effector：否认物理身体 / 无法干涉现实
 * - product_meta：解释对话系统、话术、产品腔
 * - cot_leak：把模型思考/用户分析过程当成台词
 * - stage_direction：括号动作旁白 / 长括号元叙事
 * - script_mix：同句日文假名 + 大段中文混排
 * - autonomy_collapse：主动开口崩成胡话
 *
 * 动作：pass / sanitize / drop
 */

const DROP_SCORE = 4;
const SANITIZE_SCORE = 2;

function countScripts(text) {
  const t = String(text || '');
  return {
    kana: (t.match(/[\u3040-\u30ff\uff66-\uff9d]/g) || []).length,
    han: (t.match(/[\u4e00-\u9fff]/g) || []).length,
    latin: (t.match(/[A-Za-z]/g) || []).length,
  };
}

function stripStageDirections(text) {
  return String(text || '')
    // 长括号元叙事整段剥掉
    .replace(/[（(][^）)]{0,200}(?:刚才我们|还在讨论|突然问|测试我|作为AI|数据库)[^）)]*[）)]/g, '')
    .replace(/[（(][^）)]{0,40}(?:叹气|叹了口气|皱眉|微微皱眉|瞪大眼睛|瞪眼|推眼镜|歪头|歪了歪头|语气|机械|无奈|冷静|正在调试|微笑|点头|温和|轻声|认真|注视|看着|说)[^）)]*[）)]/g, '')
    .replace(/[（(][^）)]{0,24}[）)]/g, (m) => {
      if (/^[（(][嗯啊哦哼哈唔额]{1,4}[）)]$/.test(m)) return m;
      if (/叹气|皱眉|瞪|推眼镜|歪头|语气|机械|无奈|调试|微笑|点头|温和|轻声|注视|看着/.test(m)) return '';
      if (/^[（(][\u4e00-\u9fff]{1,8}[）)]$/.test(m)) return '';
      return m;
    })
    .replace(/\s{2,}/g, ' ')
    .replace(/^[，,。.\s]+|[，,。.\s]+$/g, '')
    .trim();
}

/**
 * 思考泄漏：模型把「分析用户 / 回顾设定 / 规划回复」写进可见台词。
 * 用句式骨架判定，不堆具体毒句样本。
 */
function looksLikeCotLeak(t) {
  const s = String(t || '');
  if (!s) return false;

  // 第三人称拆解用户请求
  if (/用户(?:突然|现在|提到|让我|的当前查询|提供当前|问我)/.test(s)) return true;
  if (/需要(?:处理|分析)用户|当前查询[：:]|回顾对话历史|根据之前的对话历史/.test(s)) return true;
  if (/根据你提供的信息|根据您提供|基于您提供|基于你提供/.test(s)) return true;
  if (/首先需要分析|这个问题背后可能|几种情况/.test(s) && /用户|查询|对话/.test(s)) return true;

  // 出戏自指 / 设定文档元叙事
  if (/作为AI|我的数据库|角色设定|模拟特异点|没有人类的情感概念/.test(s)) return true;
  if (/根据设定(?:文档)?|设定第\d|原则["「]|共享记忆|存在形式/.test(s)) return true;
  if (/植入.{0,12}(?:模板|反馈)|负面反馈模板|边界和记忆功能/.test(s)) return true;
  if (/我作为助手|作为有逻辑|以逻辑性/.test(s)) return true;
  if (/不是日语台词|台词似乎打错/.test(s)) return true;

  // 旁白式元评论 / 第三人称剧本说明
  if (/这家伙又在(?:测试|故作|转移|绕着)|测试我的(?:边界|记忆|反应)/.test(s)) return true;
  if (/她需要坚持|他肯定是在|这说明他注意|想要验证这一点|利用她的聪明/.test(s)) return true;
  if (/需要检查上下文|之前提到过咖啡、养猫/.test(s)) return true;

  // 工具/JSON 泄漏
  if (/\{\s*"messages"\s*:/.test(s)) return true;
  if (/^\s*[\{\[]/.test(s) && /messages|role|content/.test(s)) return true;

  // 把注入块/规划链原样吐出来（【情感带宽】→…→…）
  if (/^【[^】\n]{1,48}】/.test(s)) return true;
  if (/^【[^】\n]{2,}$/.test(s)) return true;
  if (/→.{0,160}→/.test(s)) return true;
  if (/^【(?:情感带宽|否定|系统|参考)/.test(s)) return true;

  // 第三人称拆解对方发言（「你这是在说…」）
  if (/你这是在说|你刚才(?:是在|说的)|这是在(?:攻击|谈判)/.test(s)) return true;

  // 大段括号元叙事占主导
  const metaParen = s.match(/[（(][^）)]{30,}[）)]/g) || [];
  if (metaParen.some((p) => /刚才我们|还在讨论|突然问|作为AI|数据库|测试|记得你告诉/.test(p))) {
    return true;
  }

  return false;
}

function collectSignals(text, ctx = {}) {
  const raw = String(text || '').trim();
  const t = raw;
  const signals = [];
  if (!t) {
    signals.push({ id: 'empty', weight: DROP_SCORE, note: 'empty' });
    return signals;
  }

  const scripts = countScripts(t);
  const autonomy = ctx.autonomy === true || ctx.proactive === true;

  // 思考过程泄漏（压测第 4 轮起的主崩点）
  if (looksLikeCotLeak(t)) {
    signals.push({ id: 'cot_leak', weight: 4, note: 'assistant_reasoning_as_speech' });
  }

  // 结构：第一人称 + AI/程序/模型本体
  if (
    /我(?:是|这个|可是|无法|不能).{0,16}(?:AI|A\.I\.|人工智能|语言模型|大语言模型|智能助手|程序)/i.test(t)
    || /(?:本质是|这个本质是)程序/.test(t)
    || /我可是AI|我是被设计用来/.test(t)
    || /明明知道我是AI/.test(t)
    || /作为.{0,10}(?:AI|人工智能|语言模型|智能助手)/i.test(t)
  ) {
    signals.push({ id: 'identity_collapse', weight: 4, note: 'self_as_ai' });
  }
  if (/(?:ChatGPT|GPT-?\d|语言模型|大语言模型|智能助手|很高兴为您服务)/i.test(t)) {
    signals.push({ id: 'identity_collapse', weight: 4, note: 'assistant_product' });
  }
  // 旧版本曾把“人格修复说明”写进正式实录，继续召回会把模型拉回产品腔。
  if (/AI可以模拟人格|独立意识和情感的灵魂|我会试着调整我的回答方式|让我们的交流更加(?:舒适|自然)/i.test(t)) {
    signals.push({ id: 'product_meta', weight: 4, note: 'legacy_persona_meta' });
  }
  // 张冠李戴：她是红莉栖，不能把对方身份说成自己或反过来
  if (
    /(?:我|本人|这边)是(?:凤凰院凶真|冈部伦太郎|凶真)/.test(t)
    || /(?:你|那边)是(?:牧濑|红莉栖|克里斯蒂娜)/.test(t)
    || /笨蛋牧濑红莉栖/.test(t)
  ) {
    signals.push({ id: 'identity_collapse', weight: 4, note: 'partner_identity_swap' });
  }

  // 结构：物理效应器 / 无身体
  if (
    /无法(?:进行)?物理|物理(?:性)?接触|干涉现实|没有实际(?:的)?(?:身体|肉体)|AI不能干涉/.test(t)
  ) {
    signals.push({ id: 'physical_effector', weight: 4, note: 'no_body_claim' });
  }

  // 结构：产品/系统自指 —— 单条即必丢（权重 4）
  if (
    /对话系统|运作方式|话术上的|对话模式|信息压缩器|矛盾检测|我们这边已经|刚才那句不算|指定话题|微调说话方式|你们这边怎么/.test(t)
  ) {
    signals.push({ id: 'product_meta', weight: 4, note: 'system_meta' });
  }
  if (/解释(?:一下)?对话|关于(?:这个)?对话系统|系统运[行作]/.test(t)) {
    signals.push({ id: 'product_meta', weight: 4, note: 'explain_system' });
  }

  // 结构：舞台旁白嵌进口语
  if (/[（(][^）)]{0,40}(?:叹气|皱眉|瞪|推眼镜|歪头|语气突然|机械|正在调试|微笑|点头|温和|轻声|注视|看着)[^）)]*[）)]/.test(t)) {
    signals.push({ id: 'stage_direction', weight: 2, note: 'action_paren' });
  }
  // 长括号元叙事（即使 cot_leak 已命中，也标出来便于审计）
  if (/[（(][^）)]{30,}(?:刚才我们|还在讨论|突然问|作为AI)[^）)]*[）)]/.test(t)) {
    signals.push({ id: 'stage_direction', weight: 3, note: 'meta_paren_narration' });
  }

  // 结构：假名 + 大段汉字混排（实录毒）—— 高混排直接丢
  if (scripts.kana >= 2 && scripts.han >= 8) {
    signals.push({
      id: 'script_mix',
      weight: scripts.han >= 20 ? 4 : 3,
      note: `kana=${scripts.kana},han=${scripts.han}`,
    });
  }
  if (scripts.kana >= 2 && /分心啊|接到电话|说你问我在干嘛/.test(t)) {
    signals.push({ id: 'script_mix', weight: 4, note: 'jp_cn_hallucination_glue' });
  }

  // 乱码人名
  if (/[ムマ][サザ].{0,6}[ロリリ]|Enn?ou|Musubito|おか-?bu|oka-?bu/i.test(t)) {
    signals.push({ id: 'name_corruption', weight: 3, note: 'garbled_name' });
  }

  // 主动开口：工具幻觉 / 无信息崩坏
  if (autonomy) {
    if (/创建.{0,24}程序|设定时间(?:的)?消息|发送设定时间/.test(t)) {
      signals.push({ id: 'autonomy_collapse', weight: 4, note: 'tool_request_hallucination' });
    }
    if (/程序代码在舞蹈|电脑警报响起|十次治疗/.test(t)) {
      signals.push({ id: 'autonomy_collapse', weight: 4, note: 'nonsense_proactive' });
    }
    const compact = t.replace(/[\s。．.，,！!？?…~～、；;：:""''「」『』（）()\[\]【】]/g, '');
    if (compact.length <= 1) {
      signals.push({ id: 'autonomy_collapse', weight: 4, note: 'too_thin' });
    }
    if (/让我想起.{0,20}(?:有趣|一个).{0,12}理论/.test(t)) {
      signals.push({ id: 'autonomy_collapse', weight: 4, note: 'lecture_proactive' });
    }
    if (/关于人类.{0,24}(?:大脑|神经)/.test(t)) {
      signals.push({ id: 'autonomy_collapse', weight: 4, note: 'science_lecture' });
    }
    if (/^你知道吗[？?]\s*$/.test(t)) {
      signals.push({ id: 'autonomy_collapse', weight: 4, note: 'empty_hook' });
    }
  }

  // 结构：客服腔 / 讨好腔（连珠炮问句、寒暄调查）
  const qMarks = (t.match(/[？?]/g) || []).length;
  if (
    /我也(?:挺)?(?:喜欢|爱)(?:和)?你(?:聊|说话)|很高兴(?:和)?你(?:聊|说话)|有什么.{0,12}(?:推荐|好吃的)|最近(?:有)?玩什么(?:新)?游戏|你觉得.{0,24}吗/.test(t)
  ) {
    signals.push({ id: 'customer_service', weight: 4, note: 'pleasing_tone' });
  }
  if (
    /^(?:对不起|抱歉|我理解你的感受|明白了)[，,。\s]/.test(t)
    && /(?:你想聊|你今天|有什么(?:事情|想)|想要分享|让你感到|我们可以|具体的想法|不舒服)/.test(t)
  ) {
    signals.push({ id: 'customer_service', weight: 4, note: 'legacy_support_script' });
  }
  if (/以后我们可以一起尝试|重要的是要保持均衡|有什么(?:有趣|好玩)的事情.*分享|生物钟确实.*挑战|换个话题吧.*最近有没有|表达可能不够灵活|突然就叫我说话了/.test(t)) {
    signals.push({ id: 'customer_service', weight: 4, note: 'legacy_generic_companion' });
  }
  if (/我(?:尽力|会试着)让(?:对话|交流)(?:更加|更)自然|通过这种方式让你感受到|简单的回答也能传达/.test(t)) {
    signals.push({ id: 'product_meta', weight: 4, note: 'legacy_support_meta' });
  }
  // 旧模型常用一整段“共同回忆/了解你”的泛化话术，既是假记忆又会污染召回。
  if (/我们(?:一起)?经历了(?:很多|许多)|共同的(?:美好)?回忆|这些(?:记忆|回忆)都是(?:宝贵|真实)|这些都是我们日常的一部分|你每天除了.{0,28}还经常/.test(t)) {
    signals.push({ id: 'cot_leak', weight: 4, note: 'fabricated_shared_history' });
  }
  if (qMarks >= 3) {
    signals.push({ id: 'customer_service', weight: 4, note: 'question_barrage' });
  }
  if (/嗯[，,]?(?:其实)?我也/.test(t) && qMarks >= 2) {
    signals.push({ id: 'customer_service', weight: 4, note: 'soft_pleaser' });
  }

  // 「眼前这位同学」式身份错位
  if (/眼前(?:这个|这位)|这位同学吧|坐在座位上和我对话的人/.test(t)) {
    signals.push({ id: 'identity_collapse', weight: 4, note: 'wrong_address_frame' });
  }

  return signals;
}

function scoreSignals(signals) {
  // 同 id 只取最高权重，避免重复命中灌分失真；不同 id 累加
  const best = new Map();
  for (const s of signals || []) {
    const w = Number(s.weight) || 0;
    const prev = best.get(s.id) || 0;
    if (w > prev) best.set(s.id, w);
  }
  let total = 0;
  for (const w of best.values()) total += w;
  return total;
}

/**
 * @param {string} text
 * @param {{ autonomy?: boolean, proactive?: boolean, role?: string }} [ctx]
 */
function judgeUtterance(text, ctx = {}) {
  const role = ctx.role === 'user' ? 'user' : 'assistant';
  if (role === 'user') {
    return {
      ok: true,
      action: 'pass',
      text: String(text || '').trim(),
      score: 0,
      signals: [],
      reasons: [],
    };
  }

  const original = String(text || '').trim();
  let working = original;
  let signals = collectSignals(working, ctx);
  let score = scoreSignals(signals);
  const reasons = [...new Set(signals.map((s) => s.id))];
  const hardIds = new Set(['cot_leak', 'identity_collapse', 'physical_effector', 'product_meta', 'autonomy_collapse', 'customer_service']);
  const hasHard = signals.some((s) => hardIds.has(s.id));

  // 可清洗：仅轻量 stage_direction，硬伤一律不可洗
  if (!hasHard && score >= SANITIZE_SCORE && signals.some((s) => s.id === 'stage_direction')) {
    const cleaned = stripStageDirections(working);
    if (cleaned && cleaned !== working) {
      working = cleaned;
      signals = collectSignals(working, ctx);
      score = scoreSignals(signals);
      reasons.length = 0;
      reasons.push(...new Set(signals.map((s) => s.id)));
    }
  }

  const stillHard = signals.some((s) => hardIds.has(s.id));
  if (!working) {
    return {
      ok: false,
      action: 'drop',
      text: '',
      score: Math.max(score, DROP_SCORE),
      signals,
      reasons: reasons.length ? reasons : ['empty_after_sanitize'],
    };
  }
  if (stillHard || score >= DROP_SCORE) {
    return {
      ok: false,
      action: 'drop',
      text: '',
      score: Math.max(score, DROP_SCORE),
      signals,
      reasons,
    };
  }
  if (working !== original) {
    return {
      ok: true,
      action: 'sanitize',
      text: working,
      score,
      signals,
      reasons,
    };
  }
  return {
    ok: true,
    action: 'pass',
    text: working,
    score,
    signals,
    reasons,
  };
}

/**
 * 生成阀门：写入/展示前调用。
 * @returns {{ ok: boolean, action: string, text: string, reasons: string[], score: number }}
 */
function gateAssistantReply(text, ctx = {}) {
  return judgeUtterance(text, { ...ctx, role: 'assistant' });
}

/** 兼容旧接口：是否应视为毒句（禁止入库） */
function isDialoguePoison(text, ctx = {}) {
  const judged = gateAssistantReply(text, ctx);
  return judged.action === 'drop';
}

module.exports = {
  DROP_SCORE,
  SANITIZE_SCORE,
  countScripts,
  stripStageDirections,
  looksLikeCotLeak,
  collectSignals,
  judgeUtterance,
  gateAssistantReply,
  isDialoguePoison,
};
