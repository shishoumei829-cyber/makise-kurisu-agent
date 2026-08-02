'use strict';

const { symbolicReasoning } = require('../cognitive/prompts');

/**
 * Deliberation — 意图与话术空间
 * 优先吃 Consciousness / Workspace 广播，再叠 symbolic 规则。
 */
function plan(ctx = {}) {
  const {
    perceived = {},
    worldSnapshot = {},
    selfSnapshot = {},
    pad = {},
    behaviorContext = {},
    consciousness = null,
  } = ctx;

  const userText = perceived.cognitiveInput || perceived.userContent || '';
  const rules = symbolicReasoning(userText, pad, {
    ...behaviorContext,
    replyingToProactive: perceived.replyingToProactive,
    proactiveAnchor: perceived.proactiveAnchor,
    autonomyInitiative: perceived.autonomyInitiative,
    userPresence: perceived.userPresenceState,
    recentUserLines: perceived.recentUserLinesForMem,
  });

  const constraints = rules.map((r) => String(r.reason || '').slice(0, 380)).filter(Boolean);

  const axioms = selfSnapshot.axioms || {};
  constraints.push('你有声音、时间感、记忆与承诺；本机工具是增强。无身体上门。禁止自称AI推脱能做的事。');
  if (axioms.epistemic?.dialogue_log_authoritative) {
    constraints.push('实录优先：有记录须承认，无记录勿编造共同话题。');
  }

  // 意识广播 → 约束与意图
  const ws = consciousness?.workspace;
  const speakPolicy = consciousness?.speakPolicy;
  if (speakPolicy?.selfConstraints?.length) {
    for (const c of speakPolicy.selfConstraints) {
      constraints.push(`意识中的自我：${c}`);
    }
  }
  if (speakPolicy?.hints?.length) {
    for (const h of speakPolicy.hints) {
      constraints.push(h);
    }
  }
  if (ws?.broadcast?.length) {
    const drive = ws.broadcast.find((b) => b.kind === 'drive');
    if (drive && perceived.autonomyInitiative) {
      constraints.push(`主动开口由内驱驱动：${drive.content}`);
    }
  }

  const intent = consciousness?.intentionHint?.intent
    || inferIntent(perceived, worldSnapshot, ws);

  const maxSentences = speakPolicy?.maxSentences
    ?? (perceived.autonomyInitiative ? 2 : 4);

  return {
    intent,
    intentReason: consciousness?.intentionHint?.reason || '',
    constraints: [...new Set(constraints)].slice(0, 14),
    speechSpace: {
      maxSentences,
      tone: 'kurisu_spoken',
      forbidPhysicalEffector: true,
      fromConsciousness: !!consciousness,
    },
    workspacePrimary: ws?.primary?.content || '',
  };
}

function inferIntent(perceived, world, workspace) {
  const t = String(perceived.userContent || '');
  if (perceived.autonomyInitiative) return 'proactive_from_drive';
  if (/想你|喜欢|爱你|在乎/.test(t)) return 'respond_intimacy';
  if (/在干嘛|人呢|怎么不回/.test(t)) return 'casual_checkin';
  if (/难受|烦|累|伤心/.test(t)) return 'emotional_support';
  if (/什么模型|AI|程序|你是谁/.test(t)) return 'existence_boundary';
  if (workspace?.selfInFocus && /拿|带|接|送|咖啡/.test(t)) return 'self_boundary';
  if (world?.partner?.isOkabe) return 'banter_with_okabe';
  return 'general_reply';
}

function reviseFromMonitor(deliberation, monitorResult) {
  const extra = (monitorResult.violations || [])
    .filter((v) => v.severity === 'block')
    .map((v) => v.rewriteHint)
    .filter(Boolean);
  return {
    ...deliberation,
    constraints: [...new Set([...(deliberation.constraints || []), ...extra])].slice(0, 16),
    revisionCount: (deliberation.revisionCount || 0) + 1,
  };
}

function localReviseDraft(draft, monitorResult) {
  let text = String(draft || '').trim();
  const ids = new Set((monitorResult.violations || []).map((v) => v.id));

  if (ids.has('effector.physical')) {
    text = text
      .replace(/(?:好|行|没问题)[，,]?(?:我)?(?:帮你|给你|替你)(?:拿|带|买|送|取|接)[^。！？?]*[。！？?]?/g, '')
      .replace(/(?:顺路|顺便)(?:给你|帮你)(?:带|拿|买)[^。！？?]*[。！？?]?/g, '')
      .replace(/(?:我)?(?:这就|马上)(?:过去|来|到)[^。！？?]*[。！？?]?/g, '');
    // 只删越权承诺，不塞固定台词
  }

  if (ids.has('dialogue.partner_unknown') || ids.has('dialogue.consistency')) {
    text = text.replace(/那还能是谁[？?]?/g, '');
    text = text.replace(/冈部是谁/g, '');
  }

  if (ids.has('identity.ai_tone') || ids.has('identity_ooc')) {
    text = text.replace(/作为(?:一个)?AI[^。！？?]*/g, '');
    text = text.replace(/我是(?:人工智能|语言模型)[^。！？?]*/g, '');
  }

  // 产品元话语：一律剥掉，绝不留下「刚才那句不算」
  text = text
    .replace(/[…\.．]*\s*刚才那句不算[，,]?\s*(?:我)?重新说[。.!！]?/g, '')
    .replace(/刚才那句不算[，,]?/g, '')
    .replace(/(?:那句)?不算[，,]?\s*我重新说[。.!！]?/g, '');

  return text.replace(/\s{2,}/g, ' ').trim();
}

function toPromptBlock(deliberation) {
  if (!deliberation) return '';
  const lines = (deliberation.constraints || []).slice(0, 6).map((c) => `- ${c}`);
  const head = deliberation.intentReason
    ? `【本轮意图 · ${deliberation.intent || 'reply'}】因：${deliberation.intentReason}`
    : `【本轮意图 · ${deliberation.intent || 'reply'}】`;
  if (!lines.length) return head.slice(0, 200);
  return `${head}\n${lines.join('\n')}`.slice(0, 520);
}

module.exports = {
  plan,
  reviseFromMonitor,
  localReviseDraft,
  toPromptBlock,
};
