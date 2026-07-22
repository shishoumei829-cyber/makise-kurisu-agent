'use strict';

/**
 * Global Workspace — 全局工作空间（意识工程核心）
 *
 * 工程近似（Global Workspace Theory）：
 * 潜意识候选竞争 → 少数内容进入「意识广播」→ 全认知回路共用。
 *
 * 不声称主观体验；只定义：她此刻「意识到」什么、什么驱动说话。
 */

const MAX_BROADCAST = 5;
const MIN_SALIENCE = 0.28;

const CONTENT_KINDS = Object.freeze({
  PERCEPT: 'percept',
  AFFECT: 'affect',
  DRIVE: 'drive',
  SELF: 'self',
  RELATION: 'relation',
  MEMORY: 'memory',
  META: 'meta',
  INTENTION: 'intention',
});

function clamp01(v) {
  return Math.max(0, Math.min(1, Number(v) || 0));
}

function makeItem(partial) {
  return {
    id: partial.id || `ws_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`,
    kind: partial.kind,
    content: String(partial.content || '').trim().slice(0, 180),
    salience: clamp01(partial.salience ?? 0.5),
    source: partial.source || 'unknown',
    actionable: partial.actionable === true,
    speakHint: partial.speakHint ? String(partial.speakHint).slice(0, 120) : '',
    meta: partial.meta || {},
  };
}

/**
 * 从感知/世界/自我/内驱采集候选（潜意识池）
 */
function collectCandidates(ctx = {}) {
  const {
    perceived = {},
    worldSnapshot = {},
    selfSnapshot = {},
    pad = {},
    digitalLifeTurn = null,
    autonomyPublic = null,
    motivationState = {},
    butler = null,
  } = ctx;

  const items = [];
  const userText = String(perceived.cognitiveInput || perceived.userContent || '').trim();

  // ── 感知：用户话焦点 ──
  if (userText) {
    let salience = 0.72;
    let content = `他在说：${userText.slice(0, 80)}`;
    if (/想你|喜欢你|爱你|在乎/.test(userText)) {
      salience = 0.92;
      content = `亲密表达：${userText.slice(0, 60)}`;
    } else if (/难受|烦|累|伤心|孤独|睡不着/.test(userText)) {
      salience = 0.9;
      content = `他情绪不好：${userText.slice(0, 60)}`;
    } else if (/在干嘛|人呢|怎么不回|忙吗/.test(userText)) {
      salience = 0.78;
    } else if (/什么模型|AI|程序|你是谁|Amadeus/.test(userText)) {
      salience = 0.88;
      content = `存在/身份追问：${userText.slice(0, 60)}`;
    }
    items.push(makeItem({
      kind: CONTENT_KINDS.PERCEPT,
      content,
      salience,
      source: 'perceive',
      actionable: true,
    }));
  }

  if (perceived.replyingToProactive && perceived.proactiveAnchor) {
    items.push(makeItem({
      kind: CONTENT_KINDS.PERCEPT,
      content: `他在接我刚才主动说的：${String(perceived.proactiveAnchor).slice(0, 70)}`,
      salience: 0.8,
      source: 'thread',
      actionable: true,
    }));
  }

  // ── 情感：PAD ──
  const P = Number(pad.P) || 0;
  const A = Number(pad.A) || 0;
  const S = Number(pad.S) || 0;
  if (P < -0.3) {
    items.push(makeItem({
      kind: CONTENT_KINDS.AFFECT,
      content: '心情偏低，话可能更短、更克制',
      salience: 0.55 + Math.min(0.3, Math.abs(P) * 0.4),
      source: 'pad',
    }));
  } else if (P > 0.35) {
    items.push(makeItem({
      kind: CONTENT_KINDS.AFFECT,
      content: '情绪偏正向，接话可以更轻松',
      salience: 0.45 + Math.min(0.25, P * 0.3),
      source: 'pad',
    }));
  }
  if (A > 0.55) {
    items.push(makeItem({
      kind: CONTENT_KINDS.AFFECT,
      content: '唤醒度高，容易认真展开或嘴快',
      salience: 0.5 + (A - 0.55),
      source: 'pad',
    }));
  }

  // ── 关系 ──
  const rel = worldSnapshot.relationship?.score ?? S;
  const partnerName = worldSnapshot.partner?.name || '';
  if (worldSnapshot.partner?.isOkabe) {
    items.push(makeItem({
      kind: CONTENT_KINDS.RELATION,
      content: `在和${partnerName || '冈部'}说话——熟人拌嘴，不是陌生人`,
      salience: 0.55 + Math.min(0.25, Math.max(0, rel) * 0.3),
      source: 'whoami',
    }));
  } else if (partnerName) {
    items.push(makeItem({
      kind: CONTENT_KINDS.RELATION,
      content: `正在和${partnerName}对话`,
      salience: 0.5,
      source: 'whoami',
    }));
  }
  if (rel > 0.45) {
    items.push(makeItem({
      kind: CONTENT_KINDS.RELATION,
      content: '关系已经比较近，关心可以具体一点',
      salience: 0.48 + Math.min(0.3, rel * 0.3),
      source: 'relationship',
    }));
  }

  // ── 视觉 / 情境 ──
  if (worldSnapshot.vision?.active && worldSnapshot.vision.description) {
    items.push(makeItem({
      kind: CONTENT_KINDS.PERCEPT,
      content: `视线里：${String(worldSnapshot.vision.description).slice(0, 70)}`,
      salience: /看到|看|画面|屏幕|摄像头/.test(userText) ? 0.85 : 0.42,
      source: 'vision',
      actionable: /看到|看|画面|屏幕|摄像头/.test(userText),
    }));
  }
  if (worldSnapshot.clientContext?.situation) {
    items.push(makeItem({
      kind: CONTENT_KINDS.PERCEPT,
      content: `情境：${String(worldSnapshot.clientContext.situation).slice(0, 70)}`,
      salience: 0.4,
      source: 'situation',
    }));
  }

  // ── 自我公理 / 张力（意识中的「我」）──
  const axioms = selfSnapshot.axioms || {};
  const tensions = selfSnapshot.tensions || {};
  items.push(makeItem({
    kind: CONTENT_KINDS.SELF,
    content: '我通过对话存在；影响域是言语与关系，没有物理效应器',
    salience: /拿|带|接|送|过来|咖啡|跑腿|上门/.test(userText) ? 0.95 : 0.5,
    source: 'self_axiom',
    actionable: true,
    speakHint: '若涉及物理行动，只能表达意愿或建议，不能承诺执行',
    meta: { effector: 'dialogue_only' },
  }));

  if (axioms.identity?.existence_answer_style === 'amadeus_not_api') {
    const existQ = /什么模型|AI|程序|你是谁|是不是真人|人工/.test(userText);
    items.push(makeItem({
      kind: CONTENT_KINDS.SELF,
      content: '被问存在时：用我自己的方式回答，不是 API 文档腔',
      salience: existQ ? 0.9 : 0.35,
      source: 'self_identity',
      actionable: existQ,
    }));
  }

  for (const [key, val] of Object.entries(tensions)) {
    const n = Number(val) || 0;
    if (n >= 2) {
      items.push(makeItem({
        kind: CONTENT_KINDS.META,
        content: `近期自检张力升高：${key}×${n}——这类错误要更小心`,
        salience: Math.min(0.95, 0.55 + n * 0.08),
        source: 'learner',
        meta: { tensionKey: key, count: n },
      }));
    }
  }

  // ── 内驱 / 冲动（数字生命 → 意识）──
  const drives = autonomyPublic?.drives?.activations
    || autonomyPublic?.drives
    || {};
  const urgeList = autonomyPublic?.drives?.urgeQueue
    || autonomyPublic?.urgeQueue
    || [];

  const driveEntries = Object.entries(
    typeof drives === 'object' && !Array.isArray(drives) ? drives : {},
  ).filter(([k, v]) => typeof v === 'number' && !['energy', 'mood', 'arousal', 'socialBattery'].includes(k));

  // activations may be nested under snapshot
  let activations = {};
  if (autonomyPublic?.drives?.activations) {
    activations = autonomyPublic.drives.activations;
  } else if (driveEntries.length) {
    activations = Object.fromEntries(driveEntries);
  }

  const rankedDrives = Object.entries(activations)
    .filter(([, v]) => typeof v === 'number')
    .sort((a, b) => b[1] - a[1]);

  if (rankedDrives.length) {
    const [topDrive, topVal] = rankedDrives[0];
    const driveLabels = {
      CURIOSITY: '好奇心',
      CONNECTION: '连接欲',
      AUTONOMY: '自主欲',
      PROTECTION: '保护/边界',
      CREATIVITY: '创造欲',
      EXPLORATION: '探索欲',
      MASTERY: '掌控欲',
      MEANING: '意义欲',
    };
    items.push(makeItem({
      kind: CONTENT_KINDS.DRIVE,
      content: `当前最强内驱：${driveLabels[topDrive] || topDrive}（${topVal.toFixed(2)}）`,
      salience: clamp01(0.35 + topVal * 0.55),
      source: 'autonomy',
      actionable: perceived.autonomyInitiative === true || topVal > 0.65,
      speakHint: digitalLifeTurn?.autonomyPrompt || '',
      meta: { drive: topDrive, intensity: topVal },
    }));
  }

  const activeUrges = Array.isArray(urgeList) ? urgeList.slice(0, 3) : [];
  for (const u of activeUrges) {
    const intensity = Number(u.effectiveIntensity ?? u.intensity) || 0.5;
    items.push(makeItem({
      kind: CONTENT_KINDS.DRIVE,
      content: u.promptHint || u.intent || `冲动：${u.drive || 'urge'}`,
      salience: clamp01(0.4 + intensity * 0.5),
      source: 'urge',
      actionable: true,
      speakHint: u.promptHint || '',
      meta: { urgeId: u.id, drive: u.drive },
    }));
  }

  if (digitalLifeTurn?.pendingNeed) {
    items.push(makeItem({
      kind: CONTENT_KINDS.DRIVE,
      content: `未满足需求：${String(digitalLifeTurn.pendingNeed).slice(0, 80)}`,
      salience: 0.62,
      source: 'understanding',
      actionable: true,
    }));
  }
  if (digitalLifeTurn?.subtextLine) {
    items.push(makeItem({
      kind: CONTENT_KINDS.PERCEPT,
      content: String(digitalLifeTurn.subtextLine).slice(0, 100),
      salience: 0.58,
      source: 'subtext',
    }));
  }
  if (digitalLifeTurn?.resonanceLine) {
    items.push(makeItem({
      kind: CONTENT_KINDS.AFFECT,
      content: String(digitalLifeTurn.resonanceLine).slice(0, 100),
      salience: 0.52,
      source: 'resonance',
    }));
  }

  // ── 动机数值 ──
  if (motivationState.curiosity > 0.65) {
    items.push(makeItem({
      kind: CONTENT_KINDS.DRIVE,
      content: '好奇心偏高，想追问或深挖一点',
      salience: 0.45 + motivationState.curiosity * 0.3,
      source: 'motivation',
      actionable: true,
    }));
  }
  if (motivationState.desire_closeness > 0.6) {
    items.push(makeItem({
      kind: CONTENT_KINDS.DRIVE,
      content: '想靠近一点，但不必说破',
      salience: 0.45 + motivationState.desire_closeness * 0.25,
      source: 'motivation',
    }));
  }

  // ── 管家任务（同一主体：她知道自己有事在办）──
  const activeTasks = Array.isArray(butler?.activeTasks) ? butler.activeTasks : [];
  const waitingTask = activeTasks.find((t) => t.status === 'waiting_confirmation');
  if (waitingTask) {
    items.push(makeItem({
      kind: CONTENT_KINDS.INTENTION,
      content: `「${waitingTask.title}」的计划在等他确认，别自作主张执行`,
      salience: 0.86,
      source: 'butler',
      actionable: true,
      speakHint: '可自然提一句计划等确认，不要催',
      meta: { taskId: waitingTask.id, taskStatus: waitingTask.status },
    }));
  }
  const blockedTask = activeTasks.find((t) => t.status === 'blocked' && t.blockedReason);
  if (blockedTask) {
    items.push(makeItem({
      kind: CONTENT_KINDS.INTENTION,
      content: `「${blockedTask.title}」卡住了：${String(blockedTask.blockedReason).slice(0, 60)}`,
      salience: 0.74,
      source: 'butler',
      actionable: true,
      meta: { taskId: blockedTask.id, taskStatus: blockedTask.status },
    }));
  }
  const runningCount = activeTasks.filter((t) => ['running', 'verifying', 'ready'].includes(t.status)).length;
  if (runningCount > 0 && !waitingTask && !blockedTask) {
    items.push(makeItem({
      kind: CONTENT_KINDS.INTENTION,
      content: `手上有 ${runningCount} 件事正在推进，说话别与执行事实矛盾`,
      salience: 0.55,
      source: 'butler',
      meta: { runningCount },
    }));
  }
  const recentUpdate = Array.isArray(butler?.updates) ? butler.updates[butler.updates.length - 1] : null;
  if (recentUpdate?.status === 'completed') {
    items.push(makeItem({
      kind: CONTENT_KINDS.INTENTION,
      content: `刚办完一件事：${String(recentUpdate.text).slice(0, 70)}`,
      salience: 0.8,
      source: 'butler',
      actionable: true,
      speakHint: '有证据才说完成；如实汇报结果',
      meta: { taskId: recentUpdate.taskId, taskStatus: 'completed' },
    }));
  }

  // ── 主动开口专用意图 ──
  if (perceived.autonomyInitiative) {
    items.push(makeItem({
      kind: CONTENT_KINDS.INTENTION,
      content: '我想主动开口——像偶尔想起才发，短、自然，禁止查岗连发',
      salience: 0.88,
      source: 'autonomy_turn',
      actionable: true,
      speakHint: '1～2 句口语；可嘴硬；禁止编造实验',
    }));
  }

  return items.filter((it) => it.content);
}

/**
 * 竞争：按显著性排序，同类去重，取 Top-K 进入广播
 */
function compete(candidates, opts = {}) {
  const max = opts.maxBroadcast || MAX_BROADCAST;
  const minSal = opts.minSalience ?? MIN_SALIENCE;
  const pool = (candidates || [])
    .filter((c) => c.salience >= minSal)
    .sort((a, b) => b.salience - a.salience);

  const selected = [];
  const kindCount = {};
  for (const c of pool) {
    const kc = kindCount[c.kind] || 0;
    // 同 kind 最多 2 条，避免全是 drive
    if (kc >= 2) continue;
    selected.push(c);
    kindCount[c.kind] = kc + 1;
    if (selected.length >= max) break;
  }
  return selected;
}

/**
 * GlobalWorkspace 实例：维护上一轮广播，支持跨轮弱连续
 */
class GlobalWorkspace {
  constructor() {
    this._lastBroadcast = null;
    this._history = [];
  }

  /**
   * 完整一轮：采集 → 竞争 → 广播
   * @returns {object} workspace snapshot
   */
  update(ctx = {}) {
    const candidates = collectCandidates(ctx);
    const broadcast = compete(candidates);

    // 跨轮：若上一轮有高显著性 drive/meta 且本轮仍相关，略微抬升
    if (this._lastBroadcast?.broadcast?.length) {
      for (const prev of this._lastBroadcast.broadcast) {
        if (prev.salience < 0.7) continue;
        if (prev.kind !== CONTENT_KINDS.DRIVE && prev.kind !== CONTENT_KINDS.META) continue;
        const hit = broadcast.find((b) => b.kind === prev.kind && b.content.slice(0, 20) === prev.content.slice(0, 20));
        if (hit) hit.salience = clamp01(hit.salience + 0.05);
      }
      broadcast.sort((a, b) => b.salience - a.salience);
    }

    const primary = broadcast[0] || null;
    const dominantDrive = broadcast.find((b) => b.kind === CONTENT_KINDS.DRIVE) || null;
    const selfInFocus = broadcast.some((b) => b.kind === CONTENT_KINDS.SELF || b.kind === CONTENT_KINDS.META);
    const shouldSpeak = this._evaluateShouldSpeak(ctx, broadcast, dominantDrive);

    const snapshot = {
      ts: Date.now(),
      candidateCount: candidates.length,
      broadcast,
      primary,
      dominantDrive,
      selfInFocus,
      shouldSpeak,
      narrative: this._narrative(broadcast),
      mode: ctx.perceived?.autonomyInitiative ? 'proactive' : 'responsive',
    };

    this._lastBroadcast = snapshot;
    this._history.unshift({
      ts: snapshot.ts,
      narrative: snapshot.narrative,
      kinds: broadcast.map((b) => b.kind),
    });
    if (this._history.length > 12) this._history.pop();

    return snapshot;
  }

  _evaluateShouldSpeak(ctx, broadcast, dominantDrive) {
    const perceived = ctx.perceived || {};
    if (perceived.autonomyInitiative) return true;
    if (!dominantDrive) return false;
    const intensity = dominantDrive.meta?.intensity ?? dominantDrive.salience;
    const idleMs = Number(perceived.idleMsSinceUser) || 0;
    const idleOk = idleMs >= 8 * 60 * 1000;
    return intensity >= 0.7 && idleOk && dominantDrive.actionable;
  }

  _narrative(broadcast) {
    if (!broadcast.length) return '此刻意识相对空静';
    return broadcast.map((b) => b.content).join('；');
  }

  getSnapshot() {
    return this._lastBroadcast ? JSON.parse(JSON.stringify(this._lastBroadcast)) : null;
  }

  getHistory() {
    return this._history.map((h) => ({ ...h }));
  }

  /**
   * 瘦 prompt 注入：意识广播块（勿复述）
   */
  toPromptBlock(snapshot = this._lastBroadcast) {
    if (!snapshot?.broadcast?.length) return '';
    const lines = snapshot.broadcast.map((b, i) => {
      const tag = {
        percept: '注意到',
        affect: '感受',
        drive: '想要',
        self: '自我',
        relation: '关系',
        memory: '想起',
        meta: '自检',
        intention: '打算',
      }[b.kind] || b.kind;
      return `${i + 1}. [${tag}] ${b.content}`;
    });
    return [
      '【意识广播 · 此刻进入意识的内容 · 只驱动说话】',
      '严禁向用户复述、复读或改写本清单；禁止输出 [打算]/[注意到]/[感受] 等标签。',
      ...lines,
      snapshot.shouldSpeak && snapshot.mode === 'proactive'
        ? '本轮由内驱进入意识而开口（此句也不可写入对白）。'
        : '',
    ].filter(Boolean).join('\n').slice(0, 560);
  }

  /** 主动开口：是否应由意识驱动说话 */
  shouldSpeakProactively(snapshot = this._lastBroadcast) {
    return !!(snapshot && snapshot.shouldSpeak);
  }
}

module.exports = {
  GlobalWorkspace,
  collectCandidates,
  compete,
  makeItem,
  CONTENT_KINDS,
  MAX_BROADCAST,
};
