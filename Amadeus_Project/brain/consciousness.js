'use strict';

const { GlobalWorkspace } = require('./workspace');

/**
 * ConsciousnessLayer — 意识工程层（Amadeus Brain）
 *
 * 定位（对标 LAAP「人工意识」的工程近似，不声称主观体验）：
 * 1. 统一当前状态进入工作空间竞争
 * 2. 少数内容被广播到 Deliberation / Prompt / Monitor
 * 3. 自我与边界作为意识内容在场
 * 4. 内驱可进入意识并驱动主动开口
 *
 * 循环：World/Self → Workspace.broadcast → Deliberation ← Monitor/Learn
 */
class ConsciousnessLayer {
  constructor(opts = {}) {
    this.workspace = opts.workspace || new GlobalWorkspace();
    this._lastCycle = null;
  }

  /**
   * 跑一轮意识更新
   * @param {object} ctx
   * @returns {object} cycle result
   */
  cycle(ctx = {}) {
    const autonomyPublic = this._resolveAutonomy(ctx);
    const butler = this._resolveButler(ctx);
    const workspace = this.workspace.update({
      ...ctx,
      autonomyPublic,
      butler,
    });

    const cycle = {
      ts: workspace.ts,
      workspace,
      // 意识品质指标（工程可观测，非 IIT）
      metrics: {
        broadcastCount: workspace.broadcast.length,
        candidateCount: workspace.candidateCount,
        maxSalience: workspace.primary?.salience ?? 0,
        selfInFocus: workspace.selfInFocus,
        driveInFocus: !!workspace.dominantDrive,
        coherence: this._coherence(workspace),
      },
      // 给 Deliberation 的压缩意图提示
      intentionHint: this._intentionHint(workspace, ctx),
      speakPolicy: this._speakPolicy(workspace, ctx),
    };

    this._lastCycle = cycle;
    return cycle;
  }

  _resolveAutonomy(ctx) {
    if (ctx.autonomyPublic) return ctx.autonomyPublic;
    try {
      return ctx.deps?.digitalLife?.autonomy?.getPublicState?.() || null;
    } catch {
      return null;
    }
  }

  /** 脑管衔接：管家任务状态作为意识候选进入工作空间（同一主体，不是外挂工具） */
  _resolveButler(ctx) {
    if (ctx.butler) return ctx.butler;
    const kernel = ctx.butlerKernel || ctx.deps?.butlerKernel;
    if (!kernel) return null;
    try {
      return {
        activeTasks: (kernel.getActiveTasks?.() || []).slice(0, 6),
        updates: kernel.updatesSince?.(Date.now() - 30 * 60000, 5) || [],
      };
    } catch {
      return null;
    }
  }

  /** 广播内容之间是否「拧成一股绳」（简单一致性） */
  _coherence(workspace) {
    const kinds = new Set((workspace.broadcast || []).map((b) => b.kind));
    if (kinds.size <= 1) return 0.5;
    // 有 percept + (drive|self|relation) 视为更连贯的回应态
    const hasPercept = kinds.has('percept');
    const hasAgentic = kinds.has('drive') || kinds.has('self') || kinds.has('intention');
    if (hasPercept && hasAgentic) return 0.85;
    if (hasPercept || hasAgentic) return 0.65;
    return 0.45;
  }

  _intentionHint(workspace, ctx) {
    const perceived = ctx.perceived || {};
    const userText = String(perceived.userContent || '');
    const primary = workspace.primary;

    if (perceived.autonomyInitiative) {
      return {
        intent: 'proactive_from_drive',
        reason: workspace.dominantDrive?.content || '内驱推动开口',
      };
    }
    if (/想你|喜欢|爱你/.test(userText)) {
      return { intent: 'respond_intimacy', reason: primary?.content || '亲密' };
    }
    if (/难受|烦|累|伤心/.test(userText)) {
      return { intent: 'emotional_support', reason: primary?.content || '情绪支持' };
    }
    if (workspace.selfInFocus && /拿|带|接|送|咖啡|过来|模型|AI|你是谁/.test(userText)) {
      return { intent: 'self_boundary', reason: '自我边界进入意识' };
    }
    if (workspace.dominantDrive?.meta?.drive === 'CURIOSITY') {
      return { intent: 'curious_engage', reason: workspace.dominantDrive.content };
    }
    if (workspace.dominantDrive?.meta?.drive === 'CONNECTION') {
      return { intent: 'seek_connection', reason: workspace.dominantDrive.content };
    }
    return {
      intent: primary?.kind === 'percept' ? 'respond_to_percept' : 'general_reply',
      reason: primary?.content || '一般回应',
    };
  }

  _speakPolicy(workspace, ctx) {
    const hints = (workspace.broadcast || [])
      .filter((b) => b.speakHint)
      .map((b) => b.speakHint);
    const selfItems = (workspace.broadcast || []).filter((b) => b.kind === 'self' || b.kind === 'meta');
    return {
      maxSentences: ctx.perceived?.autonomyInitiative ? 2 : 4,
      forbidPhysicalEffector: true,
      hints: hints.slice(0, 3),
      selfConstraints: selfItems.map((s) => s.content).slice(0, 2),
      fromConsciousness: true,
    };
  }

  getLastCycle() {
    return this._lastCycle ? JSON.parse(JSON.stringify(this._lastCycle)) : null;
  }

  getWorkspaceSnapshot() {
    return this.workspace.getSnapshot();
  }

  toPromptBlock(cycle = this._lastCycle) {
    if (!cycle?.workspace) return '';
    return this.workspace.toPromptBlock(cycle.workspace);
  }

  /** 对外可观测摘要 */
  toPublicState(cycle = this._lastCycle) {
    if (!cycle) return null;
    return {
      narrative: cycle.workspace.narrative,
      broadcast: cycle.workspace.broadcast.map((b) => ({
        kind: b.kind,
        content: b.content,
        salience: +b.salience.toFixed(3),
        source: b.source,
      })),
      metrics: cycle.metrics,
      intentionHint: cycle.intentionHint,
      shouldSpeak: cycle.workspace.shouldSpeak,
      mode: cycle.workspace.mode,
    };
  }
}

module.exports = { ConsciousnessLayer };
