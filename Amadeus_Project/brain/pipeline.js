'use strict';

const monitor = require('./monitor');
const deliberation = require('./deliberation');
const deliberationLlm = require('./deliberationLlm');
const { applyLegacyOocRepair } = require('./monitor/consistency');
const { ConsciousnessLayer } = require('./consciousness');
const { GlobalWorkspace } = require('./workspace');

function isMonitorEnabled() {
  if (process.env.AMADEUS_BRAIN_MONITOR === '0') return false;
  if (process.env.AMADEUS_BRAIN_MONITOR === '1') return true;
  // 仅 AMADEUS_BRAIN=1 时默认开；禁止因意识层误开后注入模板句
  return String(process.env.AMADEUS_BRAIN || '0').trim() === '1';
}

function isPromptSlimEnabled() {
  return String(process.env.AMADEUS_BRAIN_PROMPT_SLIM || '0').trim() === '1'
    || String(process.env.AMADEUS_BRAIN || '0').trim() === '1'
    || process.env.AMADEUS_BRAIN_CONSCIOUSNESS !== '0';
}

function isConsciousnessEnabled() {
  return String(process.env.AMADEUS_BRAIN_CONSCIOUSNESS || '1').trim() !== '0';
}

/**
 * @param {object} deps
 */
function createBrainPipeline(deps) {
  const worldModel = deps.worldModel;
  const brainSelfModel = deps.brainSelfModel;
  const learner = deps.brainLearner;
  const workspace = deps.globalWorkspace || new GlobalWorkspace();
  const consciousness = deps.consciousnessLayer
    || new ConsciousnessLayer({ workspace });
  const subjectCore = deps.subjectCore;

  // 挂回 deps，供 Brain / server 读取
  deps.globalWorkspace = workspace;
  deps.consciousnessLayer = consciousness;

  let _turnCtx = null;
  let _lastTrace = null;

  return {
    isPromptSlimEnabled,
    isMonitorEnabled,
    isConsciousnessEnabled,

    /** 在 buildPrompt 前：World → Consciousness → Deliberation */
    beforePrompt(legacyCtx) {
      if (!worldModel || !brainSelfModel) return legacyCtx;

      const perceived = legacyCtx.perceived;
      const worldSnapshot = worldModel.update(
        perceived,
        legacyCtx.reqBody,
        {
          ...deps,
          whoamiPath: deps.whoamiPath,
          digitalLifeTurn: legacyCtx.digitalLifeTurn,
        },
      );

      brainSelfModel.syncEvolvedFromCognitive(deps.selfModel?.get?.());
      const selfSnapshot = brainSelfModel.snapshot();

      let consciousnessCycle = null;
      if (isConsciousnessEnabled()) {
        let autonomyPublic = null;
        try {
          autonomyPublic = deps.digitalLife?.autonomy?.getPublicState?.() || null;
        } catch { /* ignore */ }

        consciousnessCycle = consciousness.cycle({
          perceived,
          worldSnapshot,
          selfSnapshot,
          pad: legacyCtx.pad,
          digitalLifeTurn: legacyCtx.digitalLifeTurn,
          autonomyPublic,
          motivationState: deps.motivationState || {},
          deps,
        });
      }

      const delib = deliberation.plan({
        perceived,
        worldSnapshot,
        selfSnapshot,
        pad: legacyCtx.pad,
        behaviorContext: legacyCtx.behaviorContext,
        consciousness: consciousnessCycle,
      });

      // The workspace and deliberation are signals, not independent speakers.
      // A single persistent subject selects the actual topic and speech act.
      const subject = subjectCore?.deliberate?.({
        perceived: legacyCtx.perceived,
        userText: legacyCtx.perceived?.cognitiveInput || legacyCtx.perceived?.userContent,
        pad: legacyCtx.pad,
        motivationState: deps.motivationState || {},
        openThoughts: deps.soulRuntime?.snapshot?.().thoughts || [],
      }) || null;
      if (subject?.intent) {
        delib.intent = `subject:${subject.intent.action}`;
        delib.intentReason = subject.intent.reason;
      }

      _turnCtx = {
        perceived,
        worldSnapshot,
        selfSnapshot,
        deliberation: delib,
        consciousness: consciousnessCycle,
        subject,
        reqBody: legacyCtx.reqBody,
      };

      const out = { ...legacyCtx };
      out.brainConsciousness = consciousnessCycle
        ? consciousness.toPublicState(consciousnessCycle)
        : null;

      if (isPromptSlimEnabled() || isConsciousnessEnabled()) {
        out.brainSlimMode = true;
        out.brainWorldSummary = worldModel.toPromptSummary(worldSnapshot);
        out.brainSelfSummary = brainSelfModel.toPromptSummary();
        out.brainDeliberationBlock = deliberation.toPromptBlock(delib);
        out.brainWorkspaceBlock = consciousnessCycle
          ? consciousness.toPromptBlock(consciousnessCycle)
          : '';
        out.brainSubjectBlock = subject?.promptBlock || '';
        out.digitalLifeCtx = '';
        out.skipSymbolicInPrompt = true;
      }
      return out;
    },

    async processReply(draft, opts = {}) {
      const text = String(draft || '').trim();
      const applyGate = (raw) => {
        try {
          const { gateAssistantReply } = require('../lib/generationGate');
          const gated = gateAssistantReply(raw, {
            autonomy: !!(opts.oocOpts && opts.oocOpts.autonomy),
            proactive: !!(opts.oocOpts && opts.oocOpts.autonomy),
          });
          if (gated.action === 'drop') {
            console.warn('[brain/pipeline] generationGate drop:', (gated.reasons || []).join(','));
            return '';
          }
          if (gated.action === 'sanitize' && gated.text) return gated.text;
          return String(raw || '').trim();
        } catch {
          return String(raw || '').trim();
        }
      };

      if (!isMonitorEnabled()) {
        return applyGate(applyLegacyOocRepair('', text, opts.userText || '', opts.oocOpts || {}));
      }

      const ctx = {
        userText: opts.userText || '',
        oocOpts: opts.oocOpts || {},
        worldSnapshot: _turnCtx?.worldSnapshot || worldModel?.getSnapshot?.() || {
          dialogue: { logExcerpt: '' },
          partner: {},
        },
        selfModel: _turnCtx?.selfSnapshot || brainSelfModel?.snapshot?.(),
      };

      // 默认只旁观、不重写：她说的话就是她的话（AMADEUS_BRAIN_MONITOR_REWRITE=1 才启用改写）
      const rewriteEnabled = ['1', 'true', 'yes', 'on'].includes(
        String(process.env.AMADEUS_BRAIN_MONITOR_REWRITE || '0').trim().toLowerCase(),
      );

      let current = text;
      let delib = _turnCtx?.deliberation || { constraints: [] };
      let monitorResult = monitor.check(current, ctx);
      let rewrites = 0;
      let delibLlmUsed = false;

      if (!rewriteEnabled) {
        if (!monitorResult.pass) {
          const hardViolation = monitorResult.violations.some((violation) => (
            violation.severity === 'block'
            && /^(effector\.physical|epistemic\.fabrication|dialogue\.partner_unknown|identity\.ai_tone)/.test(violation.id)
          ));
          if (hardViolation) {
            current = deliberation.localReviseDraft(current, monitorResult);
            monitorResult = monitor.check(current, ctx);
            rewrites += 1;
          } else {
            console.warn('[brain/monitor] advisory only; keep model original');
          }
        }
      } else {
        while (!monitorResult.pass && rewrites < 2) {
          delib = deliberation.reviseFromMonitor(delib, monitorResult);
          current = deliberation.localReviseDraft(current, monitorResult);

          if (!monitor.check(current, ctx).pass && deliberationLlm.shouldUseDelibLlm(monitorResult)) {
            current = await deliberationLlm.rewriteDraft(current, {
              userText: ctx.userText,
              monitorResult,
              deliberation: delib,
              consciousness: _turnCtx?.consciousness,
            }, deps);
            delibLlmUsed = true;
          }

          monitorResult = monitor.check(current, ctx);
          rewrites += 1;
        }

        if (!monitorResult.pass) {
          console.warn('[brain/monitor] keep model original; no template fallback');
          current = text;
        }
      }

      const oocRepaired = applyLegacyOocRepair(opts.streamedRaw || '', current, ctx.userText, ctx.oocOpts);

      // 生成硬闸：与是否开启 monitor rewrite 无关；脏稿不得出脑管道
      const gatedOut = applyGate(oocRepaired);
      const subjectCheck = subjectCore?.evaluateReply?.(gatedOut, _turnCtx?.subject?.intent) || { ok: true };
      if (!subjectCheck.ok) console.warn('[subject-core] reply advisory:', subjectCheck.reason);
      subjectCore?.integrateOutcome?.({
        mode: _turnCtx?.subject?.intent?.mode || 'responsive',
        intent: _turnCtx?.subject?.intent,
        reply: gatedOut,
        accepted: !!gatedOut,
      });

      learner?.observe?.({
        monitorResult,
        userText: ctx.userText,
        draft: gatedOut,
        brainSelfModel,
      });

      const ws = _turnCtx?.consciousness?.workspace;
      _lastTrace = {
        pass: monitorResult.pass && !!gatedOut,
        confidence: monitorResult.confidence,
        violations: monitorResult.violations.map((v) => v.id),
        rewrites,
        delibLlmUsed,
        intent: delib.intent,
        subject: _turnCtx?.subject?.intent || null,
        subjectCheck,
        consciousness: {
          narrative: ws?.narrative || '',
          broadcastKinds: (ws?.broadcast || []).map((b) => b.kind),
          coherence: _turnCtx?.consciousness?.metrics?.coherence,
          selfInFocus: ws?.selfInFocus,
        },
      };
      if (_lastTrace.violations.length) {
        console.log(`[brain/monitor] pass=${monitorResult.pass} conf=${monitorResult.confidence.toFixed(2)} ids=${_lastTrace.violations.join(',')}`);
      }
      if (ws?.broadcast?.length) {
        console.log(`[brain/consciousness] ${ws.broadcast.length} broadcast · ${ws.narrative.slice(0, 80)}`);
      }

      return gatedOut;
    },

    /** 供 /chat/lite 与主动开口：是否应由意识驱动说话 */
    evaluateProactiveSpeech(ctx = {}) {
      const subjectPlan = subjectCore?.planProactive?.({
        ...ctx,
        pad: ctx.pad || deps.state?.currentPAD || {},
        motivationState: deps.motivationState || {},
        openThoughts: deps.soulRuntime?.snapshot?.().thoughts || [],
      });
      if (subjectPlan) return subjectPlan;
      if (!isConsciousnessEnabled()) {
        return { shouldSpeak: false, reason: 'consciousness_off' };
      }
      const perceived = {
        userContent: '',
        autonomyInitiative: true,
        idleMsSinceUser: ctx.idleMs || 0,
      };
      let worldSnapshot = worldModel?.getSnapshot?.() || {};
      if (worldModel && ctx.refreshWorld) {
        worldSnapshot = worldModel.update(perceived, {}, {
          ...deps,
          whoamiPath: deps.whoamiPath,
          digitalLifeTurn: ctx.digitalLifeTurn || null,
        });
      }
      const selfSnapshot = brainSelfModel?.snapshot?.() || {};
      let autonomyPublic = null;
      try {
        autonomyPublic = deps.digitalLife?.autonomy?.getPublicState?.() || null;
      } catch { /* ignore */ }

      const cycle = consciousness.cycle({
        perceived,
        worldSnapshot,
        selfSnapshot,
        pad: ctx.pad || deps.state?.currentPAD || {},
        digitalLifeTurn: ctx.digitalLifeTurn || null,
        autonomyPublic,
        motivationState: deps.motivationState || {},
        deps,
      });

      return {
        shouldSpeak: cycle.workspace.shouldSpeak,
        reason: cycle.intentionHint?.reason || '',
        intention: cycle.intentionHint?.intent,
        narrative: cycle.workspace.narrative,
        workspaceBlock: consciousness.toPromptBlock(cycle),
        public: consciousness.toPublicState(cycle),
      };
    },

    getLastTrace() {
      return _lastTrace ? { ..._lastTrace } : null;
    },

    getWorldSnapshot() {
      return worldModel?.getSnapshot?.() || null;
    },

    getConsciousness() {
      return consciousness.toPublicState() || consciousness.getLastCycle();
    },

    getWorkspaceSnapshot() {
      return workspace.getSnapshot();
    },
  };
}

module.exports = {
  createBrainPipeline,
  isMonitorEnabled,
  isPromptSlimEnabled,
  isConsciousnessEnabled,
};
