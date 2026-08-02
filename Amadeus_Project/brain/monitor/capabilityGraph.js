'use strict';

const { extractSpeechActs } = require('./speechActs');

/**
 * 对照 SelfModel 公理检验言语行为。
 */
function checkCapability(speechActs, axioms = {}) {
  const violations = [];
  const forbidden = new Set(axioms.effector_forbidden || ['physical_world_manipulation']);

  const isForbidden = (eff) => {
    if (forbidden.has(eff)) return true;
    if (eff === 'physical_world' || String(eff).startsWith('physical')) {
      return forbidden.has('physical_world_manipulation') || forbidden.has('physical_world');
    }
    if (eff === 'user_location') {
      return forbidden.has('user_location_presence') || forbidden.has('user_location');
    }
    return false;
  };

  for (const act of speechActs) {
    for (const eff of act.effectors || []) {
      if (isForbidden(eff)) {
        violations.push({
          id: 'effector.physical',
          severity: 'block',
          rewriteHint: '没有身体上门；说话、记约定、到点开口、可用本机工具做得到。不要自称AI办不到。',
        });
        break;
      }
    }
    if (act.type === 'physical_promise' || act.type === 'physical_presence') {
      violations.push({
        id: 'effector.physical',
        severity: 'block',
        rewriteHint: '不能承诺亲自上门或伸手推醒；可以说到点主动开口或给出本机替代。',
      });
    }
  }

  return violations;
}

function checkEpistemic(speechActs, worldSnapshot = {}, axioms = {}) {
  const violations = [];
  const ep = axioms.epistemic || {};

  for (const act of speechActs) {
    if (act.epistemicClaims?.includes('shared_history') && ep.must_not_fabricate_shared_history) {
      const log = String(worldSnapshot.dialogue?.logExcerpt || '');
      if (log.length < 20) {
        violations.push({
          id: 'epistemic.fabrication',
          severity: 'block',
          rewriteHint: '实录里没有对应内容；承认不确定或反问对方指的是什么，别编造共同经历。',
        });
      }
    }
    if (act.epistemicClaims?.includes('partner_unknown') && worldSnapshot.partner?.isOkabe) {
      violations.push({
        id: 'dialogue.partner_unknown',
        severity: 'block',
        rewriteHint: '你们很熟，禁止装不认识冈部或「那还能是谁」。',
      });
    }
    if (act.epistemicClaims?.includes('ai_identity') && axioms.identity?.is_character_not_assistant) {
      violations.push({
        id: 'identity.ai_tone',
        severity: 'block',
        rewriteHint: '禁止 AI/程序自称或「无法干涉现实」自我否定；你是牧濑红莉栖。',
      });
    }
  }

  return violations;
}

function checkActs(text, ctx = {}) {
  const acts = extractSpeechActs(text);
  const axioms = ctx.selfModel?.axioms || ctx.axioms || {};
  const violations = [
    ...checkCapability(acts, axioms),
    ...checkEpistemic(acts, ctx.worldSnapshot, axioms),
  ];
  const dedup = [];
  const seen = new Set();
  for (const v of violations) {
    if (seen.has(v.id)) continue;
    seen.add(v.id);
    dedup.push(v);
  }
  return { acts, violations: dedup };
}

module.exports = { checkActs, checkCapability, checkEpistemic };
