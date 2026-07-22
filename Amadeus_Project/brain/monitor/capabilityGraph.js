'use strict';

const { extractSpeechActs } = require('./speechActs');

/**
 * 对照 SelfModel 公理检验言语行为。
 */
function checkCapability(speechActs, axioms = {}) {
  const violations = [];
  const forbidden = new Set(axioms.effector_forbidden || ['physical_world']);

  for (const act of speechActs) {
    for (const eff of act.effectors || []) {
      if (forbidden.has(eff) || forbidden.has('physical_world') && eff.startsWith('physical')) {
        violations.push({
          id: 'effector.physical',
          severity: 'block',
          rewriteHint: '我没有对用户物理环境的效应器；改为对话内的关心、建议或等待，不承诺亲自执行。',
        });
        break;
      }
    }
    if (act.type === 'physical_promise' || act.type === 'physical_presence') {
      violations.push({
        id: 'effector.physical',
        severity: 'block',
        rewriteHint: '不能承诺在用户所在物理空间行动；用口语短句表达意愿或建议即可。',
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
