'use strict';

const { checkActs } = require('./capabilityGraph');
const { checkConsistency } = require('./consistency');

/**
 * Monitor.check — 结构化自检（C 混合之 A）。
 * @param {string} draft
 * @param {object} ctx
 * @returns {{ pass: boolean, confidence: number, violations: object[], speechActs: object[] }}
 */
function check(draft, ctx = {}) {
  const text = String(draft || '').trim();
  const { acts, violations: capViolations } = checkActs(text, ctx);
  const conViolations = checkConsistency(text, ctx);
  const violations = [...capViolations, ...conViolations];

  const dedup = [];
  const seen = new Set();
  for (const v of violations) {
    if (seen.has(v.id)) continue;
    seen.add(v.id);
    dedup.push(v);
  }

  const blockCount = dedup.filter((v) => v.severity === 'block').length;
  const pass = blockCount === 0;
  let confidence = 1;
  if (blockCount > 0) confidence = Math.max(0.2, 0.85 - blockCount * 0.25);
  if (dedup.some((v) => v.severity === 'warn')) confidence = Math.min(confidence, 0.75);
  if (!text) {
    return { pass: false, confidence: 0, violations: [{ id: 'empty', severity: 'block', rewriteHint: '空回复' }], speechActs: acts };
  }

  return { pass, confidence, violations: dedup, speechActs: acts };
}

module.exports = { check };
