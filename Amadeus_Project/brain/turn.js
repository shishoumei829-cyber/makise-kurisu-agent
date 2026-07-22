'use strict';

const { runChatTurn } = require('./legacyChat');

/**
 * Brain.turn() — 中央认知循环
 * Perceive → World/Self → Consciousness(Workspace) → Deliberate → Generate → Monitor → Learn
 */
async function brainTurn(deps, req, res) {
  console.log('[brain] turn start');
  const result = await runChatTurn(req, res);
  const trace = deps.brainPipeline?.getLastTrace?.();
  const cons = deps.brainPipeline?.getConsciousness?.();
  if (trace || cons) {
    console.log(
      `[brain] turn end intent=${trace?.intent || '-'} `
      + `pass=${trace?.pass ?? '-'} `
      + `ws=${cons?.broadcast?.length ?? 0} `
      + `coherence=${cons?.metrics?.coherence ?? '-'}`,
    );
  } else {
    console.log('[brain] turn end');
  }
  return result;
}

module.exports = { brainTurn };
