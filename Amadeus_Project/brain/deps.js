'use strict';

/**
 * Brain 运行时依赖包 — 由 server.js 在启动时组装。
 * 阶段 1：legacy /chat 逻辑通过 deps 访问单例与可变状态。
 *
 * @typedef {object} BrainRuntimeDeps
 * @property {string} OLLAMA_BASE
 * @property {string} padPath
 * @property {string} whoamiPath
 * @property {string} rootPath
 * @property {string} cachedSoulContent
 * @property {string} cachedVoiceContent
 * @property {string} cachedCharacterRules
 * @property {object} memorySystem
 * @property {object} unifiedDialogueLog
 * @property {object} behaviorIngest
 * @property {object} innerStateSix
 * @property {object} motivSystem
 * @property {object} behaviorSys
 * @property {object} selfModel
 * @property {object} goalSystem
 * @property {object} strategyLayer
 * @property {object} digitalLife
 * @property {object} userModelInst
 * @property {object} analyticsInst
 * @property {object} habitExtractor
 * @property {object} reinforcementLearning
 * @property {object} personalityEvolution
 * @property {object} selfReflection
 * @property {object} valueConsistency
 * @property {object} state - 可变轮次状态（引用传递）
 * @property {object} motivationState
 * @property {Function} needsLongTermMemory
 * @property {Function} retrieveTopContexts
 * @property {Function} updateMotivationFromMemory
 * @property {Function} applyOocRepair
 * @property {Function} postReplyPadUpdate
 * @property {Function} analyzeUserEmotion
 * @property {Function} agentDebugLog
 * @property {Function} readOllamaErrorBody
 * @property {Function} ollamaErrorLooksLikeModelLoadFail
 * @property {Function} ollamaErrorLooksLikeContextOverflow
 * @property {Function} ollamaChatStreamRawPiece
 * @property {Function} ollamaStreamToDelta
 * @property {Function} stripModelThinkingAll
 */

/**
 * @param {BrainRuntimeDeps} runtime
 * @returns {BrainRuntimeDeps}
 */
function wrapBrainRuntime(runtime) {
  if (!runtime || !runtime.state) {
    throw new Error('[brain/deps] runtime.state is required');
  }
  return runtime;
}

module.exports = { wrapBrainRuntime };
