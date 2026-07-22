'use strict';

const { parseIncomingChat } = require('../cognitive/chatTurns');
const {
  extractLastRealUserLine,
  detectReplyingToHerThread,
} = require('../cognitive/turnContinuity');
const userPresence = require('../lib/userPresence');

/**
 * 聚合单轮感知输入：用户话、多轮、presence、主动/被动上下文。
 * Brain v1 感知层；阶段 1 从 server.js /chat 前段抽取。
 *
 * @param {object} body - req.body
 * @param {object} deps - { unifiedDialogueLog, needsLongTermMemory }
 * @returns {object}
 */
function perceiveIncomingChat(body, deps) {
  const { unifiedDialogueLog, needsLongTermMemory } = deps;

  const parsed = parseIncomingChat(body);
  let userContent = String(parsed.lastUser || '').trim();
  if (!userContent) {
    userContent = String(body.userMsg || body.message || '').trim();
  }

  // 浏览器会单独提交真实定稿；普通 /chat 不再反向导入整段客户端历史，
  // 防止摘要、标签、冲突检测等内部提示混入角色记忆。
  if (body.allowHistoryImport === true) unifiedDialogueLog.syncFromDialogue(parsed.dialogue);

  const autonomyInitiative = body.autonomyInitiative === true;
  const idleMsSinceUser = Math.max(0, Number(body.idleMsSinceUser) || 0);
  const threadHint = detectReplyingToHerThread(parsed.dialogue);
  const replyingToProactive = body.replyingToProactive === true
    || (threadHint.active && !autonomyInitiative);
  const proactiveAnchor = String(body.proactiveAnchor || threadHint.anchor || '').trim();

  const recentUserLinesForMem = (parsed.userLines && parsed.userLines.length)
    ? parsed.userLines.filter((l) => l && !/^（想说话）|^（转移话题）|^（以下是最近对话/.test(String(l).trim())).slice(-14)
    : (userContent && !/^（想说话）/.test(userContent) ? [userContent] : []);

  const lastRealUserLine = extractLastRealUserLine(parsed.dialogue)
    || (recentUserLinesForMem.length ? recentUserLinesForMem[recentUserLinesForMem.length - 1] : '');

  let userPresenceState = body.userPresence && body.userPresence.active
    ? body.userPresence
    : null;
  if (!userPresence.isPresenceActive(userPresenceState)) {
    userPresenceState = userPresence.resolvePresenceFromDialogue(parsed.dialogue);
  }
  if (lastRealUserLine) {
    userPresenceState = userPresence.mergePresenceState(
      userPresenceState,
      userPresence.analyzeUserPresence(lastRealUserLine, {
        recentUserLines: recentUserLinesForMem.slice(0, -1),
      }),
    );
  }

  const useLongTermMemory = autonomyInitiative
    ? true
    : needsLongTermMemory(userContent, recentUserLinesForMem);
  const useRagForTurn = useLongTermMemory;
  const cognitiveInput = (autonomyInitiative && lastRealUserLine) ? lastRealUserLine : userContent;

  return {
    parsed,
    userContent,
    autonomyInitiative,
    idleMsSinceUser,
    replyingToProactive,
    proactiveAnchor,
    recentUserLinesForMem,
    lastRealUserLine,
    userPresenceState,
    useLongTermMemory,
    useRagForTurn,
    cognitiveInput,
  };
}

module.exports = { perceiveIncomingChat };
