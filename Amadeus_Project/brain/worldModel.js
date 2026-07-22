'use strict';

const { normalizeClientContext } = require('../lib/clientContext');
const {
  ensureWhoamiOnDisk,
  partnerIsOkabe,
  resolvePartnerDisplayName,
} = require('../lib/partnerIdentity');
const { effectiveRelScore } = require('../cognitive/companionMode');

/**
 * WorldModel — 当前轮「什么是真的」：实录、通道、关系、视觉、数字生命感知。
 */
class WorldModel {
  constructor() {
    this._lastSnapshot = null;
  }

  /**
   * @param {object} perceived - perceiveIncomingChat 输出
   * @param {object} reqBody
   * @param {object} deps
   * @returns {object} snapshot
   */
  update(perceived, reqBody, deps) {
    const {
      unifiedDialogueLog,
      memorySystem,
      getLastVision,
      digitalLifeTurn,
    } = deps;

    const clientCtx = normalizeClientContext(reqBody?.clientContext || {});
    const vision = getLastVision?.() || { description: '', timestamp: 0 };
    const visionActive = !!clientCtx.vision || !!(vision.description && vision.description.trim());

    let whoami = {};
    let partnerName = '';
    let isOkabe = false;
    try {
      whoami = ensureWhoamiOnDisk(deps.whoamiPath);
      partnerName = resolvePartnerDisplayName(whoami) || '';
      isOkabe = partnerIsOkabe(whoami);
    } catch { /* ignore */ }

    const relScore = effectiveRelScore(memorySystem?.getRelationshipScore?.() ?? 0);
    const dialogueRecent = unifiedDialogueLog?.getRecent?.(8) || [];
    const dialogueLogBlock = unifiedDialogueLog?.toPromptBlock?.({ maxChars: 600 }) || '';

    const snapshot = {
      ts: Date.now(),
      userText: perceived.userContent,
      cognitiveInput: perceived.cognitiveInput,
      autonomyInitiative: perceived.autonomyInitiative,
      replyingToProactive: perceived.replyingToProactive,
      proactiveAnchor: perceived.proactiveAnchor,
      userPresence: perceived.userPresenceState,
      clientContext: clientCtx,
      vision: {
        active: visionActive,
        description: clientCtx.vision || vision.description || '',
        timestamp: vision.timestamp || 0,
      },
      partner: {
        name: partnerName,
        isOkabe,
        whoami,
      },
      relationship: {
        score: relScore,
        closeness: Math.max(0, relScore),
      },
      dialogue: {
        recent: dialogueRecent,
        logExcerpt: dialogueLogBlock,
        entryCount: unifiedDialogueLog?.entriesCount ?? dialogueRecent.length,
      },
      digitalLife: digitalLifeTurn ? {
        resonanceLine: digitalLifeTurn.resonanceLine || '',
        subtextLine: digitalLifeTurn.subtextLine || '',
        pendingNeed: digitalLifeTurn.pendingNeed || '',
      } : null,
      channels: {
        hasVision: visionActive,
        hasPalace: !!(clientCtx.wantLongMemory && clientCtx.palace),
        longTermMemory: perceived.useLongTermMemory,
      },
    };

    this._lastSnapshot = snapshot;
    return snapshot;
  }

  getSnapshot() {
    return this._lastSnapshot ? JSON.parse(JSON.stringify(this._lastSnapshot)) : null;
  }

  /** 供瘦 prompt（≤200 字） */
  toPromptSummary(snapshot = this._lastSnapshot) {
    if (!snapshot) return '';
    const parts = [];
    if (snapshot.partner?.name) {
      parts.push(snapshot.partner.isOkabe
        ? `与${snapshot.partner.name}（冈部）对话，很熟。`
        : `与${snapshot.partner.name}对话。`);
    }
    if (snapshot.vision?.active && snapshot.vision.description) {
      parts.push(`视觉：${String(snapshot.vision.description).slice(0, 60)}`);
    }
    if (snapshot.clientContext?.situation) {
      parts.push(`情境：${String(snapshot.clientContext.situation).slice(0, 50)}`);
    }
    parts.push(`关系 ${snapshot.relationship.score.toFixed(2)}`);
    if (snapshot.dialogue?.entryCount) {
      parts.push(`实录 ${snapshot.dialogue.entryCount} 条`);
    }
    return `【世界模型】${parts.join(' ')}`.slice(0, 220);
  }
}

module.exports = { WorldModel };
