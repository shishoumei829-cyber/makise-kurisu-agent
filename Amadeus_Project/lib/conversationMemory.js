'use strict';

/** @deprecated 使用 lib/unifiedDialogueLog.js */
const { UnifiedDialogueLog, needsConversationRecall, normText } = require('./unifiedDialogueLog');

class ConversationMemory extends UnifiedDialogueLog {
  constructor(dataDir) {
    super(dataDir);
    this.logPath = this.logPath;
  }
}

module.exports = {
  ConversationMemory,
  needsConversationRecall,
  normText,
};
