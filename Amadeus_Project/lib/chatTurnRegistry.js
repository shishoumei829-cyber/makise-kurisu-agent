'use strict';

class ChatTurnRegistry {
  constructor() {
    this.active = new Map();
  }

  begin(conversationId, turnId) {
    const conversation = String(conversationId || 'default');
    const turn = String(turnId || `turn_${Date.now()}`);
    const previous = this.active.get(conversation);
    if (previous && !previous.controller.signal.aborted) {
      previous.controller.abort('superseded-by-new-turn');
    }
    const entry = { conversationId: conversation, turnId: turn, controller: new AbortController() };
    this.active.set(conversation, entry);
    return {
      conversationId: conversation,
      turnId: turn,
      signal: entry.controller.signal,
      isCurrent: () => this.active.get(conversation) === entry && !entry.controller.signal.aborted,
      cancel: (reason = 'cancelled') => {
        if (!entry.controller.signal.aborted) entry.controller.abort(reason);
      },
      complete: () => {
        if (this.active.get(conversation) === entry) this.active.delete(conversation);
      },
    };
  }
}

module.exports = { ChatTurnRegistry };
