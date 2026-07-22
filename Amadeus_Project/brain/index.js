'use strict';

const { wrapBrainRuntime } = require('./deps');
const { brainTurn } = require('./turn');
const { init: initLegacyChat, runChatTurn } = require('./legacyChat');
const { createBrainPipeline } = require('./pipeline');
const { ConsciousnessLayer } = require('./consciousness');
const { GlobalWorkspace } = require('./workspace');

class Brain {
  /**
   * @param {{ runtime: import('./deps').BrainRuntimeDeps, runChatTurn?: Function }} options
   */
  constructor(options) {
    this.runtime = wrapBrainRuntime(options.runtime);
    if (!this.runtime.globalWorkspace) {
      this.runtime.globalWorkspace = new GlobalWorkspace();
    }
    if (!this.runtime.consciousnessLayer) {
      this.runtime.consciousnessLayer = new ConsciousnessLayer({
        workspace: this.runtime.globalWorkspace,
      });
    }
    this.runtime.brainPipeline = createBrainPipeline(this.runtime);
    initLegacyChat(this.runtime);
    this._runChatTurn = options.runChatTurn || runChatTurn;
  }

  async turn(input) {
    const { req, res } = input;
    return brainTurn(this.runtime, req, res);
  }

  runLegacyChat(req, res) {
    return this._runChatTurn(req, res);
  }

  getLastTrace() {
    return this.runtime.brainPipeline?.getLastTrace?.() || null;
  }

  getWorldSnapshot() {
    return this.runtime.brainPipeline?.getWorldSnapshot?.()
      || this.runtime.worldModel?.getSnapshot?.()
      || null;
  }

  getSelfSnapshot() {
    return this.runtime.brainSelfModel?.snapshot?.() || null;
  }

  getConsciousness() {
    return this.runtime.brainPipeline?.getConsciousness?.()
      || this.runtime.consciousnessLayer?.toPublicState?.()
      || null;
  }

  getWorkspace() {
    return this.runtime.brainPipeline?.getWorkspaceSnapshot?.()
      || this.runtime.globalWorkspace?.getSnapshot?.()
      || null;
  }

  /** 主动开口：意识驱动评估 */
  evaluateProactiveSpeech(ctx = {}) {
    return this.runtime.brainPipeline?.evaluateProactiveSpeech?.(ctx)
      || { shouldSpeak: false, reason: 'no_pipeline' };
  }
}

module.exports = { Brain };
