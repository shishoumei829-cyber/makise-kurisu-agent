'use strict';

/**
 * 环境理解：视觉描述 → 场景与用户状态推断。
 */
class EnvironmentUnderstanding {
  constructor() {
    this.sceneContext = null;
    this.userState = null;
    this._history = [];
  }

  understandScene(visualData) {
    const text = String(visualData || '').trim();
    if (!text) return null;

    const scene = {
      raw: text.slice(0, 200),
      tags: [],
      at: Date.now(),
    };

    if (/疲惫|困|打哈欠|累/.test(text)) scene.tags.push('tired');
    if (/微笑|笑|开心/.test(text)) scene.tags.push('positive');
    if (/专注|工作|打字|认真/.test(text)) scene.tags.push('focused');
    if (/离开|不在|空|没人/.test(text)) scene.tags.push('absent');
    if (/手机|分心|低头/.test(text)) scene.tags.push('distracted');
    if (/黑暗|暗|灯关/.test(text)) scene.tags.push('dim');

    this.sceneContext = scene;
    this._history.push(scene);
    if (this._history.length > 20) this._history.shift();
    return scene;
  }

  inferUserState(scene) {
    if (!scene) return null;
    const state = { mood: 'neutral', availability: 'present', note: '' };

    if (scene.tags.includes('absent')) {
      state.availability = 'away';
      state.note = '他好像不在';
    } else if (scene.tags.includes('focused')) {
      state.availability = 'busy';
      state.note = '他在专注，别打扰太深';
    } else if (scene.tags.includes('distracted')) {
      state.availability = 'distracted';
      state.note = '他分心看手机';
    }
    if (scene.tags.includes('tired')) {
      state.mood = 'tired';
      state.note = (state.note ? state.note + '；' : '') + '看起来累';
    }
    if (scene.tags.includes('positive')) state.mood = 'positive';

    this.userState = state;
    return state;
  }

  toPromptLine() {
    if (!this.sceneContext) return '';
    const state = this.userState || this.inferUserState(this.sceneContext);
    const tags = this.sceneContext.tags.length ? this.sceneContext.tags.join('、') : '平常';
    const note = state?.note ? `；${state.note}` : '';
    return `环境：${tags}${note}`;
  }

  load(data) {
    if (!data) return;
    if (data.sceneContext) this.sceneContext = data.sceneContext;
    if (data.userState) this.userState = data.userState;
    if (data._history) this._history = data._history;
  }

  snapshot() {
    return {
      sceneContext: this.sceneContext,
      userState: this.userState,
      history: this._history.slice(-3),
    };
  }
}

module.exports = { EnvironmentUnderstanding };
