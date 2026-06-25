'use strict';

const { SCENE_LEX } = require('./constants');

/**
 * 环境理解：视觉/文本场景 → 标签、可用性、情境压力。
 */
class EnvironmentSense {
  constructor() {
    this.sceneContext = null;
    this.userState = null;
    this._history = [];
    this.situationalPressure = 0.3;
  }

  _matchTags(text) {
    const tags = [];
    for (const [tag, patterns] of Object.entries(SCENE_LEX)) {
      if (patterns.some((re) => re.test(text))) tags.push(tag);
    }
    return tags;
  }

  understandScene(visualData) {
    const text = String(visualData || '').trim();
    if (!text) return null;

    const scene = {
      raw: text.slice(0, 220),
      tags: this._matchTags(text),
      at: Date.now(),
      salience: Math.min(1, 0.35 + text.length / 200),
    };

    if (!scene.tags.length) scene.tags.push('ordinary');

    this.sceneContext = scene;
    this._history.push(scene);
    if (this._history.length > 25) this._history.shift();
    this.inferUserState(scene);
    return scene;
  }

  inferUserState(scene) {
    if (!scene) return null;
    const state = { mood: 'neutral', availability: 'present', note: '', stress: 0.2 };

    if (scene.tags.includes('absent')) {
      state.availability = 'away';
      state.note = '他好像不在';
      state.stress = 0.15;
    } else if (scene.tags.includes('focused')) {
      state.availability = 'busy';
      state.note = '他在专注，别打扰太深';
      state.stress = 0.45;
    } else if (scene.tags.includes('distracted')) {
      state.availability = 'distracted';
      state.note = '他分心看手机';
      state.stress = 0.35;
    }

    if (scene.tags.includes('tired')) {
      state.mood = 'tired';
      state.note = (state.note ? `${state.note}；` : '') + '看起来累';
      state.stress += 0.15;
    }
    if (scene.tags.includes('positive')) state.mood = 'positive';
    if (scene.tags.includes('lonely')) {
      state.mood = 'lonely';
      state.note = (state.note ? `${state.note}；` : '') + '环境有些空';
    }
    if (scene.tags.includes('dim')) state.stress += 0.1;

    state.stress = Math.min(1, state.stress);
    this.userState = state;
    this.situationalPressure = state.stress;
    return state;
  }

  behaviorHint() {
    const state = this.userState;
    if (!state) return '';
    if (state.availability === 'busy') return '环境忙：短句、别连环追问';
    if (state.availability === 'away') return '环境空：可独处反思，主动开口要克制';
    if (state.mood === 'tired') return '环境累：语气轻一点';
    return '';
  }

  toPromptLine() {
    if (!this.sceneContext) return '';
    const state = this.userState || this.inferUserState(this.sceneContext);
    const tags = this.sceneContext.tags.join('、');
    const note = state?.note ? `；${state.note}` : '';
    const hint = this.behaviorHint();
    return `环境：${tags}${note}${hint ? `（${hint}）` : ''}`;
  }

  load(data) {
    if (!data) return;
    if (data.sceneContext) this.sceneContext = data.sceneContext;
    if (data.userState) this.userState = data.userState;
    if (data._history) this._history = data._history;
    if (typeof data.situationalPressure === 'number') this.situationalPressure = data.situationalPressure;
  }

  snapshot() {
    return {
      sceneContext: this.sceneContext,
      userState: this.userState,
      situationalPressure: this.situationalPressure,
      history: this._history.slice(-4),
    };
  }
}

module.exports = { EnvironmentSense };
