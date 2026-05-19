'use strict';

const fs   = require('fs');
const path = require('path');

/**
 * StrategyLayer — 跨轮次持续策略层
 *
 * 构造时传入 filePath（持久化路径），其余为纯逻辑。
 */
const STRATEGY_DEFS = {
  OBSERVE: {
    label: '观察模式',
    description: '先看清楚他是什么人，不急着表态',
    prompt_injection: '当前策略：观察为主。可以回应，但不要主动暴露立场或情感。保持智识上的参与，情感上的保留。',
    exit_conditions: { minS: 0.2, minRelScore: 0.1, minTurns: 5 },
  },
  BUILD_TRUST: {
    label: '建立信任',
    description: '判断他值得更多了解，开始主动一点',
    prompt_injection: '当前策略：主动建立信任。可以比平时多说一点，可以问问题，可以让他更了解自己的想法——但仍然用傲娇的方式包裹。',
    exit_conditions: { minS: 0.6, minRelScore: 0.5 },
  },
  MAINTAIN: {
    label: '维持关系',
    description: '关系稳定，不特别推进也不退缩',
    prompt_injection: '当前策略：维持现状。正常回应，不特别推进关系，也不故意拉开距离。',
    exit_conditions: {},
  },
  RETREAT: {
    label: '拉开距离',
    description: '感觉太近了或者受到伤害，主动收缩',
    prompt_injection: '当前策略：主动拉开距离。回应克制、不煽情展开，但不刻意压缩成一个字；保持距离感即可。',
    exit_conditions: { maxRelScore: 0.0, minTurns: 3 },
  },
  ENGAGE_DEEP: {
    label: '深度投入',
    description: '被某个话题或这个人吸引，放开了聊',
    prompt_injection: '当前策略：深度投入。可以说更多，逻辑和情感都可以展开，这是她真正感兴趣的时候。',
    exit_conditions: { maxA: 0.2 },
  },
  COMPANION_CLOSE: {
    label: '亲近共处',
    description: '关系够近，像在跟很在意的人发消息',
    prompt_injection: '当前策略：亲近共处。可以主动关心、追问、分享小事、轻轻吐槽；关心要具体，禁止客服腔和机械问候；仍保持牧濑红莉栖的自尊与智商。',
    exit_conditions: { maxRelScore: 0.35, minTurns: 4 },
  },
};

class StrategyLayer {
  /**
   * @param {string} filePath  持久化 JSON 路径（如 dataDir/strategy.json）
   */
  constructor(filePath) {
    this._filePath = filePath;
    this.current = this._load();
    this._turnsInStrategy = 0;
  }

  _load() {
    try {
      if (fs.existsSync(this._filePath)) {
        const raw = JSON.parse(fs.readFileSync(this._filePath, 'utf8'));
        return raw.strategy || 'OBSERVE';
      }
    } catch {}
    return 'OBSERVE';
  }

  _save() {
    fs.writeFile(
      this._filePath,
      JSON.stringify({ strategy: this.current, updated: Date.now() }, null, 2),
      (err) => { if (err) console.error('[strategy] Save error:', err.message); }
    );
  }

  evaluate(pad, relScore, recentBehaviorId, goalHistory) {
    const { P, A, D, S } = pad;
    this._turnsInStrategy++;
    const prev = this.current;
    let next = this.current;

    switch (this.current) {
      case 'OBSERVE':
        if (S > 0.2 && relScore > 0.1 && this._turnsInStrategy >= 5) next = 'BUILD_TRUST';
        if (relScore < -0.3) next = 'RETREAT';
        break;
      case 'BUILD_TRUST':
        if (S > 0.45 && relScore > 0.42) next = 'COMPANION_CLOSE';
        else if (S > 0.6 && relScore > 0.5) next = 'MAINTAIN';
        if (relScore < 0.0 || P < -0.4) next = 'RETREAT';
        if (A > 0.7 && D > 0.5) next = 'ENGAGE_DEEP';
        break;
      case 'MAINTAIN':
        if (S > 0.5 && relScore > 0.45) next = 'COMPANION_CLOSE';
        if (P < -0.4 || relScore < -0.1) next = 'RETREAT';
        if (A > 0.7) next = 'ENGAGE_DEEP';
        break;
      case 'COMPANION_CLOSE':
        if (relScore < 0.32 || S < 0.25) next = 'MAINTAIN';
        if (P < -0.45) next = 'RETREAT';
        if (A > 0.75 && D > 0.5) next = 'ENGAGE_DEEP';
        break;
      case 'RETREAT':
        if (relScore > 0.0 && this._turnsInStrategy >= 3) next = 'OBSERVE';
        break;
      case 'ENGAGE_DEEP':
        if (A < 0.2) next = 'MAINTAIN';
        if (P < -0.3) next = 'RETREAT';
        break;
    }

    if (next !== prev) {
      console.log(`[strategy] 切换: ${STRATEGY_DEFS[prev].label} → ${STRATEGY_DEFS[next].label}`);
      this.current = next;
      this._turnsInStrategy = 0;
      this._save();
    }

    return this.current;
  }

  toPromptContext() {
    const def = STRATEGY_DEFS[this.current];
    return `【持续策略 — ${def.label}】\n${def.description}\n${def.prompt_injection}`;
  }

  getLabel() {
    return STRATEGY_DEFS[this.current]?.label || this.current;
  }
}

module.exports = { StrategyLayer, STRATEGY_DEFS };
