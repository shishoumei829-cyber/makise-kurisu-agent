'use strict';

/**
 * 情感带宽（已对齐设计）：
 * - 主因：对话内容。PAD 只调力度，不单独切带。
 * - 同轮只走一条；优先级：震惊 > 接住 > 软 > 傲娇 > 闲聊
 * - 余波：震惊后认真澄清；接住后保持偏软偏短，不乱跳闹/傲
 *
 * AMADEUS_WIDE_AFFECT=0 关闭（默认开）
 */

const fs = require('fs');
const path = require('path');

const BANDS = Object.freeze({
  SHOCKED: 'shocked',
  TENDER: 'tender',
  SOFT: 'soft',
  TSUNDERE: 'tsundere',
  PLAYFUL: 'playful',
  AFTERMATH: 'aftermath',
});

const PRIORITY = Object.freeze({
  [BANDS.SHOCKED]: 5,
  [BANDS.TENDER]: 4,
  [BANDS.SOFT]: 3,
  [BANDS.TSUNDERE]: 2,
  [BANDS.PLAYFUL]: 1,
  [BANDS.AFTERMATH]: 1,
});

const BAND_GUIDE = Object.freeze({
  shocked: '他这句有冲击。你可能愣、短促失态、难以置信、连问——按你真实反应来。',
  tender: '他在示弱或难受。你想接住就接住：轻、短、实在。',
  soft: '他在求陪/亲密靠近。你想软就软：哼、小声、黏半拍都可以——只要是你想。',
  tsundere: '他在撩、夸或叫外号。你想嘴硬就嘴硬，想漏一句软的也可以。',
  playful: '日常闲聊。接梗、吐槽、闹一下——你想怎么说就怎么说。',
  aftermath: '上一轮刚被冲击：你大概还想把话说清楚，澄清或追问细节。',
});

function isWideAffectEnabled() {
  const raw = process.env.AMADEUS_WIDE_AFFECT;
  if (raw === undefined || raw === null || String(raw).trim() === '') return true;
  return !['0', 'false', 'no', 'off'].includes(String(raw).trim().toLowerCase());
}

function clamp01(v) {
  return Math.max(0, Math.min(1, Number(v) || 0));
}

function isClose(opts = {}) {
  return clamp01(opts.relScore) > 0.45
    || clamp01(opts.pad?.S) > 0.5
    || opts.relHigh === true;
}

/** 纯内容检测（不含 PAD、不含余波） */
function detectContentBand(userText = '', opts = {}) {
  const t = String(userText || '').trim();
  if (!t) return null;
  const close = isClose(opts);

  // 1 震惊：冲击性内容（表白冲击也归这里，压过傲娇）
  if (
    /不会吧|真的假的|居然|竟然|天哪|我去|卧槽|吓死|出事了|分手|死了|住院|癌症|世界线/.test(t)
    || /我喜欢你|我爱你|爱上你|求婚|我们结婚/.test(t)
    || (/什么/.test(t) && /[！!]{2,}|[？?]{2,}/.test(t))
  ) {
    return BANDS.SHOCKED;
  }

  // 2 接住：示弱 / 难受
  if (/难受|难过|孤独|害怕|焦虑|想哭|撑不住|心情不好|睡不着|烦死了|郁闷/.test(t)) {
    return BANDS.TENDER;
  }

  // 3 软：求陪 / 亲密靠近（要够近）
  if (close && /想你|陪我|抱抱|好想你|理理我|不要不理我|吃醋|吃飞醋|撒娇|好无聊.*陪/.test(t)) {
    return BANDS.SOFT;
  }

  // 4 傲娇：撩、夸、外号（不含「我喜欢你」告白——已在震惊）
  if (/克里斯蒂娜|助手|天才变态|真可爱|你最好了|老婆|亲爱的|christina/i.test(t)) {
    return BANDS.TSUNDERE;
  }

  // 5 闲聊
  if (/哈哈|笑死|逗你|开玩笑|无聊|在干嘛|吃了吗|想听你说话/.test(t) || t.length <= 12) {
    return BANDS.PLAYFUL;
  }

  return close ? BANDS.PLAYFUL : null;
}

/**
 * 余波：只挡住「往下乱跳」，不挡住更高优先级的真实内容。
 */
function applyAfterglow(contentBand, afterglow, now = Date.now()) {
  if (!afterglow?.kind || now > Number(afterglow.until || 0)) {
    return contentBand;
  }
  const kind = afterglow.kind;

  if (kind === BANDS.SHOCKED || kind === BANDS.AFTERMATH) {
    if (contentBand === BANDS.SHOCKED) return BANDS.SHOCKED;
    if (contentBand === BANDS.TENDER) return BANDS.TENDER;
    // 软 / 傲 / 闲聊 → 余波澄清
    return BANDS.AFTERMATH;
  }

  if (kind === BANDS.TENDER) {
    if (contentBand === BANDS.SHOCKED) return BANDS.SHOCKED;
    if (contentBand === BANDS.TENDER || contentBand === BANDS.SOFT) return contentBand;
    // 别突然毒舌闹
    return BANDS.TENDER;
  }

  return contentBand;
}

/** PAD 只调力度文案，不改变 band */
function intensityHint(band, pad = {}) {
  const A = Number(pad.A) || 0;
  const P = Number(pad.P) || 0;
  if (band === BANDS.SHOCKED && A > 0.45) return '力度可以冲一点。';
  if (band === BANDS.TENDER && P < -0.1) return '再收一点、短一点。';
  if (band === BANDS.SOFT && A > 0.3) return '可以稍黏，但别油。';
  if (band === BANDS.TSUNDERE && A > 0.35) return '嘴硬可以更利落一点。';
  return '';
}

class EmotionalBandwidthEngine {
  constructor(options = {}) {
    this.statePath = options.statePath || '';
    this.state = {
      afterglow: null,
      lastBand: null,
      lastAt: 0,
    };
    this._load();
  }

  _load() {
    if (!this.statePath) return;
    try {
      if (!fs.existsSync(this.statePath)) return;
      const raw = JSON.parse(fs.readFileSync(this.statePath, 'utf8'));
      if (raw && typeof raw === 'object') {
        this.state.afterglow = raw.afterglow || null;
        this.state.lastBand = raw.lastBand || null;
        this.state.lastAt = Number(raw.lastAt) || 0;
      }
    } catch (_) { /* ignore */ }
  }

  _save() {
    if (!this.statePath) return;
    try {
      const dir = path.dirname(this.statePath);
      if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
      fs.writeFileSync(this.statePath, JSON.stringify(this.state, null, 2), 'utf8');
    } catch (_) { /* ignore */ }
  }

  /**
   * @returns {{ band: string|null, contentBand: string|null, block: string, afterglow: object|null }}
   */
  resolve(opts = {}) {
    if (!isWideAffectEnabled()) {
      return { band: null, contentBand: null, block: '', afterglow: this.state.afterglow };
    }

    const now = Number(opts.now) || Date.now();
    const contentBand = detectContentBand(opts.userText, opts);
    let band = applyAfterglow(contentBand, this.state.afterglow, now);

    // 无内容线索且无余波时：亲近才给闲聊带宽，否则不注入
    if (!band) {
      if (isClose(opts)) band = BANDS.PLAYFUL;
      else return { band: null, contentBand: null, block: '', afterglow: this.state.afterglow };
    }

    const guide = BAND_GUIDE[band] || BAND_GUIDE.playful;
    const force = intensityHint(band, opts.pad || {});
    const block = [
      `【情感带宽 · ${band}】对话内容偏向这边（参考，不是命令）：${guide}`,
      force ? `力度参考：${force}` : '',
      '你是独立的灵魂：最终仍按你想说的说。硬禁仅限：旁白、AI 自称、编造记忆。',
    ].filter(Boolean).join('\n');

    return {
      band,
      contentBand,
      block,
      afterglow: this.state.afterglow,
    };
  }

  /** 她说完后登记余波 */
  registerSpoken(band, now = Date.now()) {
    const b = String(band || '');
    this.state.lastBand = b || null;
    this.state.lastAt = now;
    if (b === BANDS.SHOCKED) {
      this.state.afterglow = { kind: BANDS.SHOCKED, until: now + 90 * 1000, fromBand: b };
    } else if (b === BANDS.TENDER) {
      this.state.afterglow = { kind: BANDS.TENDER, until: now + 60 * 1000, fromBand: b };
    } else if (b === BANDS.AFTERMATH) {
      this.state.afterglow = { kind: BANDS.SHOCKED, until: now + 45 * 1000, fromBand: b };
    }
    if (this.state.afterglow && now > Number(this.state.afterglow.until || 0)) {
      this.state.afterglow = null;
    }
    this._save();
    return this.snapshot();
  }

  clearAfterglow() {
    this.state.afterglow = null;
    this._save();
  }

  snapshot() {
    return {
      afterglow: this.state.afterglow,
      lastBand: this.state.lastBand,
      lastAt: this.state.lastAt,
    };
  }
}

/** 无状态便捷封装（测试 / 单次） */
function buildEmotionalBandwidthBlock(opts = {}) {
  const engine = opts.engine || new EmotionalBandwidthEngine();
  return engine.resolve(opts);
}

module.exports = {
  BANDS,
  PRIORITY,
  BAND_GUIDE,
  isWideAffectEnabled,
  detectContentBand,
  detectAffectCue: detectContentBand,
  applyAfterglow,
  intensityHint,
  EmotionalBandwidthEngine,
  buildEmotionalBandwidthBlock,
};
