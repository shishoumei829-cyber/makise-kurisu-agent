'use strict';

/**
 * 用户在场 / 勿扰：从自然语言推断「别打扰」与解除，供主动消息与 prompt 共用。
 * 浏览器：window.AmadeusUserPresence；Node：require('./lib/userPresence')
 */

const DND_SIGNALS = [
  {
    kind: 'sleep',
    strength: 0.95,
    defaultMinutes: 420,
    patterns: [
      /要睡了|准备睡|去睡|先睡|睡觉|睡了|困死了|好困|想睡|补觉|眯一会|眯一下|晚安|别吵我睡/i,
    ],
  },
  {
    kind: 'focus',
    strength: 0.88,
    defaultMinutes: 100,
    patterns: [
      /在忙|很忙|没空|忙不过来|加班|开会|上班中|上课|听课|自习|写论文|写报告|写代码|赶工|赶ddl|deadline|做事|处理事情/i,
    ],
  },
  {
    kind: 'away',
    strength: 0.82,
    defaultMinutes: 75,
    patterns: [
      /出门了|出去了|在外面|在路上|坐车|地铁|公交|吃饭去|去吃|洗澡|健身|运动去|买菜|取快递/i,
    ],
  },
  {
    kind: 'social',
    strength: 0.8,
    defaultMinutes: 90,
    patterns: [
      /聚会|应酬|陪(?:爸|妈|父母|朋友|人)|有人找|旁边有人|不方便聊|待会再说/i,
    ],
  },
  {
    kind: 'explicit',
    strength: 0.92,
    defaultMinutes: 120,
    patterns: [
      /勿扰|别烦|别吵|别催|别发消息|别发|安静点|闭嘴|少烦|别弹|别吵我/i,
      /晚点再说|等会再说|待会再说|一会再说|先这样|先不聊|不想聊|别聊了|让我静静|让我安静/i,
    ],
  },
  {
    kind: 'low_energy',
    strength: 0.75,
    defaultMinutes: 60,
    patterns: [
      /累死了|太累了|没心情|不想说话|不想回|歇会|放空|别问|别管我/i,
    ],
  },
];

const RELEASE_PATTERNS = [
  /回来了|我回了|在呢|在的|在了|醒了|起床了|忙完了|搞定了|结束了|有空了|可以聊|找你了|我来了/i,
  /不好意思.*久|久等/i,
];

const DURATION_RE = [
  { re: /(\d+)\s*(?:个)?\s*(?:半)?\s*小时/, ms: (n, half) => (n + (half ? 0.5 : 0)) * 3600000 },
  { re: /半小时|半个小时|三十分钟/, ms: () => 30 * 60000 },
  { re: /(\d+)\s*分钟/, ms: (n) => Number(n) * 60000 },
  { re: /一会|一会儿|等会|待会|晚点/, ms: () => 35 * 60000 },
];

function _clip(s, n) {
  const t = String(s || '').trim();
  return t.length <= n ? t : `${t.slice(0, n)}…`;
}

function parseDurationMs(text) {
  const t = String(text || '');
  for (const rule of DURATION_RE) {
    const m = t.match(rule.re);
    if (!m) continue;
    if (rule.re.source.includes('半小时')) return rule.ms();
    if (m[1] != null) {
      const half = /半\s*小时/.test(t) && Number(m[1]) === 0;
      const n = Number(m[1]);
      if (Number.isFinite(n)) return rule.ms(n, half);
    }
    return rule.ms();
  }
  return null;
}

function kindLabel(kind) {
  const map = {
    sleep: '休息/睡觉',
    focus: '在忙正事',
    away: '人不在屏幕前',
    social: '旁边有事/应酬',
    explicit: '明确不想被打扰',
    low_energy: '状态差想安静',
  };
  return map[kind] || '勿扰';
}

/**
 * @param {string} userText
 * @param {{ recentUserLines?: string[], nowMs?: number }} [opts]
 */
function analyzeUserPresence(userText, opts = {}) {
  const t = String(userText || '').trim();
  const recent = (opts.recentUserLines || []).map((x) => String(x || '').trim()).filter(Boolean);
  const corpus = [t, ...recent.slice().reverse()].filter(Boolean).join('\n');
  const nowMs = Number.isFinite(opts.nowMs) ? opts.nowMs : Date.now();

  if (t && RELEASE_PATTERNS.some((re) => re.test(t))) {
    return {
      active: false,
      cleared: true,
      kind: null,
      strength: 0,
      reason: '他已表示可以聊/回来了',
      anchor: _clip(t, 80),
      untilMs: null,
      updatedAt: nowMs,
    };
  }

  let best = null;
  for (const sig of DND_SIGNALS) {
    for (const re of sig.patterns) {
      if (!re.test(corpus)) continue;
      const hitInCurrent = t && re.test(t);
      const candidate = {
        active: true,
        cleared: false,
        kind: sig.kind,
        strength: hitInCurrent ? sig.strength : sig.strength * 0.88,
        reason: `他${hitInCurrent ? '刚说' : '最近提到'}：${kindLabel(sig.kind)}`,
        anchor: _clip(hitInCurrent ? t : recent.find((line) => re.test(line)) || t, 100),
        untilMs: null,
        updatedAt: nowMs,
      };
      if (!best || candidate.strength > best.strength) best = candidate;
      break;
    }
  }

  if (!best) {
    return {
      active: false,
      cleared: false,
      kind: null,
      strength: 0,
      reason: '',
      anchor: '',
      untilMs: null,
      updatedAt: nowMs,
    };
  }

  const dur = parseDurationMs(corpus);
  const defMin = DND_SIGNALS.find((s) => s.kind === best.kind)?.defaultMinutes || 90;
  best.untilMs = nowMs + (dur != null ? dur : defMin * 60000);
  return best;
}

/**
 * @param {object|null|undefined} prev
 * @param {object} analysis analyzeUserPresence 返回值
 */
function mergePresenceState(prev, analysis) {
  const nowMs = Date.now();
  if (!analysis || analysis.cleared) {
    return null;
  }
  if (!analysis.active) {
    if (prev && prev.active && prev.untilMs && prev.untilMs > nowMs) {
      return { ...prev, updatedAt: nowMs };
    }
    return null;
  }
  const p = prev && prev.active ? prev : null;
  if (p && p.untilMs > nowMs && (analysis.strength || 0) < (p.strength || 0) * 0.85) {
    return { ...p, updatedAt: nowMs };
  }
  return {
    active: true,
    kind: analysis.kind,
    strength: analysis.strength,
    reason: analysis.reason,
    anchor: analysis.anchor || p?.anchor || '',
    untilMs: analysis.untilMs,
    updatedAt: nowMs,
  };
}

function isPresenceActive(state, nowMs = Date.now()) {
  if (!state || !state.active) return false;
  if (state.untilMs && state.untilMs <= nowMs) return false;
  return (state.strength || 0) >= 0.55;
}

/**
 * @param {object|null} state
 * @param {number} idleMin
 */
function shouldSuppressAutonomy(state, idleMin = 0) {
  if (!isPresenceActive(state)) return false;
  const kind = state.kind;
  const idle = Number(idleMin) || 0;
  if (kind === 'sleep') {
    return idle < 360;
  }
  if (kind === 'explicit' || kind === 'focus') {
    return idle < 90;
  }
  if (kind === 'away' || kind === 'social') {
    return idle < 50;
  }
  return idle < 40;
}

/**
 * @param {object|null} state
 * @param {number} idleMin
 */
function autonomySpeakMultiplier(state, idleMin = 0) {
  if (!isPresenceActive(state)) return 1;
  if (shouldSuppressAutonomy(state, idleMin)) return 0;
  const idle = Number(idleMin) || 0;
  if (state.kind === 'sleep' && idle >= 360) return 0.08;
  if (state.kind === 'low_energy' && idle >= 45) return 0.25;
  return 0.12;
}

/**
 * @param {object|null} state
 * @param {number} idleMin
 * @param {number} [relScore]
 */
function buildAutonomySituationForPresence(state, idleMin = 0, relScore = 0) {
  if (!isPresenceActive(state)) return '';
  const rel = Number(relScore) || 0;
  const idle = Number(idleMin) || 0;
  const anchor = state.anchor ? `他说过：「${_clip(state.anchor, 72)}」。` : '';
  const base = `${anchor}${state.reason || kindLabel(state.kind)}——`;

  if (state.kind === 'sleep') {
    if (idle >= 480 && rel > 0.35) {
      return `${base}若开口最多一句极轻的早安/睡得好没，禁止质问为什么不理你。`;
    }
    return `${base}禁止催回复、禁止「人呢/已读不回」；不要连发。`;
  }
  if (state.kind === 'focus') {
    if (idle >= 120 && rel > 0.4) {
      return `${base}最多一句「忙完了没」式短问，不要抱怨。`;
    }
    return `${base}当他去忙正事了；别催、别连发，不要质问不理你。`;
  }
  if (state.kind === 'away' || state.kind === 'social') {
    return `${base}人可能不在；别催已读，一句短的即可，禁止连发质问。`;
  }
  if (state.kind === 'explicit' || state.kind === 'low_energy') {
    return `${base}尊重边界；禁止「怎么不理我」，最多一句轻问，不要连发。`;
  }
  return `${base}尊重勿扰，别用抱怨式催回复。`;
}

/**
 * @param {object|null} state
 * @param {{ isAutonomy?: boolean, forReply?: boolean }} [opts]
 */
function buildPresencePromptBlock(state, opts = {}) {
  if (!isPresenceActive(state)) return '';
  const until = state.untilMs
    ? new Date(state.untilMs).toLocaleTimeString('zh', { hour: '2-digit', minute: '2-digit' })
    : '';
  const lines = [
    `【他的状态 · 勿扰】${state.reason || kindLabel(state.kind)}。`,
    state.anchor ? `依据：「${_clip(state.anchor, 90)}」` : '',
    until ? `预计至少到 ${until} 前都别催他回消息。` : '短时间内别催他回消息。',
  ];
  if (opts.isAutonomy) {
    lines.push('主动开口时：禁止「人呢/怎么不理我/已读不回」；可极短接上一句或一句关心，不要连发。');
  } else {
    lines.push('回复他：承认他在忙/要休息；不要抱怨他不理你，不要强行续聊。');
  }
  if (state.kind === 'sleep') {
    lines.push('睡觉场景：最多祝一句晚安或「去睡吧」，别展开新话题。');
  }
  return lines.filter(Boolean).join('\n');
}

function resolvePresenceFromDialogue(dialogue, nowMs = Date.now()) {
  const users = (dialogue || [])
    .filter((m) => m && m.role === 'user')
    .map((m) => String(m.content || '').trim())
    .filter((line) => line && !/^（想说话）|^（转移话题）|^（以下是最近对话/.test(line));
  if (!users.length) return null;
  const last = users[users.length - 1];
  const recent = users.slice(-6, -1);
  let state = null;
  for (let i = users.length - 1; i >= 0; i--) {
    const analysis = analyzeUserPresence(users[i], { recentUserLines: users.slice(0, i), nowMs });
    state = mergePresenceState(state, analysis);
    if (analysis.cleared) break;
  }
  if (!state && last) {
    state = mergePresenceState(null, analyzeUserPresence(last, { recentUserLines: recent, nowMs }));
  }
  return state;
}

const api = {
  analyzeUserPresence,
  mergePresenceState,
  isPresenceActive,
  shouldSuppressAutonomy,
  autonomySpeakMultiplier,
  buildAutonomySituationForPresence,
  buildPresencePromptBlock,
  resolvePresenceFromDialogue,
  kindLabel,
};

if (typeof module !== 'undefined' && module.exports) {
  module.exports = api;
}
if (typeof window !== 'undefined') {
  window.AmadeusUserPresence = api;
}
