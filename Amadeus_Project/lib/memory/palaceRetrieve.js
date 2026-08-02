'use strict';

/**
 * 宫殿检索：按查询相关度对节点打分，取 top-k。
 * 目标：问得到、找得准——不是「最近几条塞进 prompt」。
 */

const STOP = new Set('的了吗呢吧啊呀哦嗯在是有和与就都也又很还把被让给到从对为以而及或与这那我你他她它'.split(''));

function tokenize(text) {
  const t = String(text || '').toLowerCase();
  const out = new Set();
  const latin = t.match(/[a-z0-9]{2,}/g) || [];
  for (const w of latin) out.add(w);
  const han = t.replace(/[^\u4e00-\u9fff]/g, '');
  for (let i = 0; i < han.length; i++) {
    const c = han[i];
    if (!STOP.has(c)) out.add(c);
    if (i + 1 < han.length) {
      const bigram = han.slice(i, i + 2);
      if (![...bigram].every((x) => STOP.has(x))) out.add(bigram);
    }
    if (i + 2 < han.length) out.add(han.slice(i, i + 3));
  }
  return [...out].filter((x) => x.length >= 1);
}

function extractKeywords(userText, assistantText = '') {
  const merged = `${userText} ${assistantText}`;
  const toks = tokenize(merged)
    .filter((t) => t.length >= 2)
    .sort((a, b) => b.length - a.length);
  const picked = [];
  for (const t of toks) {
    if (picked.some((p) => p.includes(t) && p !== t)) continue;
    picked.push(t);
    if (picked.length >= 12) break;
  }
  return picked;
}

function nodeSearchText(node) {
  if (!node) return '';
  if (typeof node === 'string') return node;
  return [
    node.text,
    node.detail,
    node.topic,
    node.user,
    node.assistant,
    ...(Array.isArray(node.keywords) ? node.keywords : []),
  ].filter(Boolean).join(' ');
}

/**
 * @param {string} query
 * @param {object} node
 * @param {{ preferredRoom?: string, now?: number }} [opts]
 */
function scoreNode(query, node, opts = {}) {
  const q = String(query || '').trim();
  if (!q || !node) return 0;
  const hay = nodeSearchText(node);
  if (!hay) return 0;

  const qTokens = tokenize(q);
  if (!qTokens.length) return 0;

  let score = 0;
  let hits = 0;
  for (const tok of qTokens) {
    if (hay.includes(tok)) {
      hits += 1;
      score += tok.length >= 3 ? 3 : tok.length >= 2 ? 2 : 0.5;
    }
  }
  if (hits === 0) return 0;

  // 整段子串命中（如「胡椒博士」「量子纠缠」）
  const compactQ = q.replace(/\s+/g, '');
  for (let len = Math.min(8, compactQ.length); len >= 3; len--) {
    for (let i = 0; i <= compactQ.length - len; i++) {
      const slice = compactQ.slice(i, i + len);
      if (STOP.has(slice)) continue;
      if (hay.includes(slice)) {
        score += len;
        break;
      }
    }
  }

  const room = node.room || opts.nodeRoom;
  if (opts.preferredRoom && room === opts.preferredRoom) score += 1.5;
  if (room === 'hall') score += 0.3;

  const ageMs = Math.max(0, (opts.now || Date.now()) - (Number(node.ts) || 0));
  const ageDays = ageMs / 86400000;
  score *= Math.max(0.55, 1 - ageDays * 0.02);

  // 命中覆盖率
  score *= 0.7 + 0.3 * Math.min(1, hits / Math.max(2, Math.min(qTokens.length, 8)));
  return score;
}

/**
 * @param {string} query
 * @param {object} rooms  { hall:[], lab:[]... }
 * @param {{ topK?: number, minScore?: number, preferredRoom?: string }} [opts]
 */
function retrieveFromRooms(query, rooms = {}, opts = {}) {
  const topK = Number(opts.topK) > 0 ? Math.floor(opts.topK) : 6;
  const minScore = Number.isFinite(Number(opts.minScore)) ? Number(opts.minScore) : 2.5;
  const now = Number(opts.now) || Date.now();
  const preferredRoom = opts.preferredRoom || '';
  const scored = [];

  for (const [room, list] of Object.entries(rooms || {})) {
    if (!Array.isArray(list)) continue;
    for (const node of list) {
      const n = typeof node === 'string'
        ? { text: node, room, ts: now }
        : { ...node, room: node.room || room };
      const score = scoreNode(query, n, { preferredRoom, now, nodeRoom: room });
      if (score >= minScore) scored.push({ node: n, room, score });
    }
  }

  scored.sort((a, b) => b.score - a.score || (b.node.ts || 0) - (a.node.ts || 0));
  return scored.slice(0, topK);
}

function formatRetrievalExcerpt(hits) {
  if (!hits.length) return '';
  const byRoom = new Map();
  for (const h of hits) {
    const room = h.room || 'hall';
    if (!byRoom.has(room)) byRoom.set(room, []);
    byRoom.get(room).push(h);
  }
  const labels = { hall: 'HALL', lab: 'LAB', cafe: 'CAFÉ', forbidden: 'FORBIDDEN' };
  const parts = [];
  for (const [room, list] of byRoom) {
    const lines = list.map((h) => {
      const n = h.node;
      const detail = String(n.detail || n.text || '').trim();
      return `- ${detail}`;
    });
    parts.push(`[${labels[room] || room.toUpperCase()}]\n${lines.join('\n')}`);
  }
  return parts.join('\n\n');
}

module.exports = {
  tokenize,
  extractKeywords,
  nodeSearchText,
  scoreNode,
  retrieveFromRooms,
  formatRetrievalExcerpt,
};
