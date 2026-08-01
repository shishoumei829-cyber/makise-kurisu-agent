'use strict';

const fs = require('fs');
const path = require('path');
const {
  ROOMS,
  classifyRoom,
  assessArchive,
  compressDeterministic,
  buildRecallableDetail,
  buildTopic,
} = require('./writeGate');
const {
  extractKeywords,
  retrieveFromRooms,
  formatRetrievalExcerpt,
} = require('./palaceRetrieve');

const MAX_PER_ROOM = 40;
const MAX_FORBIDDEN = 24;

function emptyRooms() {
  return { hall: [], lab: [], cafe: [], forbidden: [] };
}

class MemoryPalaceStore {
  constructor(dataDir = '') {
    this.dataDir = dataDir;
    this.path = dataDir ? path.join(dataDir, 'memory_palace.json') : '';
    this.state = {
      version: 2,
      rooms: emptyRooms(),
      proactiveBuffer: [],
      audits: [],
      updatedAt: 0,
    };
    this._load();
  }

  _load() {
    try {
      if (!this.path || !fs.existsSync(this.path)) return;
      const raw = JSON.parse(fs.readFileSync(this.path, 'utf8'));
      const rooms = emptyRooms();
      for (const room of ROOMS) {
        rooms[room] = Array.isArray(raw?.rooms?.[room]) ? raw.rooms[room] : [];
      }
      this.state = {
        version: 2,
        rooms,
        proactiveBuffer: Array.isArray(raw.proactiveBuffer) ? raw.proactiveBuffer : [],
        audits: Array.isArray(raw.audits) ? raw.audits.slice(-80) : [],
        updatedAt: Number(raw.updatedAt) || 0,
      };
    } catch { /* fresh */ }
  }

  _save() {
    if (!this.path) return;
    this.state.updatedAt = Date.now();
    try {
      if (this.dataDir && !fs.existsSync(this.dataDir)) {
        fs.mkdirSync(this.dataDir, { recursive: true });
      }
      fs.writeFileSync(this.path, JSON.stringify(this.state, null, 2));
    } catch (e) {
      console.warn('[palace] save failed:', e.message);
    }
  }

  _audit(action, detail) {
    this.state.audits.push({
      at: Date.now(),
      action,
      ...detail,
    });
    if (this.state.audits.length > 80) this.state.audits = this.state.audits.slice(-80);
  }

  bufferProactive(text, meta = {}) {
    const t = String(text || '').trim();
    if (!t) return null;
    const item = {
      id: `p_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
      text: t.slice(0, 200),
      ts: Number(meta.ts) || Date.now(),
      source: meta.source || 'autonomy',
    };
    this.state.proactiveBuffer.push(item);
    if (this.state.proactiveBuffer.length > 12) {
      this.state.proactiveBuffer = this.state.proactiveBuffer.slice(-12);
    }
    this._audit('proactive_buffer', { preview: t.slice(0, 40) });
    this._save();
    return item;
  }

  consumeProactiveOnUserReply() {
    const pending = this.state.proactiveBuffer.slice();
    this.state.proactiveBuffer = [];
    if (pending.length) {
      this._audit('proactive_consumed', { count: pending.length });
      this._save();
    }
    return pending;
  }

  hasPendingProactive() {
    return this.state.proactiveBuffer.length > 0;
  }

  archiveTurn(input = {}) {
    const decision = assessArchive(input);
    if (!decision.admit || !decision.allowPalace) {
      this._audit('archive_reject', {
        reason: decision.reason,
        preview: String(input.userText || '').slice(0, 40),
      });
      this._save();
      return { ok: false, ...decision, node: null };
    }

    const userText = String(input.userText || '').trim();
    const assistantText = String(input.assistantText || '').trim();
    const room = decision.room || classifyRoom(`${userText} ${assistantText}`);
    const label = String(input.compressed || '').trim() || compressDeterministic(userText, assistantText);
    const detail = String(input.detail || '').trim() || buildRecallableDetail(userText, assistantText);
    if (!detail && !label) {
      this._audit('archive_reject', { reason: 'empty_compress' });
      this._save();
      return { ok: false, admit: false, action: 'reject', reason: 'empty_compress', node: null };
    }

    const conflict = String(input.conflict || '').trim();
    const keywords = extractKeywords(userText, assistantText);
    const topic = buildTopic(userText, assistantText);
    const node = {
      id: `n_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
      text: conflict ? `[矛盾] ${label}` : label,
      detail: conflict ? `[矛盾] ${detail}` : detail,
      topic,
      keywords,
      user: userText.slice(0, 160),
      assistant: assistantText.slice(0, 160),
      room,
      source: input.proactive ? 'proactive_promoted' : 'chat',
      evidence: decision.reason,
      ts: Date.now(),
      conflict: conflict || '',
    };

    const list = this.state.rooms[room] || (this.state.rooms[room] = []);
    list.push(node);
    const cap = room === 'forbidden' ? MAX_FORBIDDEN : MAX_PER_ROOM;
    while (list.length > cap) list.shift();

    this._audit('archive_ok', { room, preview: node.detail.slice(0, 48) });
    this._save();
    return { ok: true, ...decision, room, node };
  }

  /**
   * 相关度检索（可靠召回的核心）
   */
  retrieve(query, opts = {}) {
    const preferredRoom = opts.preferredRoom || classifyRoom(query);
    return retrieveFromRooms(query, this.state.rooms, {
      topK: opts.topK,
      minScore: opts.minScore,
      preferredRoom,
      now: opts.now,
    });
  }

  /**
   * 召回摘录：只注入相关命中，不是「房间最近几条」
   */
  navigate(userInput, opts = {}) {
    const q = String(userInput || '').trim();
    const preferredRoom = classifyRoom(q);
    const hits = this.retrieve(q, {
      topK: Number(opts.topK) > 0 ? Number(opts.topK) : 6,
      minScore: Number.isFinite(Number(opts.minScore)) ? Number(opts.minScore) : 2.5,
      preferredRoom,
    });

    // 低相关时：仍给 hall 里最高分 1～2 条弱提示（若有分数）
    let used = hits;
    if (!used.length && q) {
      const weak = this.retrieve(q, { topK: 2, minScore: 1.2, preferredRoom });
      used = weak;
    }

    const excerpt = formatRetrievalExcerpt(used);
    return {
      room: preferredRoom,
      excerpt,
      hits: used.map((h) => ({
        room: h.room,
        score: Math.round(h.score * 100) / 100,
        detail: h.node.detail || h.node.text,
        topic: h.node.topic || '',
        id: h.node.id,
      })),
      counts: this.counts(),
    };
  }

  counts() {
    const out = {};
    for (const room of ROOMS) out[room] = (this.state.rooms[room] || []).length;
    out.total = Object.values(out).reduce((s, n) => s + n, 0);
    out.proactiveBuffer = this.state.proactiveBuffer.length;
    return out;
  }

  snapshot() {
    return {
      version: this.state.version,
      rooms: this.state.rooms,
      counts: this.counts(),
      proactiveBuffer: this.state.proactiveBuffer,
      recentAudits: this.state.audits.slice(-12),
      updatedAt: this.state.updatedAt,
    };
  }

  importLegacyRooms(legacy = {}) {
    let added = 0;
    for (const room of ROOMS) {
      const arr = Array.isArray(legacy[room]) ? legacy[room] : [];
      for (const item of arr) {
        const text = typeof item === 'string' ? item : String(item?.text || item?.detail || '');
        if (!text.trim()) continue;
        this.state.rooms[room].push({
          id: `mig_${Date.now()}_${added}`,
          text: text.slice(0, 80),
          detail: text.slice(0, 160),
          topic: '',
          keywords: extractKeywords(text, ''),
          user: '',
          assistant: '',
          room,
          source: 'legacy_localStorage',
          evidence: 'import',
          ts: Date.now(),
          conflict: '',
        });
        added += 1;
      }
      const cap = room === 'forbidden' ? MAX_FORBIDDEN : MAX_PER_ROOM;
      while (this.state.rooms[room].length > cap) this.state.rooms[room].shift();
    }
    if (added) {
      this._audit('import_legacy', { added });
      this._save();
    }
    return added;
  }

  clear() {
    this.state.rooms = emptyRooms();
    this.state.proactiveBuffer = [];
    this._audit('clear', {});
    this._save();
    return this.counts();
  }
}

module.exports = {
  MemoryPalaceStore,
  ROOMS,
};
