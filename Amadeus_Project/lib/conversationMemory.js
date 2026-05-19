'use strict';

const fs = require('fs');
const path = require('path');

const MAX_TURNS = 600;
const MAX_AGE_MS = 8 * 86400000;
const MAX_TEXT_LEN = 420;

function debounceFileWrite(ms, fn) {
  let t = null;
  return () => {
    if (t) clearTimeout(t);
    t = setTimeout(() => {
      t = null;
      fn();
    }, ms);
  };
}

function normText(s) {
  return String(s || '')
    .replace(/\r\n/g, '\n')
    .replace(/\n\s*JP\s*[:：][\s\S]*$/i, '')
    .replace(/<[^>]+>/g, '')
    .trim();
}

function formatClock(ts) {
  const d = new Date(ts);
  const h = String(d.getHours()).padStart(2, '0');
  const m = String(d.getMinutes()).padStart(2, '0');
  return `${h}:${m}`;
}

/** 是否在核对/回忆/睡眠作息类问题 */
function needsConversationRecall(userText) {
  const t = String(userText || '').trim();
  if (!t) return false;
  return /睡觉|睡了|去睡|补觉|睡醒|醒来|醒了|起来|核对|还记得|记得吗|记得我|你记得|我说过|你说过|有没有说|有没有提|我们.*说|今天.*说|刚才|之前说|那会儿|那会|几点|上午|下午|中午|晚上|今早|昨晚|记忆里|回忆|聊过|说过什么|你说了什么|我说了什么/.test(t);
}

class ConversationMemory {
  constructor(dataDir) {
    if (!fs.existsSync(dataDir)) {
      fs.mkdirSync(dataDir, { recursive: true });
    }
    this.logPath = path.join(dataDir, 'conversation_log.json');
    this.turns = this._load();
    this._scheduleSave = debounceFileWrite(250, () => this._save());
  }

  _load() {
    try {
      if (fs.existsSync(this.logPath)) {
        const raw = JSON.parse(fs.readFileSync(this.logPath, 'utf8'));
        return Array.isArray(raw) ? raw.filter((x) => x && x.role && x.text) : [];
      }
    } catch {}
    return [];
  }

  _save() {
    fs.writeFile(this.logPath, JSON.stringify(this.turns, null, 2), (err) => {
      if (err) console.error('[conversation] Save error:', err.message);
    });
  }

  _prune() {
    const cutoff = Date.now() - MAX_AGE_MS;
    this.turns = this.turns.filter((t) => (t.ts || 0) >= cutoff);
    if (this.turns.length > MAX_TURNS) {
      this.turns = this.turns.slice(-MAX_TURNS);
    }
  }

  _isDup(role, text, ts = Date.now()) {
    const last = this.turns[this.turns.length - 1];
    if (!last) return false;
    return last.role === role
      && last.text === text
      && Math.abs((last.ts || 0) - ts) < 120000;
  }

  addTurn(role, text, ts = Date.now()) {
    const r = role === 'assistant' ? 'assistant' : 'user';
    const t = normText(text);
    if (!t || t.length < 1) return null;
    if (/^（想说话）|^（转移话题）|^（以下是最近对话/.test(t)) return null;
    if (this._isDup(r, t.slice(0, MAX_TEXT_LEN), ts)) return null;

    const item = {
      id: `${ts}_${r}_${Math.random().toString(36).slice(2, 7)}`,
      ts,
      role: r,
      text: t.slice(0, MAX_TEXT_LEN),
    };
    this.turns.push(item);
    this._prune();
    this._scheduleSave();
    return item;
  }

  /** 用前端多轮历史补全尚未落盘的轮次（仅补缺，不重复） */
  syncFromDialogue(dialogue) {
    const list = Array.isArray(dialogue) ? dialogue : [];
    if (!list.length) return 0;

    const existing = new Set(this.turns.map((x) => `${x.role}:${x.text}`));
    const missing = [];
    for (const m of list) {
      if (!m || !m.content) continue;
      const role = m.role === 'user' ? 'user' : 'assistant';
      const text = normText(m.content).slice(0, MAX_TEXT_LEN);
      if (!text || /^（想说话）/.test(text)) continue;
      const key = `${role}:${text}`;
      if (existing.has(key)) continue;
      missing.push({ role, text });
      existing.add(key);
    }
    if (!missing.length) return 0;

    const now = Date.now();
    const step = 50000;
    let ts = now - missing.length * step;
    for (const item of missing) {
      this.addTurn(item.role, item.text, ts);
      ts += step;
    }
    return missing.length;
  }

  getTurnsInWindow({ hours = 14, sinceStartOfDay = false } = {}) {
    const now = Date.now();
    let since = now - Math.max(1, hours) * 3600000;
    if (sinceStartOfDay) {
      const d = new Date();
      d.setHours(0, 0, 0, 0);
      since = Math.min(since, d.getTime());
    }
    return this.turns.filter((t) => (t.ts || 0) >= since);
  }

  toPromptBlock(opts = {}) {
    const hours = Number(opts.hours) > 0 ? Number(opts.hours) : 14;
    const maxChars = Number(opts.maxChars) > 200 ? Number(opts.maxChars) : 1400;
    const sinceStartOfDay = opts.sinceStartOfDay !== false;
    const userText = String(opts.userText || '');

    let pool = this.getTurnsInWindow({ hours, sinceStartOfDay });
    if (!pool.length) return '';

    if (userText && needsConversationRecall(userText)) {
      const toks = (userText.match(/[\u4e00-\u9fa5A-Za-z0-9]{2,}/g) || [])
        .filter((x) => x.length >= 2)
        .slice(0, 12);
      if (toks.length) {
        const scored = pool.map((turn, idx) => {
          const doc = turn.text || '';
          let hit = 0;
          for (const tok of toks) {
            if (doc.includes(tok)) hit++;
          }
          return { turn, idx, hit };
        });
        const relevant = scored.filter((x) => x.hit > 0).sort((a, b) => b.hit - a.hit);
        if (relevant.length) {
          const pickIdx = new Set();
          for (const r of relevant.slice(0, 16)) {
            for (let j = Math.max(0, r.idx - 1); j <= Math.min(pool.length - 1, r.idx + 1); j++) {
              pickIdx.add(j);
            }
          }
          pool = [...pickIdx].sort((a, b) => a - b).map((i) => pool[i]);
        }
      }
    }

    const lines = [];
    let used = 0;
    const header = '【今日对话实录】按时间列出你们说过的话（核对「我说过/你说过」时以此为准，有记录就承认，勿说没有）：\n';
    used += header.length;

    for (const turn of pool) {
      const who = turn.role === 'user' ? '他' : 'Kurisu';
      const line = `[${formatClock(turn.ts)}] ${who}: ${turn.text}\n`;
      if (used + line.length > maxChars) break;
      lines.push(line);
      used += line.length;
    }

    if (!lines.length) return '';
    return header + lines.join('').trim();
  }
}

module.exports = {
  ConversationMemory,
  needsConversationRecall,
  normText,
};
