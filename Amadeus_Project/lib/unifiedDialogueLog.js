'use strict';

const fs = require('fs');
const path = require('path');

const MAX_ENTRIES = 800;
const MAX_AGE_MS = 10 * 86400000;
const MAX_TEXT_LEN = 480;

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
    .replace(/【意识广播[\s\S]*?(?:】|$)/g, '')
    .replace(/\[(?:打算|注意到|感受|想要|自我|关系|想起|自检)\]\s*/g, '')
    .replace(/我会先找话题[—\-~～]*然后就闲聊。?/g, '')
    .replace(/<[^>]+>/g, '')
    .trim();
}

const {
  gateAssistantReply,
  isDialoguePoison: isStructuralPoison,
} = require('./generationGate');

/** 内部工具输出绝不能进入角色对话、Prompt 或界面。 */
function isInternalControlLeak(text) {
  const t = String(text || '').trim();
  if (!t) return false;
  const profileFields = ['NAME:', 'TRAIT:', 'PREFER:', 'BASIC:'].filter((x) => t.includes(x)).length;
  return profileFields >= 2
    || /个人信息[:：]/.test(t)
    || /信息提取器[：:]/.test(t)
    || /NO_CONFLICT/.test(t)
    || /压[缩縮]为\s*[≤<]?\s*\d+\s*字标签/.test(t)
    || /只输出结果/.test(t)
    || /有矛盾输出中文/.test(t)
    || /^已有[：:]|\n新[：:]/.test(t)
    || /【意识广播/.test(t)
    || /\[打算\]/.test(t)
    || /我会先找话题/.test(t)
    || /本轮由内驱进入意识而开口/.test(t);
}

/**
 * 身份/人设毒句：走 generationGate 结构判定（非样本标签堆）。
 * 与内部工具泄漏分开——这类是「角色说了不该说的」。
 */
function isDialoguePoison(text, ctx = {}) {
  const t = String(text || '').trim();
  if (!t) return false;
  if (isInternalControlLeak(t)) return true;
  return isStructuralPoison(t, ctx);
}

/**
 * 写入前阀门：drop 拒绝；sanitize 用清洗后文本；pass 原样。
 */
function admitDialogueText(text, ctx = {}) {
  const t = String(text || '').trim();
  if (!t) return { ok: false, action: 'drop', text: '', reasons: ['empty'] };
  if (isInternalControlLeak(t)) {
    return { ok: false, action: 'drop', text: '', reasons: ['internal_leak'] };
  }
  if (ctx.role === 'user') {
    return { ok: true, action: 'pass', text: t, reasons: [] };
  }
  return gateAssistantReply(t, ctx);
}

function formatClock(ts) {
  const d = new Date(ts);
  return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;
}

/** @deprecated 使用 needsConversationRecall from unifiedDialogueLog */
function needsConversationRecall(userText) {
  const t = String(userText || '').trim();
  if (!t) return false;
  return /睡觉|睡了|去睡|补觉|睡醒|醒来|醒了|起来|核对|还记得|记得吗|记得我|你记得|我说过|你说过|有没有说|有没有提|我们.*说|今天.*说|刚才|之前说|那会儿|那会|几点|上午|下午|中午|晚上|今早|昨晚|记忆里|回忆|聊过|说过什么|你说了什么|我说了什么|喜欢喝|喝什么|怎么叫我|叫我什么|养的是什么|养什么|过敏|几点睡|明天去哪|总结一下|记成|是不是说/.test(t);
}

/**
 * 统一对话实录：唯一定稿源。
 * - 允许同侧连续条目（不强制 user/assistant 交替）
 * - toOllamaDialogue() 做合并层（相邻同 role 合并、裁切预算）
 */
class UnifiedDialogueLog {
  constructor(dataDir) {
    if (!fs.existsSync(dataDir)) {
      fs.mkdirSync(dataDir, { recursive: true });
    }
    this.logPath = path.join(dataDir, 'unified_dialogue_log.json');
    this.legacyPath = path.join(dataDir, 'conversation_log.json');
    this.entries = this._load();
    this._scheduleSave = debounceFileWrite(200, () => this._save());
    this._eventSink = null;
    if (this._loadRemovedCount > 0) {
      console.warn(`[dialogue-log] 启动时隔离 ${this._loadRemovedCount} 条旧污染台词`);
      this._scheduleSave();
    }
  }

  /** 统一事件源接线：每条定稿对话同步进入全局 journal */
  setEventSink(sink) {
    this._eventSink = typeof sink === 'function' ? sink : null;
  }

  _load() {
    const filterLoaded = (entries) => {
      const list = Array.isArray(entries) ? entries : [];
      const filtered = list.filter((x) => x && x.role && x.text && !isDialoguePoison(x.text, {
        role: x.role,
        autonomy: x.proactive || x.autonomy,
        proactive: x.proactive || x.autonomy,
      }));
      this._loadRemovedCount = (this._loadRemovedCount || 0) + (list.length - filtered.length);
      return filtered;
    };
    try {
      if (fs.existsSync(this.logPath)) {
        const raw = JSON.parse(fs.readFileSync(this.logPath, 'utf8'));
        if (Array.isArray(raw.entries)) {
          return filterLoaded(raw.entries);
        }
        if (Array.isArray(raw)) {
          return filterLoaded(raw);
        }
      }
      if (fs.existsSync(this.legacyPath)) {
        const legacy = JSON.parse(fs.readFileSync(this.legacyPath, 'utf8'));
        if (Array.isArray(legacy)) {
          return filterLoaded(legacy.map((t) => ({
            id: t.id || `mig_${t.ts || Date.now()}`,
            ts: t.ts || Date.now(),
            role: t.role === 'assistant' ? 'assistant' : 'user',
            text: normText(t.text).slice(0, MAX_TEXT_LEN),
            source: 'legacy_conversation_log',
          })));
        }
      }
    } catch { /* ignore */ }
    return [];
  }

  _save() {
    const payload = {
      version: 1,
      savedAt: Date.now(),
      entries: this.entries,
    };
    fs.writeFile(this.logPath, JSON.stringify(payload, null, 2), (err) => {
      if (err) console.error('[dialogue-log] Save error:', err.message);
    });
  }

  _prune() {
    const cutoff = Date.now() - MAX_AGE_MS;
    this.entries = this.entries.filter((e) => (e.ts || 0) >= cutoff);
    if (this.entries.length > MAX_ENTRIES) {
      this.entries = this.entries.slice(-MAX_ENTRIES);
    }
  }

  _isExactDup(role, text, ts = Date.now()) {
    const last = this.entries[this.entries.length - 1];
    if (!last) return false;
    return last.role === role
      && last.text === text
      && Math.abs((last.ts || 0) - ts) < 45000;
  }

  /**
   * 追加一条定稿（允许与上一条同侧）
   * @param {'user'|'assistant'} role
   */
  append(role, text, meta = {}) {
    const r = role === 'assistant' ? 'assistant' : 'user';
    let t = normText(text);
    if (!t || t.length < 1) return null;
    if (/^（想说话）|^（转移话题）|^（以下是最近对话/.test(t)) return null;
    const isProactive = Boolean(meta.proactive || meta.autonomy || meta.lite);
    if (
      r === 'assistant'
      && isProactive
      && this.hasUnansweredUserTurn(meta.conversationId || '')
    ) {
      return null;
    }
    const admitted = admitDialogueText(t, {
      role: r,
      autonomy: Boolean(meta.autonomy || meta.proactive || meta.lite),
      proactive: Boolean(meta.proactive || meta.autonomy),
    });
    if (!admitted.ok || admitted.action === 'drop') return null;
    t = normText(admitted.text || t);
    if (!t) return null;
    const ts = meta.ts || Date.now();
    if (this._isExactDup(r, t.slice(0, MAX_TEXT_LEN), ts)) return null;
    const turnId = String(meta.turnId || '');
    const conversationId = String(meta.conversationId || '');
    if (r === 'assistant' && turnId) {
      if (this.entries.some((entry) => entry.role === 'assistant' && entry.turnId === turnId)) return null;
      const ownerIndex = this.entries.findLastIndex((entry) => entry.role === 'user' && entry.turnId === turnId);
      if (ownerIndex < 0) return null;
      const newerForeignUser = this.entries.slice(ownerIndex + 1).some((entry) => (
        entry.role === 'user'
        && entry.conversationId === conversationId
        && entry.turnId
        && entry.turnId !== turnId
      ));
      if (newerForeignUser) return null;
    }

    const item = {
      id: meta.id || `dlg_${ts}_${r}_${Math.random().toString(36).slice(2, 7)}`,
      ts,
      role: r,
      text: t.slice(0, MAX_TEXT_LEN),
      proactive: Boolean(meta.proactive),
      autonomy: Boolean(meta.autonomy),
      lite: Boolean(meta.lite),
      source: meta.source || 'chat',
      conversationId,
      turnId,
      modelJp: r === 'assistant' && meta.modelJp
        ? String(meta.modelJp).slice(0, MAX_TEXT_LEN)
        : '',
    };
    this.entries.push(item);
    this._prune();
    this._scheduleSave();
    if (this._eventSink) {
      try { this._eventSink(item); } catch { /* 事实流写入失败不影响对话主链 */ }
    }
    return item;
  }

  /** 兼容旧 ConversationMemory.addTurn */
  addTurn(role, text, ts) {
    return this.append(role, text, { ts });
  }

  get entriesCount() {
    return this.entries.length;
  }

  get turns() {
    return this.entries.map((e) => ({ role: e.role, text: e.text, ts: e.ts }));
  }

  getRecent(n = 20) {
    return this.entries.filter((e) => !isDialoguePoison(e.text)).slice(-Math.max(1, n));
  }

  syncFromDialogue(dialogue) {
    const list = Array.isArray(dialogue) ? dialogue : [];
    if (!list.length) return 0;
    let added = 0;
    const existing = new Set(this.entries.map((x) => `${x.role}:${x.text}`));
    const now = Date.now();
    let ts = now - list.length * 40000;
    for (const m of list) {
      if (!m || !m.content) continue;
      const role = m.role === 'user' ? 'user' : 'assistant';
      const text = normText(m.content).slice(0, MAX_TEXT_LEN);
      if (!text || /^（想说话）/.test(text)) continue;
      const key = `${role}:${text}`;
      if (existing.has(key)) continue;
      this.append(role, text, { ts, source: 'sync' });
      existing.add(key);
      added += 1;
      ts += 40000;
    }
    return added;
  }

  purgeInternalLeaks() {
    const before = this.entries.length;
    this.entries = this.entries.filter((e) => {
      const text = String(e?.text || '').trim();
      if (!text) return false;
      if (isDialoguePoison(text)) return false;
      return true;
    });
    const removed = before - this.entries.length;
    if (removed > 0) this._save();
    return removed;
  }

  /** 清空全部实录（测试/用户重置） */
  clearAll() {
    const removed = this.entries.length;
    this.entries = [];
    this._save();
    return removed;
  }

  /**
   * 是否存在「用户发了 chat 但尚无正式回复」的 turn。
   * proactive/autonomy 不计入回复。
   */
  hasUnansweredUserTurn(conversationId = '') {
    const cid = String(conversationId || '').trim();
    for (let i = this.entries.length - 1; i >= 0; i -= 1) {
      const e = this.entries[i];
      if (e.role !== 'user') continue;
      if (cid && e.conversationId && e.conversationId !== cid) continue;
      const text = normText(e.text);
      if (!text || text === '（转移话题）') continue;
      const turnId = String(e.turnId || '').trim();
      if (!turnId) return true;
      const answered = this.entries.some((x) => (
        x.role === 'assistant'
        && x.turnId === turnId
        && !x.proactive
        && !x.autonomy
        && !/autonomy|conversation_/.test(String(x.source || ''))
      ));
      if (!answered) return true;
      return false;
    }
    return false;
  }

  getTurnsInWindow({ hours = 14, sinceStartOfDay = false } = {}) {
    const now = Date.now();
    let since = now - Math.max(1, hours) * 3600000;
    if (sinceStartOfDay) {
      const d = new Date();
      d.setHours(0, 0, 0, 0);
      since = Math.min(since, d.getTime());
    }
    return this.entries.filter((t) => (t.ts || 0) >= since);
  }

  /**
   * 合并层：供 Ollama 多轮对话使用
   * - 相邻同 role 合并为一条（用空格连接）
   * - 裁切到 maxMsgs
   */
  toOllamaDialogue(opts = {}) {
    const maxMsgs = Number(opts.maxMsgs) > 0 ? Math.floor(opts.maxMsgs) : 24;
    const conversationId = String(opts.conversationId || '').trim();
    const sourcePool = opts.entries || this.entries;
    const pool = conversationId
      ? sourcePool.filter((entry) => String(entry?.conversationId || '').trim() === conversationId)
      : sourcePool;
    // 主动开口默认不进多轮历史：只保留末尾少量，避免凌晨连发毒化接话
    const includeProactive = opts.includeProactive === true;
    const proactiveTail = Number(opts.proactiveTail) >= 0 ? Math.floor(opts.proactiveTail) : 2;
    let proactiveKept = 0;
    const filtered = [];
    for (let i = pool.length - 1; i >= 0; i--) {
      const e = pool[i];
      const content = normText(e?.text);
      if (!content || isDialoguePoison(content)) continue;
      const isPro = e.proactive === true || e.autonomy === true || /autonomy|copresence|conversation_/.test(String(e.source || ''));
      if (isPro && !includeProactive) {
        if (proactiveKept >= proactiveTail) continue;
        proactiveKept += 1;
      }
      filtered.push(e);
    }
    filtered.reverse();
    const merged = [];
    for (const e of filtered) {
      const role = e.role === 'user' ? 'user' : 'assistant';
      const content = normText(e.text);
      if (!content) continue;
      const last = merged[merged.length - 1];
      if (last && last.role === role) {
        last.content = `${last.content} ${content}`.trim().slice(0, MAX_TEXT_LEN * 2);
      } else {
        merged.push({ role, content });
      }
    }
    if (merged.length > maxMsgs) return merged.slice(-maxMsgs);
    return merged;
  }

  toRecentContextLines(n = 8) {
    return this.getRecent(n).map((e) => {
      const who = e.role === 'user' ? '他' : 'Kurisu';
      return `${who}: ${e.text}`;
    }).join('\n');
  }

  toPromptBlock(opts = {}) {
    const hours = Number(opts.hours) > 0 ? Number(opts.hours) : 14;
    const maxChars = Number(opts.maxChars) > 200 ? Number(opts.maxChars) : 1600;
    const sinceStartOfDay = opts.sinceStartOfDay !== false;
    const userText = String(opts.userText || '');

    const conversationId = String(opts.conversationId || '').trim();
    let pool = this.getTurnsInWindow({ hours, sinceStartOfDay });
    if (conversationId) {
      pool = pool.filter((entry) => String(entry?.conversationId || '').trim() === conversationId);
    }
    if (!pool.length) return '';

    if (userText && needsConversationRecall(userText)) {
      const toks = (userText.match(/[\u4e00-\u9fa5A-Za-z0-9]{2,}/g) || [])
        .filter((x) => x.length >= 2)
        .slice(0, 12);
      if (toks.length) {
        const scored = pool.map((turn, idx) => {
          let hit = 0;
          for (const tok of toks) {
            if ((turn.text || '').includes(tok)) hit += 1;
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
      if (isDialoguePoison(turn.text)) continue;
      const who = turn.role === 'user' ? '他' : 'Kurisu';
      const line = `[${formatClock(turn.ts)}] ${who}: ${turn.text}\n`;
      if (used + line.length > maxChars) break;
      lines.push(line);
      used += line.length;
    }

    if (!lines.length) return '';
    return header + lines.join('').trim();
  }

  snapshot() {
    return {
      count: this.entries.length,
      recent: this.getRecent(6).map((e) => ({
        role: e.role,
        text: e.text.slice(0, 80),
        ts: e.ts,
        proactive: e.proactive,
      })),
    };
  }
}

module.exports = {
  UnifiedDialogueLog,
  isInternalControlLeak,
  isDialoguePoison,
  admitDialogueText,
  needsConversationRecall,
  normText,
};
