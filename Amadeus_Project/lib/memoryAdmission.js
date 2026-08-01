'use strict';

const fs = require('fs');
const path = require('path');

const INTERNAL_CONTROL = /(?:^（想说话）|^（转移话题）|NO_CONFLICT|压[缩縮]为\s*[≤<]?\s*\d+\s*字标签|只输出结果|信息提取器|个人信息[:：]|NAME:.*TRAIT:)/i;
const META_POLLUTION = /刚才那句不算|我重新说|微调|说话方式|对话模式|我们这边|已经(?:好好)?调整|语言更差|解释自己(?:的)?说话|作为(?:一个)?AI|语言模型|智能助手/;
const EXPLICIT_PROFILE = /(?:我(?:很|最|一直|通常|平时){0,2}(?:喜欢|讨厌|不喜欢|偏爱|习惯|从不|总是)|我的(?:名字|工作|职业|生日|家乡|专业|爱好)|我叫|我是(?:做|学|来自)|请记住|记住我|以后叫我)/;
const LIVED_EVENT = /(?:准备|打算|计划|决定|刚刚|刚才|回来|到了|出门|要去|去了|正在|今天|明天|今晚|健身|运动|上班|下班|吃饭|睡觉|生病|感冒|发烧|疼|痛|流血|鼻血|医院|吃药|累死|困死)/;
const RELATIONAL_EVENT = /(?:喜欢我|爱我|想我|在乎我|陪我|抱抱|情侣|恋人|女朋友|男朋友|不理我|吵架|和好|吃醋|傲娇)/;
const SUBSTANTIVE = /(?:我觉得|我认为|我发现|我最近|其实我|因为|所以|但是|不过|为什么|怎么回事|本质|原理|如果|假如|担心|害怕|难过|焦虑|计划|打算|决定)/;
const CASUAL_COMMAND = /^(?:去|来|看|听|玩|刷|吃|喝|睡|走|打开|关掉|开始|继续|算了|随便|聊聊|说说)[^。！？!?]{0,8}(?:吧|啊|呀|呗|呢)?[。！？!?]?$/;

function normalizeText(value) {
  return String(value || '').replace(/\s+/g, '').trim();
}

function textFragments(value) {
  const text = normalizeText(value)
    .replace(/[，。！？!?；;、…“”‘’（）()\[\]【】]/g, '')
    .replace(/(?:我|你|他|她|吧|啊|呀|呢|了|的|是|在|和|就|都|也|又|很|这|那|一个)/g, '');
  if (text.length < 3) return text ? [text] : [];
  const out = new Set();
  for (let size = 3; size <= Math.min(5, text.length); size += 1) {
    for (let i = 0; i <= text.length - size; i += 1) out.add(text.slice(i, i + size));
  }
  return [...out].slice(0, 40);
}

class MemoryAdmissionPolicy {
  constructor(dataDir = '') {
    this.statePath = dataDir ? path.join(dataDir, 'memory_admission.json') : '';
    this.state = { version: 1, evidence: {}, quarantined: {}, audits: [] };
    this._saveTimer = null;
    this._load();
  }

  _load() {
    try {
      if (!this.statePath || !fs.existsSync(this.statePath)) return;
      const raw = JSON.parse(fs.readFileSync(this.statePath, 'utf8'));
      this.state = { ...this.state, ...raw, evidence: raw.evidence || {}, quarantined: raw.quarantined || {} };
      this._prune();
    } catch { /* fresh policy state */ }
  }

  _save() {
    if (!this.statePath) return;
    if (this._saveTimer) clearTimeout(this._saveTimer);
    this._saveTimer = setTimeout(() => {
      this._saveTimer = null;
      try { fs.writeFileSync(this.statePath, JSON.stringify(this.state, null, 2)); } catch { /* non-blocking */ }
    }, 150);
  }

  _prune(now = Date.now()) {
    const cutoff = now - 30 * 86400000;
    for (const [key, entry] of Object.entries(this.state.evidence)) {
      if (!entry || Number(entry.lastSeen) < cutoff) delete this.state.evidence[key];
    }
    for (const [key, until] of Object.entries(this.state.quarantined)) {
      if (Number(until) <= now) delete this.state.quarantined[key];
    }
    const entries = Object.entries(this.state.evidence);
    if (entries.length > 500) {
      entries.sort((a, b) => Number(b[1].lastSeen) - Number(a[1].lastSeen));
      this.state.evidence = Object.fromEntries(entries.slice(0, 500));
    }
    this.state.audits = (this.state.audits || []).slice(-120);
  }

  observe(source, text, meta = {}) {
    const normalized = normalizeText(text);
    if (!normalized || INTERNAL_CONTROL.test(normalized)) return [];
    const kind = source === 'user' ? 'user' : source === 'proactive' ? 'proactive' : 'assistant';
    const now = Number(meta.now) || Date.now();
    const newlyQuarantined = [];
    const metaHit = META_POLLUTION.test(normalized);
    for (const fragment of textFragments(normalized)) {
      const entry = this.state.evidence[fragment] || { user: 0, assistant: 0, proactive: 0, explicitUser: 0, lastSeen: 0 };
      entry[kind] = Number(entry[kind] || 0) + 1;
      if (kind === 'user' && EXPLICIT_PROFILE.test(normalized)) {
        entry.explicitUser += 1;
        delete this.state.quarantined[fragment];
      }
      entry.lastSeen = now;
      this.state.evidence[fragment] = entry;
      if (metaHit && kind !== 'user') {
        if (!this.state.quarantined[fragment]) newlyQuarantined.push(fragment);
        this.state.quarantined[fragment] = now + 30 * 86400000;
        continue;
      }
      if (entry.proactive >= 2 && entry.user <= 1 && entry.explicitUser === 0) {
        if (!this.state.quarantined[fragment]) newlyQuarantined.push(fragment);
        this.state.quarantined[fragment] = now + 7 * 86400000;
      }
    }
    this._prune(now);
    this._save();
    return newlyQuarantined;
  }

  contaminatedFragments(text, now = Date.now()) {
    return textFragments(text).filter((fragment) => Number(this.state.quarantined[fragment] || 0) > now);
  }

  assessUserText(text, meta = {}) {
    const normalized = normalizeText(text);
    const source = String(meta.source || 'user');
    const synthetic = source !== 'user' || meta.synthetic === true || INTERNAL_CONTROL.test(normalized);
    if (!normalized || synthetic) return this._decision('reject', 'non_user_or_internal', false, false, false, false, normalized);

    const explicitProfile = EXPLICIT_PROFILE.test(normalized);
    if (explicitProfile) return this._decision('durable', 'explicit_user_fact', true, true, true, true, normalized);

    const contaminated = this.contaminatedFragments(normalized);
    if (contaminated.length) {
      return this._decision('working', 'topic_quarantined', false, false, false, false, normalized, contaminated);
    }

    const compactLength = normalized.length;
    const livedEvent = LIVED_EVENT.test(normalized);
    const relationalEvent = RELATIONAL_EVENT.test(normalized);
    if (livedEvent || relationalEvent) {
      return this._decision(
        'episodic',
        relationalEvent ? 'relationship_event' : 'lived_event',
        true,
        false,
        true,
        true,
        normalized,
      );
    }
    if (compactLength <= 4 || CASUAL_COMMAND.test(normalized)) {
      return this._decision('working', 'short_casual_utterance', false, false, false, false, normalized);
    }

    const substantive = SUBSTANTIVE.test(normalized) || compactLength >= 10;
    if (substantive) return this._decision('episodic', 'substantive_user_turn', true, false, true, true, normalized);
    return this._decision('working', 'insufficient_evidence', false, false, false, false, normalized);
  }

  _decision(tier, reason, allowEvent, allowProfile, allowCuriosity, allowInference, text, contaminated = []) {
    const decision = { tier, reason, allowEvent, allowProfile, allowCuriosity, allowInference, text, contaminated };
    this.state.audits.push({ at: Date.now(), tier, reason, preview: String(text || '').slice(0, 48) });
    this._prune();
    this._save();
    return decision;
  }

  snapshot() {
    this._prune();
    return {
      version: this.state.version,
      evidenceCount: Object.keys(this.state.evidence).length,
      quarantined: Object.keys(this.state.quarantined),
      recentAudits: this.state.audits.slice(-12),
    };
  }
}

module.exports = {
  MemoryAdmissionPolicy,
  normalizeText,
  textFragments,
  INTERNAL_CONTROL,
};
