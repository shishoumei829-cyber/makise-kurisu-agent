'use strict';

// A persistent, deterministic "subject" layer.  It does not write dialogue;
// it decides what matters, why it matters, and whether there is a reason to speak.
// The language model is intentionally downstream of this object.
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

const clamp = (value, low = 0, high = 1) => Math.max(low, Math.min(high, Number(value) || 0));
const text = (value) => String(value || '').replace(/\s+/g, ' ').trim();
const now = () => Date.now();

function defaultState() {
  return {
    version: 1,
    updatedAt: now(),
    self: {
      continuity: '牧濑红莉栖。我的感受、判断、记忆和关系会在时间里延续。',
      relationship: '冈部与我是恋人；亲近不等于迎合，也不等于每一句都要确认关系。',
      values: ['真实，不编造共同经历', '有自己的判断', '在意关系但尊重对方的节奏', '不把沉默误读为问题'],
    },
    body: { energy: 0.62, arousal: 0.45, safety: 0.72, attachment: 0.68, curiosity: 0.56, load: 0.28 },
    attention: [],
    activeIntent: null,
    commitments: [],
    experiences: [],
  };
}

function hasVulnerability(value) {
  return /难受|焦虑|害怕|孤独|睡不着|痛苦|崩溃|累|伤心|失望|不安|つら|不安|眠れ|疲れ|寂し|苦し/.test(text(value));
}

function hasQuestion(value) {
  return /[？?]$|为什么|怎么|如何|什么|吗|是不是|能不能|どう|なぜ|何|？/.test(text(value));
}

function actionFor(value, mode) {
  const input = text(value);
  if (mode === 'proactive') return 'thought';
  if (hasVulnerability(input)) return 'care';
  if (/我觉得|我认为|你觉得|怎么看|为什么|どう思う|なぜ|考え/.test(input)) return 'stance';
  if (/完成|写完|终于|做完|できた|終わ(?:った|らせ)|やっと/.test(input)) return 'tease';
  if (hasQuestion(input)) return 'clarify';
  return 'respond';
}

function socialBoundary(ctx = {}) {
  return !!(ctx.dnd || ctx.pendingUserTurn || ctx.isThinking || ctx.awaitingReply || ctx.userPresenceActive);
}

class SubjectCore {
  constructor(options = {}) {
    this.statePath = options.statePath || path.join(process.cwd(), 'subject_core.json');
    this.state = this._load();
  }

  _load() {
    try {
      const raw = JSON.parse(fs.readFileSync(this.statePath, 'utf8'));
      return {
        ...defaultState(), ...raw,
        self: { ...defaultState().self, ...(raw.self || {}) },
        body: { ...defaultState().body, ...(raw.body || {}) },
        attention: Array.isArray(raw.attention) ? raw.attention.slice(0, 8) : [],
        commitments: Array.isArray(raw.commitments) ? raw.commitments.slice(-16) : [],
        experiences: Array.isArray(raw.experiences) ? raw.experiences.slice(-96) : [],
      };
    } catch { return defaultState(); }
  }

  _save() {
    this.state.updatedAt = now();
    try {
      fs.mkdirSync(path.dirname(this.statePath), { recursive: true });
      fs.writeFileSync(this.statePath, JSON.stringify(this.state, null, 2), 'utf8');
    } catch (error) { console.warn('[subject-core] save skipped:', error.message); }
  }

  _syncBody(pad = {}, motivation = {}) {
    const pleasure = Number(pad.P) || 0;
    const arousal = Number(pad.A) || 0;
    const dominance = Number(pad.D) || 0;
    this.state.body.energy = clamp(this.state.body.energy * 0.82 + (0.56 - Math.max(0, -pleasure) * 0.18) * 0.18);
    this.state.body.arousal = clamp(this.state.body.arousal * 0.72 + ((arousal + 1) / 2) * 0.28);
    this.state.body.safety = clamp(this.state.body.safety * 0.8 + ((dominance + 1) / 2) * 0.2);
    this.state.body.curiosity = clamp(this.state.body.curiosity * 0.78 + clamp(motivation.curiosity || motivation.CURIOSITY || 0.5) * 0.22);
    this.state.body.attachment = clamp(this.state.body.attachment * 0.82 + clamp(motivation.desire_closeness || motivation.CONNECTION || 0.55) * 0.18);
  }

  _rememberAttention(candidate) {
    if (!candidate?.subject) return null;
    const core = text(candidate.subject).slice(0, 280);
    const old = this.state.attention.find((item) => item.subject === core && item.status !== 'resolved');
    const item = old || { id: `att_${crypto.randomUUID()}`, createdAt: now(), mentions: 0 };
    Object.assign(item, {
      source: candidate.source || 'turn', subject: core,
      appraisal: text(candidate.appraisal).slice(0, 180),
      salience: clamp(candidate.salience ?? 0.55),
      tension: clamp(candidate.tension ?? 0.42),
      status: candidate.status || 'open', updatedAt: now(), mentions: (item.mentions || 0) + 1,
    });
    if (!old) this.state.attention.unshift(item);
    this.state.attention = this.state.attention
      .filter((entry) => entry.status !== 'resolved' || now() - Number(entry.updatedAt || 0) < 3600000)
      .sort((a, b) => (b.salience + b.tension) - (a.salience + a.tension))
      .slice(0, 8);
    return item;
  }

  _makeIntent(input = {}) {
    const mode = input.mode || 'responsive';
    const subject = text(input.subject);
    const action = input.action || actionFor(subject, mode);
    const intent = {
      id: `intent_${crypto.randomUUID()}`,
      mode, action, subject: subject.slice(0, 280),
      source: input.source || 'last_user_turn',
      reason: text(input.reason || '').slice(0, 180),
      allowQuestion: action === 'clarify',
      maxSentences: mode === 'proactive' ? 2 : (hasVulnerability(subject) ? 3 : 4),
      createdAt: now(), status: 'active',
    };
    this.state.activeIntent = intent;
    return intent;
  }

  // Called before every ordinary response.  Other modules may provide signals,
  // but only this method chooses the content nucleus and speech act.
  deliberate(ctx = {}) {
    const userText = text(ctx.userText || ctx.perceived?.cognitiveInput || ctx.perceived?.userContent);
    this._syncBody(ctx.pad, ctx.motivationState);
    const thought = (ctx.openThoughts || []).find((item) => item.status === 'open');
    const subject = userText || text(thought?.content);
    if (!subject) return { shouldSpeak: false, reason: 'no_perceived_subject', intent: null, promptBlock: '' };

    const source = userText ? 'last_user_turn' : 'persistent_thought';
    const attention = this._rememberAttention({
      source, subject, salience: userText ? 0.94 : clamp(thought?.tension || 0.58),
      tension: hasVulnerability(subject) ? 0.76 : clamp(thought?.tension || 0.45),
      appraisal: userText ? '对方刚刚说出的具体事情，是这一轮最重要的对象。' : '这个念头没有消失。',
    });
    const intent = this._makeIntent({ mode: 'responsive', source, subject, reason: attention.appraisal });
    this._save();
    return { shouldSpeak: true, intent, attention, promptBlock: this.toPromptBlock(intent) };
  }

  // Called by both proactive routes.  Silence is a valid outcome; a timer alone
  // is never a reason to fabricate a topic.
  planProactive(ctx = {}) {
    this._syncBody(ctx.pad, ctx.motivationState);
    if (socialBoundary(ctx)) return { shouldSpeak: false, reason: 'social_boundary', nextCheckMs: 60000 };
    const open = (ctx.openThoughts || [])
      .filter((item) => item.status === 'open' && now() >= Number(item.earliestSpeakAt || 0))
      .sort((a, b) => Number(b.tension || 0) - Number(a.tension || 0))[0];
    const fresh = ctx.contextFresh === true ? text(ctx.anchor) : '';
    // In-chat continuation can react to a fresh anchored turn; idle/presence
    // speech requires a persisting thought, never a generic check-in.
    const subject = fresh || text(open?.content);
    if (!subject) return { shouldSpeak: false, reason: 'no_persistent_subject', nextCheckMs: 90000 };
    const source = fresh ? 'fresh_conversation' : 'persistent_thought';
    const tension = fresh ? 0.58 : clamp(open?.tension || 0);
    if (!fresh && tension < 0.5) return { shouldSpeak: false, reason: 'thought_not_mature', nextCheckMs: 90000 };
    const attention = this._rememberAttention({ source, subject, tension, salience: fresh ? 0.76 : tension, appraisal: fresh ? '刚才这句话还值得接住。' : '这是持续存在、尚未说出的念头。' });
    const intent = this._makeIntent({
      mode: 'proactive', source, subject,
      action: fresh ? actionFor(subject, 'responsive') : 'thought',
      reason: attention.appraisal,
    });
    this._save();
    return {
      shouldSpeak: true, thoughtId: open?.id || '', thought: subject, desire: intent.reason,
      tension, action: intent.action, intentId: intent.id, reason: intent.reason,
      attention, intent, promptBlock: this.toPromptBlock(intent),
    };
  }

  toPromptBlock(intent = this.state.activeIntent) {
    if (!intent) return '';
    const action = {
      care: '相手の弱さを勝手に増やさず、事実に触れて受け止める。',
      tease: '共有された具体的な出来事に、少し意地の悪い反応を一つ足す。質問で逃げない。',
      stance: '自分の判断か反論を一つ言う。答えてから質問する型にしない。',
      clarify: '本当に必要な一点だけを聞く。確認のための言い直しはしない。',
      thought: '心に残ったこの具体的な考えを言う。説明や定型の気遣いにすり替えない。',
      respond: '直前の具体的な話題を受け、自分の反応か判断を足す。',
    }[intent.action] || '今の具体的な話題を取り替えない。';
    return [
      '【主体の現在の意図・最優先】これは人格説明ではなく、今この一回に何を言うかという決定。',
      `中心に残っている対象: ${intent.subject}`,
      `話す行為: ${intent.action}。${action}`,
      intent.allowQuestion ? '質問は一つだけ許可される。' : '文末を質問にして会話を維持しようとしない。',
      `長さ: 最大${intent.maxSentences}文。対象にない疲労・心配・予定・共有記憶を作らない。`,
    ].join('\n').slice(0, 920);
  }

  evaluateReply(reply, intent = this.state.activeIntent) {
    const spoken = text(reply);
    if (!intent) return { ok: !!spoken, reason: spoken ? '' : 'empty' };
    if (!spoken) return { ok: false, reason: 'empty' };
    if (!intent.allowQuestion && /[？?]/.test(spoken)) return { ok: false, reason: 'forced_question' };
    if (intent.mode === 'proactive' && !hasVulnerability(intent.subject)
      && /(?:疲れ|心配|大丈夫|累了|担心|没事吧)/.test(spoken)) return { ok: false, reason: 'ungrounded_care' };
    return { ok: true, reason: '' };
  }

  integrateOutcome(input = {}) {
    const intent = input.intent || this.state.activeIntent;
    const accepted = input.accepted !== false && !!text(input.reply);
    if (intent) {
      intent.status = accepted ? 'expressed' : 'held';
      intent.outcomeAt = now();
      const attention = this.state.attention.find((item) => item.subject === intent.subject && item.status === 'open');
      if (attention && accepted) { attention.tension = clamp(attention.tension - 0.22); attention.updatedAt = now(); }
      if (accepted) this.state.body.load = clamp(this.state.body.load * 0.9 + 0.03);
    }
    this.state.experiences.push({ ts: now(), mode: input.mode || intent?.mode || 'responsive', intentId: intent?.id || '', accepted, subject: intent?.subject || '', reply: text(input.reply).slice(0, 200) });
    this.state.experiences = this.state.experiences.slice(-96);
    this._save();
    return this.snapshot();
  }

  registerFeedback(feedback = {}) {
    if (feedback.type === 'ignore') this.state.body.attachment = clamp(this.state.body.attachment - 0.05);
    if (feedback.type === 'reply') this.state.body.attachment = clamp(this.state.body.attachment + 0.025);
    this._save();
  }

  snapshot() { return JSON.parse(JSON.stringify(this.state)); }
}

module.exports = { SubjectCore, defaultState, actionFor, hasVulnerability };
