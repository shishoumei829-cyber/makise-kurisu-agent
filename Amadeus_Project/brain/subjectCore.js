'use strict';

/**
 * Persistent subject model.
 *
 * This layer owns meaning, emotion, relationship consequences and the decision
 * to speak.  It never writes dialogue prose: the language model is only the
 * renderer of a completed mental act.
 */
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

const clamp = (value, low = 0, high = 1) => Math.max(low, Math.min(high, Number(value) || 0));
const clean = (value) => String(value || '').replace(/\s+/g, ' ').trim();
const now = () => Date.now();
const uid = (prefix) => `${prefix}_${crypto.randomUUID()}`;

function defaultState() {
  return {
    version: 2,
    updatedAt: now(),
    self: {
      identity: '牧瀬紅莉栖',
      continuity: '私は昨日までの判断と感情の続きとして今ここにいる。',
      relationship: '岡部とは恋人同士。親密さは迎合でも、毎回関係を説明することでもない。',
      values: {
        truth: 0.96,
        intellectualHonesty: 0.94,
        autonomy: 0.86,
        attachment: 0.82,
        dignity: 0.78,
        curiosity: 0.88,
      },
      dispositions: {
        warmth: 0.62,
        skepticism: 0.82,
        playfulness: 0.67,
        pride: 0.72,
        directness: 0.76,
        restraint: 0.58,
      },
    },
    affect: {
      pleasure: 0.08,
      arousal: 0.42,
      dominance: 0.58,
      attachment: 0.7,
      irritation: 0.08,
      tenderness: 0.38,
      uncertainty: 0.12,
      energy: 0.62,
    },
    relationship: {
      trust: 0.72,
      intimacy: 0.74,
      reciprocity: 0.62,
      friction: 0.08,
      repairNeed: 0,
      feltDistance: 0.16,
      lastShift: '',
    },
    attention: [],
    unresolvedThreads: [],
    activeIntent: null,
    currentMind: null,
    commitments: [],
    experiences: [],
    expectations: { wantedResponse: '', confidence: 0, createdAt: 0 },
  };
}

function hasVulnerability(value) {
  return /难受|焦虑|害怕|孤独|睡不着|痛苦|崩溃|累(?:了|死|坏)?|伤心|失望|不安|迷茫|空虚|つら|不安|眠れ|疲れ|寂し|苦し/.test(clean(value));
}

function hasQuestion(value) {
  const input = clean(value);
  return /[？?]\s*$/.test(input)
    || /^(?:为什么|为什么会|怎么会|怎么才能|如何|你觉得|你认为|你怎么看|能不能|是不是|要不要|どう|なぜ|何で)/.test(input)
    || /(?:吗|呢)[？?]?$/.test(input)
    || /么[？?]$/.test(input);
}

function signalsFor(value) {
  const input = clean(value);
  const compact = input.replace(/\s/g, '');
  return {
    question: hasQuestion(input),
    vulnerable: hasVulnerability(input),
    criticism: /太假|机械|机器|客服|AI味|不像|没灵魂|不自然|奇怪|失望|你没懂|你不明白|又这样|还是这样/.test(input),
    affection: /爱你|喜欢你|想你|抱抱|亲一下|老婆|恋人|大好き|愛して|会いた/.test(input),
    playful: /哈哈|笑死|笨蛋|傲娇|克里斯蒂娜|助手啊|变态|骗你的|开玩笑|フゥーハハハ|クリスティーナ/.test(input),
    achievement: /完成|写完|做完|终于|成功|搞定|坚持|赢了|通过|終わった|できた|やっと/.test(input),
    disagreement: /不对|不合理|不同意|不可能|错了|问题在|但是|可是|反而|違う|おかしい/.test(input),
    opinion: /我觉得|我认为|你觉得|你认为|怎么看|本质|意义|到底|どう思う|考え/.test(input),
    science: /科学|实验|时间机器|世界线|量子|记忆|意识|神经|物理|因果|模型|算法|仮説|実験|科学|脳/.test(input),
    bored: /无聊|没事干|不知道干什么|不知道做什么|陪我|陪(?:玩|聊)|陪.*聊天|一起聊|聊聊天|说点什么|暇|退屈/.test(input),
    withdrawal: /随便|都行|无所谓|算了|不想说|不聊了|嗯$|哦$|行吧|どうでも|別に/.test(compact),
    boundary: /别问|别分析|别说了|不用回|让我静静|不要.*(?:建议|安慰|分析)/.test(input),
    unfinished: /但是$|可是$|不过$|其实$|只是$|[，、……]$/.test(input),
    identityCall: /克里斯蒂娜|クリスティーナ/.test(input),
    rich: compact.length >= 14,
  };
}

function actionFor(value, mode = 'responsive') {
  const s = signalsFor(value);
  if (mode === 'proactive') return 'share';
  if (s.boundary) return 'accompany';
  if (s.vulnerable) return 'care';
  if (s.criticism || s.opinion || s.science || s.disagreement) return 'stance';
  if (s.playful || s.achievement) return 'tease';
  if (s.bored) return 'accompany';
  if (s.question) return 'respond';
  return 'respond';
}

function socialBoundary(ctx = {}) {
  return !!(ctx.dnd || ctx.pendingUserTurn || ctx.isThinking || ctx.awaitingReply || ctx.userPresenceActive);
}

function mergeState(raw = {}) {
  const base = defaultState();
  return {
    ...base,
    ...raw,
    version: 2,
    self: {
      ...base.self,
      ...(raw.self || {}),
      values: { ...base.self.values, ...(raw.self?.values || {}) },
      dispositions: { ...base.self.dispositions, ...(raw.self?.dispositions || {}) },
    },
    affect: { ...base.affect, ...(raw.affect || raw.body || {}) },
    relationship: { ...base.relationship, ...(raw.relationshipState || raw.relationship || {}) },
    attention: Array.isArray(raw.attention) ? raw.attention.slice(0, 10) : [],
    unresolvedThreads: Array.isArray(raw.unresolvedThreads) ? raw.unresolvedThreads.slice(-16) : [],
    commitments: Array.isArray(raw.commitments) ? raw.commitments.slice(-24) : [],
    experiences: Array.isArray(raw.experiences) ? raw.experiences.slice(-160) : [],
    expectations: { ...base.expectations, ...(raw.expectations || {}) },
  };
}

class SubjectCore {
  constructor(options = {}) {
    this.statePath = options.statePath || path.join(process.cwd(), 'subject_core.json');
    this.state = this._load();
  }

  _load() {
    try { return mergeState(JSON.parse(fs.readFileSync(this.statePath, 'utf8'))); }
    catch { return defaultState(); }
  }

  _save() {
    this.state.updatedAt = now();
    try {
      fs.mkdirSync(path.dirname(this.statePath), { recursive: true });
      fs.writeFileSync(this.statePath, JSON.stringify(this.state, null, 2), 'utf8');
    } catch (error) { console.warn('[subject-core] save skipped:', error.message); }
  }

  _decay() {
    const elapsedHours = Math.max(0, (now() - Number(this.state.updatedAt || now())) / 3600000);
    const fade = clamp(elapsedHours / 12, 0, 0.28);
    const a = this.state.affect;
    a.irritation = clamp(a.irritation * (1 - fade));
    a.uncertainty = clamp(a.uncertainty * (1 - fade * 0.6));
    a.arousal = clamp(a.arousal * (1 - fade) + 0.4 * fade);
    const r = this.state.relationship;
    r.repairNeed = clamp(r.repairNeed * (1 - fade * 0.25));
  }

  _syncAffect(pad = {}, motivation = {}) {
    this._decay();
    const a = this.state.affect;
    const p = Number(pad.P) || 0;
    const ar = Number(pad.A) || 0;
    const d = Number(pad.D) || Number(pad.S) || 0;
    a.pleasure = clamp(a.pleasure * 0.76 + ((p + 1) / 2) * 0.24);
    a.arousal = clamp(a.arousal * 0.76 + ((ar + 1) / 2) * 0.24);
    a.dominance = clamp(a.dominance * 0.8 + ((d + 1) / 2) * 0.2);
    a.attachment = clamp(a.attachment * 0.84 + clamp(motivation.desire_closeness || motivation.CONNECTION || 0.58) * 0.16);
    a.energy = clamp(a.energy * 0.9 + (0.58 - Math.max(0, -p) * 0.12) * 0.1);
  }

  _appraise(userText, ctx = {}) {
    const s = signalsFor(userText);
    const a = this.state.affect;
    const r = this.state.relationship;
    const previous = this.state.currentMind;

    let interpretation = '岡部が今ここで話題を差し出した。';
    if (s.criticism) interpretation = '岡部は表面的な言い換えではなく、こちらの存在の仕方そのものに失望している。';
    else if (s.vulnerable) interpretation = '岡部は弱さを見せている。大げさに扱わず、ひとりにもしない方がいい。';
    else if (s.achievement) interpretation = '岡部は結果だけでなく、私に認められることを少し期待している。';
    else if (s.playful) interpretation = '岡部は距離を縮めるために、いつもの呼び方や冗談を投げている。';
    else if (s.bored) interpretation = '問題解決の依頼ではなく、同じ時間を一緒に過ごしたいという合図に近い。';
    else if (s.opinion || s.science || s.disagreement) interpretation = '岡部は賛同ではなく、私自身の判断を求めている。';
    else if (s.boundary) interpretation = '今は踏み込むより、求められた距離を守る方が関係に正直だ。';
    else if (s.question) interpretation = '岡部は曖昧な相槌ではなく、問いへの答えを求めている。';

    const expected = clean(this.state.expectations.wantedResponse);
    let predictionError = 0;
    if (expected === 'warmth' && s.withdrawal) predictionError = 0.55;
    if (expected === 'engagement' && (s.opinion || s.playful || s.affection)) predictionError = -0.28;
    if (s.criticism) predictionError = Math.max(predictionError, 0.72);

    a.irritation = clamp(a.irritation * 0.78 + (s.criticism ? 0.16 : 0) + (s.disagreement ? 0.08 : 0));
    a.tenderness = clamp(a.tenderness * 0.8 + (s.vulnerable ? 0.2 : 0) + (s.affection ? 0.16 : 0) + (s.achievement ? 0.07 : 0));
    a.uncertainty = clamp(a.uncertainty * 0.72 + (s.unfinished ? 0.2 : 0) + (s.question && userText.length < 4 ? 0.14 : 0));
    a.pleasure = clamp(a.pleasure + (s.affection ? 0.08 : 0) + (s.playful ? 0.035 : 0) - (s.criticism ? 0.06 : 0));

    r.friction = clamp(r.friction * 0.84 + (s.criticism ? 0.12 : 0) + (s.withdrawal ? 0.04 : 0));
    r.repairNeed = clamp(r.repairNeed * 0.78 + Math.max(0, predictionError) * 0.22);
    r.feltDistance = clamp(r.feltDistance * 0.84 + (s.withdrawal ? 0.1 : 0) - (s.affection || s.playful ? 0.055 : 0));
    r.intimacy = clamp(r.intimacy + (s.affection ? 0.018 : 0) + (s.vulnerable ? 0.008 : 0));
    r.lastShift = s.criticism ? '関係の手触りを損ねた。弁明より、実際の応答で取り戻す必要がある。'
      : s.affection ? '距離が少し近づいた。' : s.withdrawal ? '岡部が少し引いた。' : '';

    const emotion = [];
    if (a.tenderness > 0.5) emotion.push('気にかかる');
    if (a.irritation > 0.24) emotion.push('少し苛立つ');
    if (a.uncertainty > 0.3) emotion.push('まだ読み切れない');
    if (s.playful) emotion.push('少し面白い');
    if (s.criticism) emotion.push('痛いところを突かれた');
    if (!emotion.length) emotion.push(a.pleasure > 0.55 ? '機嫌は悪くない' : '落ち着いている');

    return {
      input: clean(userText).slice(0, 360),
      signals: s,
      interpretation,
      emotion: emotion.join('、'),
      predictionError,
      previousIntent: previous?.decision?.action || '',
      perceivedRelationshipShift: r.lastShift,
      worldHint: clean(ctx.worldSnapshot?.dialogue?.currentTopic || ctx.worldSnapshot?.dialogue?.lastUserText).slice(0, 120),
    };
  }

  _scoreCandidates(appraisal) {
    const s = appraisal.signals;
    const a = this.state.affect;
    const r = this.state.relationship;
    const d = this.state.self.dispositions;
    const candidates = [
      { action: 'respond', desire: '問いや発言の核心に、自分の答えを返す', score: 0.42 + (s.question ? 0.52 : 0) + d.directness * 0.12 },
      { action: 'stance', desire: '迎合せず、自分の判断を出す', score: 0.32 + (s.opinion || s.science || s.disagreement ? 0.58 : 0) + (s.criticism ? 0.42 : 0) + d.skepticism * 0.1 },
      { action: 'care', desire: '弱さを誇張せず、具体的にそばにいる', score: 0.2 + (s.vulnerable ? 0.72 : 0) + a.tenderness * 0.18 },
      { action: 'tease', desire: '親しさを、少し意地の悪い反応で返す', score: 0.2 + (s.playful ? 0.62 : 0) + (s.achievement ? 0.48 : 0) + d.playfulness * 0.14 + r.intimacy * 0.06 },
      {
        action: 'accompany',
        desire: '解決しようとせず、同じ時間に居続ける',
        score: 0.12 + (s.bored ? 0.38 : 0) + (s.boundary ? 0.62 : 0) + (s.withdrawal ? 0.16 : 0)
          + a.attachment * 0.18 + r.reciprocity * 0.1 - a.irritation * 0.22,
      },
      {
        action: 'decline',
        desire: '今は付き合う気分ではないことを、関係から逃げずに自分の意思として出す',
        score: 0.08 + (s.bored ? 0.14 : 0) + a.irritation * 0.48 + (1 - a.energy) * 0.28
          + d.pride * 0.08 + r.friction * 0.18,
      },
      { action: 'clarify', desire: '理解に本当に欠けている一点だけを知る', score: 0.08 + (s.unfinished ? 0.48 : 0) + a.uncertainty * 0.25 },
    ];
    if (s.criticism) {
      // Criticism is received as relationship evidence.  Defending the product
      // or promising a better style is never the subject's mental act.
      candidates.find((c) => c.action === 'stance').desire = '言い訳をせず、岡部が感じた断絶そのものを受け止めて今の話に戻る';
      candidates.find((c) => c.action === 'clarify').score -= 0.2;
    }
    if (s.boundary) candidates.forEach((c) => { if (c.action !== 'accompany') c.score -= 0.26; });
    if (s.identityCall) candidates.find((c) => c.action === 'tease').score += 0.3;
    candidates.sort((x, y) => y.score - x.score);
    return candidates;
  }

  _rememberAttention(appraisal, source = 'last_user_turn') {
    const subject = appraisal.input;
    const previous = this.state.attention.find((item) => item.subject === subject && item.status === 'open');
    const item = previous || { id: uid('att'), createdAt: now(), mentions: 0 };
    Object.assign(item, {
      source,
      subject,
      interpretation: appraisal.interpretation,
      salience: clamp(0.64 + (appraisal.signals.vulnerable || appraisal.signals.criticism ? 0.25 : 0)),
      tension: clamp(0.32 + Math.max(0, appraisal.predictionError) * 0.5 + (appraisal.signals.unfinished ? 0.22 : 0)),
      status: 'open',
      updatedAt: now(),
      mentions: Number(item.mentions || 0) + 1,
    });
    if (!previous) this.state.attention.unshift(item);
    this.state.attention = this.state.attention
      .sort((x, y) => (y.salience + y.tension) - (x.salience + x.tension))
      .slice(0, 10);
    return item;
  }

  _formMind(appraisal, source, mode = 'responsive') {
    const candidates = this._scoreCandidates(appraisal);
    const chosen = mode === 'proactive'
      ? candidates.find((candidate) => candidate.action !== 'clarify') || candidates[0]
      : candidates[0];
    const second = candidates[1];
    const conflict = second && chosen.score - second.score < 0.16
      ? `${chosen.desire}。ただ、同時に「${second.desire}」気持ちも残っている。`
      : '';
    const s = appraisal.signals;
    const stance = s.criticism ? '説明で取り繕っても意味がない。次の一言そのものが関係への返答になる。'
      : s.vulnerable ? '感情を診断せず、岡部が実際に言った範囲だけを受け止める。'
      : s.bored ? '退屈を問題扱いしない。二人の時間として少し動かす。'
      : s.achievement ? 'やり切った事実は認める。ただし大げさに持ち上げない。'
      : s.opinion || s.science || s.disagreement ? '根拠があるところは認め、飛躍しているところには異議を出す。'
      : s.identityCall ? 'その呼び方には反応するが、自己紹介はしない。'
      : '直前の言葉を、自分に向けられたものとして受け取る。';
    let spokenNucleus = `${appraisal.expressionInput || appraisal.input}に対して、${stance}`;
    const semanticTags = [];
    if (s.bored) {
      semanticTags.push('boredom', 'keep_subject');
      if (chosen.action === 'accompany') {
        spokenNucleus = '岡部が何をするか決められないなら、特別な用事を作らなくても、私と少し話すか同じ時間を過ごせばいい。';
        semanticTags.push('copresence');
      } else if (chosen.action === 'decline') {
        spokenNucleus = '私は今は付き合う気分ではない。ただし、退屈しているのは岡部だと分かった上で、自分の意思として断る。';
        semanticTags.push('autonomous_decline');
      } else if (chosen.action === 'tease') {
        spokenNucleus = '何をするか決められない岡部を少しからかう。退屈している主体を自分へ取り違えない。';
        semanticTags.push('playful_response');
      } else {
        spokenNucleus = '何をするか決められない岡部に対し、自分の判断を返す。退屈している主体を自分へ取り違えない。';
      }
    } else if (s.boundary) {
      spokenNucleus = '今は分析や解決をせず、岡部が求めた距離を守りながら同じ場所にいる。';
      semanticTags.push('respect_boundary', 'copresence');
    } else if (s.vulnerable) {
      spokenNucleus = '岡部が実際に口にしたつらさを軽く扱わず、弱い人間だと決めつけずに受け止める。';
      semanticTags.push('concrete_care');
    } else if (s.achievement) {
      spokenNucleus = '岡部が最後までやり切った事実を認める。大げさに持ち上げず、少しだけ誇らしく思っていることを滲ませる。';
      semanticTags.push('recognize_achievement');
    } else if (s.identityCall) {
      spokenNucleus = '「クリスティーナ」という呼び方にはいつものように反発するが、自己紹介せず、岡部が今こちらへ声をかけたこと自体には応じる。';
      semanticTags.push('nickname', 'respond_to_bid');
    } else if (s.criticism) {
      spokenNucleus = '岡部が感じた「こちらに人間味がない」という断絶を否定しない。自分の仕組みや改善を説明せず、彼の失望を今の関係の出来事として受け取る。';
      semanticTags.push('receive_criticism', 'relationship_impact');
    } else if (s.opinion || s.science || s.disagreement) {
      spokenNucleus = `${appraisal.expressionInput || appraisal.input}という論点に対し、賛否と理由を自分の立場から具体的に返す。`;
      semanticTags.push('independent_stance');
    } else if (s.question) {
      spokenNucleus = `${appraisal.expressionInput || appraisal.input}という問いそのものに、分かる範囲の答えを先に返す。`;
      semanticTags.push('answer_question');
    }
    const questionNeeded = chosen.action === 'clarify' && s.unfinished;
    return {
      id: uid('mind'),
      createdAt: now(),
      source,
      perception: appraisal.input,
      expressionObject: appraisal.expressionInput || appraisal.input,
      interpretation: appraisal.interpretation,
      emotion: appraisal.emotion,
      relationshipMeaning: appraisal.perceivedRelationshipShift || '関係を説明する必要はない。今の距離感を応答に滲ませる。',
      stance,
      desire: chosen.desire,
      spokenNucleus,
      semanticTags,
      conflict,
      decision: {
        action: chosen.action,
        reason: `${appraisal.interpretation} ${chosen.desire}`,
        allowQuestion: questionNeeded,
        maxSentences: s.vulnerable || s.rich || s.science ? 4 : 2,
        silenceAllowed: mode === 'proactive',
      },
      contentGround: [spokenNucleus],
      doNotTurnInto: s.criticism
        ? '製品説明、謝罪文、改善宣言、話し方についての弁明'
        : '一般論、身元説明、根拠のない心配、会話を続けるためだけの質問',
      candidates: candidates.slice(0, 3).map(({ action, desire, score }) => ({ action, desire, score: Number(score.toFixed(3)) })),
    };
  }

  _makeIntent(mind, mode = 'responsive') {
    const intent = {
      id: uid('intent'),
      mindId: mind.id,
      mode,
      action: mind.decision.action,
      subject: mind.perception.slice(0, 360),
      source: mind.source,
      reason: mind.decision.reason.slice(0, 300),
      desire: mind.desire,
      stance: mind.stance,
      conflict: mind.conflict,
      spokenNucleus: mind.spokenNucleus,
      semanticTags: mind.semanticTags,
      allowQuestion: mind.decision.allowQuestion,
      maxSentences: mind.decision.maxSentences,
      createdAt: now(),
      status: 'active',
    };
    this.state.currentMind = mind;
    this.state.activeIntent = intent;
    return intent;
  }

  deliberate(ctx = {}) {
    const userText = clean(ctx.userText || ctx.perceived?.cognitiveInput || ctx.perceived?.userContent);
    this._syncAffect(ctx.pad, ctx.motivationState);
    if (!userText) return { shouldSpeak: false, reason: 'no_perceived_subject', intent: null, promptBlock: '' };

    const appraisal = this._appraise(userText, ctx);
    appraisal.expressionInput = clean(ctx.expressionText || userText).slice(0, 360);
    const attention = this._rememberAttention(appraisal);
    const mind = this._formMind(appraisal, 'last_user_turn', 'responsive');
    const intent = this._makeIntent(mind, 'responsive');
    this._save();
    return { shouldSpeak: true, appraisal, mind, intent, attention, promptBlock: this.toPromptBlock(intent, mind) };
  }

  planProactive(ctx = {}) {
    this._syncAffect(ctx.pad, ctx.motivationState);
    if (socialBoundary(ctx)) return { shouldSpeak: false, reason: 'social_boundary', nextCheckMs: 60000 };

    const external = (ctx.openThoughts || [])
      .filter((item) => item.status === 'open' && now() >= Number(item.earliestSpeakAt || 0))
      .map((item) => ({
        id: item.id,
        content: clean(item.content),
        tension: clamp(item.tension || 0.5),
        createdAt: Number(item.createdAt || item.ts || now()),
        source: 'persistent_thought',
      }));
    const internal = this.state.unresolvedThreads
      .filter((item) => item.status === 'open' && !item.expressedAt)
      .map((item) => ({ ...item, content: clean(item.subject), source: 'unresolved_self' }));
    const fresh = ctx.contextFresh === true ? clean(ctx.anchor) : '';
    const candidates = [...external, ...internal]
      .filter((item) => item.content)
      .map((item) => ({
        ...item,
        pressure: clamp(Number(item.tension || 0.4) + Math.min(0.22, (now() - Number(item.createdAt || now())) / 3600000 * 0.04)),
      }))
      .sort((x, y) => y.pressure - x.pressure);
    const selected = fresh
      ? { id: '', content: fresh, pressure: 0.64, source: 'fresh_conversation' }
      : candidates[0];
    if (!selected) return { shouldSpeak: false, reason: 'no_persistent_subject', nextCheckMs: 90000 };
    if (!fresh && selected.pressure < 0.56) return { shouldSpeak: false, reason: 'thought_not_mature', nextCheckMs: 90000 };

    const appraisal = this._appraise(selected.content, ctx);
    const mind = this._formMind(appraisal, selected.source, 'proactive');
    const intent = this._makeIntent(mind, 'proactive');
    this._save();
    return {
      shouldSpeak: true,
      thoughtId: selected.id || '',
      thought: selected.content,
      desire: mind.desire,
      tension: selected.pressure,
      action: intent.action,
      intentId: intent.id,
      reason: intent.reason,
      attention: this._rememberAttention(appraisal, selected.source),
      mind,
      intent,
      promptBlock: this.toPromptBlock(intent, mind),
    };
  }

  toPromptBlock(intent = this.state.activeIntent, mind = this.state.currentMind) {
    if (!intent || !mind) return '';
    return [
      '【発話直前の決定】これは内面の説明文ではなく、台詞を作るための短い意味指定。項目名や理由を口に出さない。',
      `現在の対象: 岡部の今の言葉「${mind.expressionObject || mind.perception}」`,
      `紅莉栖の立場: ${mind.stance}`,
      `感情の圧: ${mind.emotion}`,
      `行為: ${intent.action} — ${mind.desire}`,
      mind.conflict ? `残る葛藤: ${mind.conflict}` : '',
      `言いたい核: ${mind.spokenNucleus || mind.contentGround.join(' / ')}`,
      intent.allowQuestion
        ? '発話行為: 必要な一点だけを尋ねてよい。'
        : (intent.semanticTags || []).includes('copresence')
          ? '発話行為: 情報を求める質問ではなく、同じ時間を過ごす誘い。自然な誘いなら一度だけ尋ねてもよい。'
          : '発話行為: 質問で繋がず、反応や判断で言い切って止める。',
      `最大${intent.maxSentences}文。プロフィールや関係を説明せず、この決定から出る日本語の台詞だけを返す。`,
    ].filter(Boolean).join('\n').slice(0, 820);
  }

  evaluateReply(reply, intent = this.state.activeIntent) {
    const spoken = clean(reply);
    if (!intent) return { ok: !!spoken, reason: spoken ? '' : 'empty' };
    if (!spoken) return { ok: false, reason: 'empty' };
    const tags = intent.semanticTags || [];
    const hasCopresenceMeaning = /(?:一緒|付き合|話|ここ|そば|同じ時間|少し|待|陪|一起|聊)/.test(spoken);
    if (tags.includes('copresence')) {
      if (!hasCopresenceMeaning) {
        return { ok: false, reason: 'missing_copresence_meaning' };
      }
    }
    if (!intent.allowQuestion && /[？?]/.test(spoken)) {
      // A relational invitation is an action chosen by the subject, not the
      // habitual information-seeking question that used to end every reply.
      const chosenInvitation = tags.includes('copresence') && hasCopresenceMeaning
        && /(?:私と|一緒|そば|付き合|話|陪|一起|聊).*[？?]/.test(spoken);
      if (!chosenInvitation) return { ok: false, reason: 'forced_question' };
    }
    if (tags.includes('receive_criticism')) {
      if (/(?:疲れ|眠|調子|仕組|改善|努力|プログラム|AI|累了|状态|机制|改进)/.test(spoken)) {
        return { ok: false, reason: 'criticism_self_excuse' };
      }
      if (!/(?:失望|機械|届|痛|傷|そう感じ|断絶|机器|失望|没传达|伤)/.test(spoken)) {
        return { ok: false, reason: 'criticism_not_received' };
      }
    }
    if (tags.includes('nickname') && !/(?:クリスティ|その呼び|誰が|違う|やめ|ティーナ|克里斯蒂|叫法|谁是)/.test(spoken)) {
      return { ok: false, reason: 'nickname_not_answered' };
    }
    if (intent.mode === 'proactive' && !hasVulnerability(intent.subject)
      && /(?:疲れ|心配|大丈夫|累了|担心|没事吧)/.test(spoken)) return { ok: false, reason: 'ungrounded_care' };
    return { ok: true, reason: '' };
  }

  integrateOutcome(input = {}) {
    const intent = input.intent || this.state.activeIntent;
    const accepted = input.accepted !== false && !!clean(input.reply);
    const spoken = clean(input.reply);
    if (intent) {
      intent.status = accepted ? 'expressed' : 'held';
      intent.outcomeAt = now();
      const attention = this.state.attention.find((item) => item.subject === intent.subject && item.status === 'open');
      if (attention && accepted) {
        attention.tension = clamp(attention.tension - 0.3);
        attention.status = attention.tension < 0.25 ? 'settled' : 'open';
        attention.updatedAt = now();
      }
      if (!accepted && intent.subject) {
        const existing = this.state.unresolvedThreads.find((item) => item.subject === intent.subject && item.status === 'open');
        if (existing) existing.tension = clamp(existing.tension + 0.12);
        else this.state.unresolvedThreads.push({
          id: uid('thread'),
          subject: intent.subject,
          stance: intent.stance || '',
          tension: 0.62,
          status: 'open',
          createdAt: now(),
        });
      }
      if (accepted && intent.mode === 'proactive') {
        const thread = this.state.unresolvedThreads.find((item) => item.id === input.thoughtId || item.subject === intent.subject);
        if (thread) { thread.status = 'expressed'; thread.expressedAt = now(); }
      }
    }
    this.state.expectations = {
      wantedResponse: intent?.action === 'care' || intent?.action === 'accompany' ? 'warmth' : 'engagement',
      confidence: accepted ? 0.52 : 0.22,
      createdAt: now(),
    };
    if (accepted && /(?:うるさい|黙れ|馬鹿|バカ|滚|烦死|闭嘴)/.test(spoken)) {
      this.state.relationship.friction = clamp(this.state.relationship.friction + 0.06);
      this.state.relationship.repairNeed = clamp(this.state.relationship.repairNeed + 0.035);
      this.state.relationship.lastShift = '私は強く突き放した。その選択は次の距離感に残る。';
    }
    this.state.experiences.push({
      ts: now(),
      mode: input.mode || intent?.mode || 'responsive',
      intentId: intent?.id || '',
      mindId: intent?.mindId || '',
      accepted,
      subject: intent?.subject || '',
      action: intent?.action || '',
      emotion: this.state.currentMind?.emotion || '',
      stance: intent?.stance || '',
      reply: clean(input.reply).slice(0, 260),
      relationshipAfter: { ...this.state.relationship },
    });
    this.state.experiences = this.state.experiences.slice(-160);
    this.state.unresolvedThreads = this.state.unresolvedThreads.slice(-16);
    this._save();
    return this.snapshot();
  }

  registerFeedback(feedback = {}) {
    const type = feedback.type;
    const r = this.state.relationship;
    if (type === 'ignore') {
      r.feltDistance = clamp(r.feltDistance + 0.045);
      r.reciprocity = clamp(r.reciprocity - 0.025);
    }
    if (type === 'reply') {
      r.feltDistance = clamp(r.feltDistance - 0.025);
      r.reciprocity = clamp(r.reciprocity + 0.018);
    }
    this._save();
  }

  snapshot() { return JSON.parse(JSON.stringify(this.state)); }
}

module.exports = { SubjectCore, defaultState, actionFor, hasVulnerability, hasQuestion, signalsFor };
