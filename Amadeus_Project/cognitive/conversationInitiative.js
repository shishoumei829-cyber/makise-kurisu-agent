'use strict';

/**
 * 主动对话设计规范（人对人，不是定时器）：
 *
 * 1. 在场优先：打开窗口 / 看见你，她应「察觉到你」，并有概率先开口。
 *    两个人相见，总有一方先说话；不必等你先打字。
 * 2. 存在感：动机可以是搞笑、骚扰、开玩笑、发出一点动静——社交动作本身可以是目的。
 * 3. 人格不让位：动机可以轻，措辞必须是牧濑红莉栖（聪明、嘴硬、熟人拌嘴）。
 *    action 只描述「为什么想开口」，不规定台词模板，不把她锁成通用戳一戳。
 * 4. 冷感回拉：察觉冷漠、敷衍、沉默变僵时，换轻话题或闹一下，而不是继续查岗。
 * 5. 对话内跟话：仍有动机才叠话；任务句/告别/她刚反问时不抢。
 * 6. 勿扰仍尊重：明确忙碌/睡觉时闭嘴。
 */

const fs = require('fs');
const path = require('path');
const { textFragments } = require('../lib/memoryAdmission');

const ACTIONS = Object.freeze({
  HOLD: 'hold',
  CARE: 'care',
  PROBE: 'probe',
  STANCE: 'stance',
  TEASE: 'tease',
  SHARE: 'share',
  POKE: 'poke',
});

function clamp(value, min = 0, max = 1) {
  return Math.max(min, Math.min(max, value));
}

function compact(value) {
  return String(value || '').replace(/\s+/g, ' ').trim();
}

function fingerprint(userText, replyText) {
  const source = `${compact(userText)}\n${compact(replyText)}`;
  let hash = 2166136261;
  for (let i = 0; i < source.length; i += 1) {
    hash ^= source.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0).toString(36);
}

function stableEntropy(seed) {
  const value = parseInt(String(seed || '0').slice(-7), 36);
  return Number.isFinite(value) ? (value % 1000) / 1000 : 0.5;
}

function analyzeTurn(userText, replyText) {
  const user = compact(userText);
  const reply = compact(replyText);
  const shortUser = user.replace(/\s/g, '').length > 0 && user.replace(/\s/g, '').length <= 4;
  return {
    user,
    reply,
    assistantAsked: /[?？]\s*$/.test(reply),
    ending: /晚安|睡了|先忙|拜拜|再见|不用回|结束|改天聊|我走了/.test(user),
    task: /^(请|帮我|给我|打开|关闭|启动|停止|删除|保存|运行|执行|修复|检查|测试|生成|写一个|查一下|搜索|计算|翻译)/.test(user),
    vulnerable: /没意思|难过|烦|焦虑|害怕|孤独|累了|撑不住|想哭|心情不好|睡不着|失望|迷茫|空虚/.test(user),
    opinion: /我觉得|我认为|我不同意|为什么|本质|意义|如果|假如|到底|有趣|不合理|不可能|更像/.test(user),
    personal: /我最近|我今天|我以前|我小时候|我刚才|其实我|有件事|我发现|我总是|我一直/.test(user),
    playful: /哈哈|笑死|笨蛋|傲娇|骗你的|开玩笑|红莉栖|克里斯蒂娜|助手啊|助手呢/.test(user),
    bored: /无聊|没事干|好闲|闲得|陪我|说点什么|在干嘛/.test(user),
    unfinished: /[，、……]$|但是|不过|只是|其实|总觉得|说不上来/.test(user),
    science: /实验|科学|时间机器|世界线|量子|记忆|意识|神经|物理|因果|模型|算法/.test(user),
    rich: user.replace(/\s/g, '').length >= 12,
    // 冷感：短、敷衍、不想接
    cold: shortUser
      || /^(嗯+|哦+|喔+|啊+|额+|行|好|随便|没事|在|哦|嗯|知道了|收到|ok|OK|嗯嗯|呵呵)[。！？.!]?$/.test(user)
      || /不想聊|随便你|都行|无所谓|你说呢/.test(user),
  };
}

class ConversationInitiativeEngine {
  constructor(options = {}) {
    this.statePath = options.statePath || '';
    this.cooldownMs = Math.max(8000, Number(options.cooldownMs) || 28000);
    this.state = {
      lastSpokenAt: 0,
      lastAction: '',
      lastTurnId: '',
      recentActions: [],
      activeThread: null,
      recentTopics: [],
      ignoredStreak: 0,
      engagedStreak: 0,
      sentCount: 0,
      lastPresenceNoticeAt: 0,
      sessionStartedAt: Date.now(),
    };
    this._load();
  }

  _load() {
    if (!this.statePath) return;
    try {
      if (fs.existsSync(this.statePath)) {
        this.state = { ...this.state, ...JSON.parse(fs.readFileSync(this.statePath, 'utf8')) };
        this.state.ignoredStreak = Math.min(3, Math.max(0, Number(this.state.ignoredStreak) || 0));
      }
    } catch { /* use fresh state */ }
    // 每次进程启动视为新见面窗口
    this.state.sessionStartedAt = Date.now();
  }

  _save() {
    if (!this.statePath) return;
    try {
      fs.mkdirSync(path.dirname(this.statePath), { recursive: true });
      fs.writeFileSync(this.statePath, JSON.stringify(this.state, null, 2));
    } catch { /* initiative must never block chat */ }
  }

  markSessionStart(now = Date.now()) {
    this.state.sessionStartedAt = now;
    this._save();
    return this.snapshot();
  }

  decide(input = {}) {
    const now = Number(input.now) || Date.now();
    const phase = input.phase === 'silence' ? 'silence' : 'floor_release';
    const features = analyzeTurn(input.userText, input.replyText);
    const turnId = fingerprint(features.user, features.reply);
    const relScore = clamp(Number(input.relScore) || 0);
    const arousal = clamp((Number(input.pad?.A) + 1) / 2);
    const entropy = Number.isFinite(input.entropy) ? clamp(input.entropy) : stableEntropy(`${turnId}${phase}`);
    const askedHold = features.assistantAsked && phase === 'floor_release';
    const hardHold = features.ending || features.task || askedHold
      || !features.user || !features.reply || this.state.lastTurnId === turnId
      || now - this.state.lastSpokenAt < this.cooldownMs;
    if (hardHold) {
      const reevaluateAfterMs = askedHold && (features.vulnerable || features.personal || features.unfinished || features.bored || features.cold)
        ? 2400 + Math.floor(stableEntropy(turnId) * 3000)
        : 0;
      return this._hold(
        turnId,
        features.assistantAsked ? 'assistant_already_passed_floor' : 'conversation_boundary',
        reevaluateAfterMs,
      );
    }

    const candidates = [];
    if (features.vulnerable) {
      candidates.push({ action: ACTIONS.CARE, score: 0.84 + (features.personal ? 0.1 : 0), probability: 0.84 });
    }
    if (features.cold && phase === 'silence') {
      candidates.push({ action: ACTIONS.POKE, score: 0.78, probability: 0.72 });
      candidates.push({ action: ACTIONS.TEASE, score: 0.7 + relScore * 0.1, probability: 0.58 });
    }
    if (features.personal || features.unfinished) {
      candidates.push({ action: ACTIONS.PROBE, score: 0.66 + (features.unfinished ? 0.16 : 0), probability: 0.64 });
    }
    if (features.opinion || features.science) {
      candidates.push({ action: ACTIONS.STANCE, score: 0.6 + (features.science ? 0.12 : 0) + arousal * 0.1, probability: 0.55 });
    }
    if (features.playful && relScore >= 0.15) {
      candidates.push({ action: ACTIONS.TEASE, score: 0.64 + relScore * 0.18, probability: 0.6 });
    }
    if ((features.bored || phase === 'silence') && !features.vulnerable && !features.task) {
      candidates.push({ action: ACTIONS.POKE, score: 0.58 + (features.bored ? 0.2 : 0.08), probability: features.bored ? 0.74 : 0.48 });
    }
    if (features.rich && !features.vulnerable && !features.task) {
      candidates.push({ action: ACTIONS.SHARE, score: 0.52 + arousal * 0.1, probability: 0.42 });
    }
    if (features.assistantAsked && phase === 'silence') {
      for (let i = candidates.length - 1; i >= 0; i -= 1) {
        if (candidates[i].action === ACTIONS.SHARE) candidates.splice(i, 1);
      }
    }
    for (const candidate of candidates) {
      if (candidate.action === this.state.lastAction) candidate.score -= 0.18;
      if (this.state.recentActions.slice(-3).includes(candidate.action)) candidate.score -= 0.05;
    }
    candidates.sort((a, b) => b.score - a.score);
    const selected = candidates[0];
    const holdScore = phase === 'floor_release' ? 0.5 : 0.4;
    if (!selected || selected.score <= holdScore || entropy > selected.probability + (phase === 'silence' ? 0.16 : 0)) {
      const salient = Boolean(features.vulnerable || features.personal || features.unfinished || features.opinion || features.playful || features.bored || features.cold);
      const reevaluateAfterMs = phase === 'floor_release' && salient
        ? 2000 + Math.floor(stableEntropy(turnId) * 3600)
        : 0;
      return this._hold(turnId, selected ? 'impulse_not_strong_enough' : 'no_inner_motive', reevaluateAfterMs);
    }

    return this._speak(now, selected.action, this._reasonFor(selected.action, features), {
      turnId,
      phase,
      score: selected.score,
      entropy: stableEntropy(`${turnId}${selected.action}delivery`),
    });
  }

  /**
   * 察觉在场：开机见面 / 人脸出现。
   * 这是「两个人相见，总有人先开口」的主入口。
   */
  decidePresence(input = {}) {
    const now = Number(input.now) || Date.now();
    this._expireThread(now);
    const entropy = Number.isFinite(input.entropy) ? clamp(input.entropy) : stableEntropy(`${now}presence${this.state.sentCount}`);
    const dnd = input.dnd === true;
    const quotaBlocked = input.proactiveQuotaOk === false;
    const sinceNotice = now - Number(this.state.lastPresenceNoticeAt || 0);
    const sinceSpoken = now - Number(this.state.lastSpokenAt || 0);
    const sessionAge = now - Number(this.state.sessionStartedAt || now);
    if (dnd || quotaBlocked) {
      return { shouldSpeak: false, action: ACTIONS.HOLD, reason: dnd ? 'presence_or_dnd' : 'quota', nextCheckMs: 60000 };
    }
    // 同一次「看见你」不要连发；但见面窗口内应明显比空闲勤
    if (sinceNotice < 45000 || sinceSpoken < 16000) {
      return {
        shouldSpeak: false,
        action: ACTIONS.HOLD,
        reason: 'presence_cooldown',
        nextCheckMs: Math.max(8000, Math.min(40000, 45000 - sinceNotice)),
      };
    }

    const relScore = clamp(Number(input.relScore) || 0);
    // 刚打开 / 刚被看见：开口概率要高，否则没有「她感觉到我」
    const bootBoost = sessionAge < 3 * 60000 ? 0.22 : 0;
    const faceBoost = input.facePresent === true ? 0.16 : 0.08;
    const speakProbability = Math.min(0.92, 0.62 + bootBoost + faceBoost + relScore * 0.1 - Math.min(3, this.state.ignoredStreak) * 0.04);
    if (entropy > speakProbability) {
      return { shouldSpeak: false, action: ACTIONS.HOLD, reason: 'noticed_but_shy', nextCheckMs: 12000 + Math.floor(entropy * 20000) };
    }

    const bag = [ACTIONS.POKE, ACTIONS.POKE, ACTIONS.TEASE, ACTIONS.SHARE, ACTIONS.POKE];
    const action = bag[Math.min(bag.length - 1, Math.floor(entropy * bag.length))];
    this.state.lastPresenceNoticeAt = now;
    return this._speak(now, action, input.facePresent
      ? '她察觉到你在场，想先发出一点存在感'
      : '窗口打开了，她想先跟你打个照面', {
      phase: 'presence',
      entropy: stableEntropy(`${now}${action}presence`),
      contextFresh: false,
      useAnchor: false,
      presence: true,
    });
  }

  /**
   * 冷感回拉：对方敷衍/短答/气氛僵住时，换轻话题或闹一下。
   */
  decideColdness(input = {}) {
    const now = Number(input.now) || Date.now();
    this._expireThread(now);
    const features = analyzeTurn(input.lastUserText || input.userText, input.replyText || '');
    const entropy = Number.isFinite(input.entropy) ? clamp(input.entropy) : stableEntropy(`${now}cold${this.state.sentCount}`);
    if (input.dnd === true || input.proactiveQuotaOk === false) {
      return { shouldSpeak: false, action: ACTIONS.HOLD, reason: 'presence_or_dnd', nextCheckMs: 90000 };
    }
    if (!features.cold && !features.bored) {
      return { shouldSpeak: false, action: ACTIONS.HOLD, reason: 'not_cold', nextCheckMs: 45000 };
    }
    if (now - this.state.lastSpokenAt < 20000) {
      return { shouldSpeak: false, action: ACTIONS.HOLD, reason: 'adaptive_cooldown', nextCheckMs: 20000 };
    }
    if (entropy > 0.78) {
      return { shouldSpeak: false, action: ACTIONS.HOLD, reason: 'held_back', nextCheckMs: 25000 };
    }
    const action = entropy > 0.45 ? ACTIONS.TEASE : ACTIONS.POKE;
    return this._speak(now, action, '察觉到气氛有点冷/敷衍，想换轻松一点的动静把场子拉回来', {
      phase: 'coldness',
      entropy: stableEntropy(`${now}${action}cold`),
      contextFresh: false,
      useAnchor: false,
    });
  }

  /** 空闲主动：你还在，但一阵子没说话。 */
  decideIdle(input = {}) {
    const now = Number(input.now) || Date.now();
    this._expireThread(now);
    const idleMs = Math.max(0, Number(input.idleMs) || 0);
    const entropy = Number.isFinite(input.entropy) ? clamp(input.entropy) : stableEntropy(`${now}${this.state.sentCount}`);
    const presenceBlocked = input.dnd === true;
    const quotaBlocked = input.proactiveQuotaOk === false;
    const ignored = Math.min(3, Math.max(0, Number(this.state.ignoredStreak) || 0));
    const adaptiveGap = Math.min(4 * 60000, this.cooldownMs + ignored * 28000);
    const minIdle = Math.min(2 * 60000, 22000 + ignored * 16000);
    if (presenceBlocked || quotaBlocked || now - this.state.lastSpokenAt < adaptiveGap || idleMs < minIdle) {
      return {
        shouldSpeak: false,
        action: ACTIONS.HOLD,
        reason: presenceBlocked ? 'presence_or_dnd' : quotaBlocked ? 'quota' : 'adaptive_cooldown',
        nextCheckMs: Math.max(10000, Math.min(70000, Math.max(minIdle - idleMs, adaptiveGap - (now - this.state.lastSpokenAt)))),
      };
    }

    // 冷感优先走冷感回拉
    const lastUser = String(input.lastUserText || '');
    if (analyzeTurn(lastUser, '').cold && idleMs >= 25000) {
      return this.decideColdness({ ...input, now, entropy });
    }

    const contextFresh = input.contextFresh === true && input.topicContaminated !== true;
    const relScore = clamp(Number(input.relScore) || 0);
    const choices = contextFresh
      ? [ACTIONS.TEASE, ACTIONS.POKE, ACTIONS.SHARE, ACTIONS.POKE]
      : [ACTIONS.POKE, ACTIONS.POKE, ACTIONS.TEASE, ACTIONS.SHARE];
    let action = choices[Math.min(choices.length - 1, Math.floor(entropy * choices.length))];
    if (relScore < 0.15 && action === ACTIONS.TEASE) action = ACTIONS.POKE;
    const speakProbability = Math.min(0.88, 0.55 + Math.min(idleMs / 360000, 0.2) + relScore * 0.12 - ignored * 0.04);
    if (entropy > speakProbability) {
      return { shouldSpeak: false, action: ACTIONS.HOLD, reason: 'no_social_impulse', nextCheckMs: 18000 + Math.floor(entropy * 35000) };
    }

    return this._speak(now, action, contextFresh
      ? '他还在附近，她想随手接一下场子'
      : '安静太久，她想制造一点存在感', {
      phase: 'idle',
      entropy: stableEntropy(`${now}${action}idle`),
      contextFresh,
      useAnchor: contextFresh && action !== ACTIONS.POKE,
    });
  }

  registerSent(input = {}) {
    const now = Number(input.now) || Date.now();
    const text = compact(input.text);
    if (!text) return this.snapshot();
    const fragments = textFragments(text);
    const topicKey = fragments.sort((a, b) => b.length - a.length)[0] || '';
    this.state.lastSpokenAt = now;
    this.state.lastAction = String(input.action || this.state.lastAction || ACTIONS.POKE);
    this.state.sentCount += 1;
    this.state.activeThread = {
      id: `pro_${now}_${this.state.sentCount}`,
      sentAt: now,
      expiresAt: now + 10 * 60000,
      text: text.slice(0, 180),
      topicKey,
      replied: false,
    };
    if (topicKey) {
      this.state.recentTopics.push({ topicKey, at: now });
      this.state.recentTopics = this.state.recentTopics.filter((entry) => now - entry.at < 24 * 3600000).slice(-20);
    }
    this._save();
    return this.snapshot();
  }

  registerFeedback(input = {}) {
    const now = Number(input.now) || Date.now();
    const type = String(input.type || 'reply');
    const thread = this.state.activeThread;
    if (type === 'reply' && thread && now <= thread.expiresAt) {
      thread.replied = true;
      thread.repliedAt = now;
      this.state.engagedStreak = Math.min(8, this.state.engagedStreak + 1);
      this.state.ignoredStreak = 0;
      this.state.activeThread = null;
    } else if (type === 'presence') {
      this.state.ignoredStreak = Math.max(0, this.state.ignoredStreak - 1);
      if (this.state.ignoredStreak === 0) this.state.engagedStreak = Math.min(8, this.state.engagedStreak + 1);
    } else if (type === 'dismissed' || type === 'ignored') {
      this.state.ignoredStreak = Math.min(4, this.state.ignoredStreak + 1);
      this.state.engagedStreak = 0;
      this.state.activeThread = null;
    }
    this._save();
    return this.snapshot();
  }

  _expireThread(now = Date.now()) {
    const thread = this.state.activeThread;
    if (!thread || thread.replied || now <= Number(thread.expiresAt || 0)) return;
    this.state.ignoredStreak = Math.min(4, this.state.ignoredStreak + 1);
    this.state.engagedStreak = 0;
    this.state.activeThread = null;
    this._save();
  }

  _speak(now, action, reason, extra = {}) {
    this.state.lastSpokenAt = now;
    this.state.lastAction = action;
    if (extra.turnId) this.state.lastTurnId = extra.turnId;
    this.state.recentActions = [...this.state.recentActions, action].slice(-8);
    this._save();
    return {
      shouldSpeak: true,
      action,
      reason,
      turnId: extra.turnId || '',
      phase: extra.phase || 'idle',
      score: Number(Number(extra.score || 0).toFixed(3)),
      delivery: this._deliveryFor(action, extra.entropy || 0.5, extra.phase || 'silence'),
      contextFresh: extra.contextFresh === true,
      useAnchor: extra.useAnchor === true,
      presence: extra.presence === true,
      reevaluateAfterMs: 0,
      nextCheckMs: 0,
    };
  }

  _hold(turnId, reason, reevaluateAfterMs) {
    return { shouldSpeak: false, action: ACTIONS.HOLD, reason, turnId, reevaluateAfterMs };
  }

  _reasonFor(action, features) {
    if (action === ACTIONS.CARE) return '她察觉到情绪没有说完，想确认具体发生了什么';
    if (action === ACTIONS.PROBE) return '对方留下了私人叙述或未完线索，她产生了具体好奇';
    if (action === ACTIONS.STANCE) return features.science ? '话题触发了她的理科判断与争辩欲' : '她对这个观点形成了不同判断';
    if (action === ACTIONS.TEASE) return features.cold
      ? '气氛太冷，她想用轻微挖苦把你逗活'
      : '关系语境允许她接住玩笑，而不是礼貌结束';
    if (action === ACTIONS.POKE) {
      if (features.bored) return '他明显无聊，她想随手戳一下';
      if (features.cold) return '察觉到敷衍/冷感，只想发出一点动静把你拉回来';
      return '只想让你感觉到她还在——搞笑、骚扰或一声动静都可以';
    }
    return '这句话在她脑中引发了一个属于自己的联想';
  }

  _deliveryFor(action, entropy, phase) {
    const micro = action === ACTIONS.POKE || action === ACTIONS.TEASE;
    let bubbleCount = 1;
    if (micro && entropy > 0.28) bubbleCount = 2;
    if (micro && entropy > 0.68) bubbleCount = 3;
    if (!micro && (action === ACTIONS.SHARE || action === ACTIONS.CARE || action === ACTIONS.PROBE) && entropy > 0.55) {
      bubbleCount = 2;
    }
    const presencePhase = phase === 'presence' || phase === 'coldness';
    return {
      style: action === ACTIONS.POKE ? 'poke' : action === ACTIONS.TEASE ? 'banter'
        : action === ACTIONS.CARE ? 'soft' : action === ACTIONS.STANCE ? 'opinion' : 'casual',
      bubbleCount,
      maxCharsPerBubble: micro ? (presencePhase ? 20 : 24) : 36,
      pauseMinMs: presencePhase ? 380 : 480,
      pauseMaxMs: presencePhase ? 1100 : 1400,
      allowNonSemantic: micro,
    };
  }

  snapshot() {
    return { ...this.state, recentActions: [...this.state.recentActions] };
  }
}

module.exports = { ACTIONS, ConversationInitiativeEngine, analyzeTurn, fingerprint };
