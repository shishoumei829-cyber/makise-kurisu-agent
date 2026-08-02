'use strict';

/**
 * 主动对话设计规范（人对人，不是定时器）：
 *
 * 1. 坐在旁边：摄像头只回答「他还在不在」。要不要讲话、讲什么、什么时候讲，
 *    由内驱冲动 × 读场时机决定——可以只是安静待着。
 * 2. 存在感：动机可以是搞笑、骚扰、开玩笑、发出一点动静——社交动作本身可以是目的。
 * 3. 人格不让位：动机可以轻，措辞必须是牧濑红莉栖（聪明、嘴硬、熟人拌嘴）。
 * 4. 冷感回拉：察觉冷漠、敷衍、沉默变僵时，换轻话题或闹一下，而不是继续查岗。
 * 5. 对话内跟话：仍有动机才叠话；任务句/告别/她刚反问时不抢。
 * 6. 勿扰仍尊重：明确忙碌/睡觉时闭嘴。
 */

const fs = require('fs');
const path = require('path');
const { textFragments } = require('../lib/memoryAdmission');
const { shouldSpeakNow } = require('./socialRead');

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
      activeThoughtId: '',
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
    // 有动机就说：不再用 entropy 骰子挡掉——沉默只因动机不够或对话边界
    if (!selected || selected.score <= holdScore) {
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
      entropy: 0.4,
    });
  }

  /**
   * 在场感开口。
   * - 尚未开聊：可以是「先打个照面」
   * - 已经在聊：仍然可以主动插话（不必你一句我一句），但情景必须是「已在同一窗口」，
   *   绝不能演「刚刚才注意到 / 电话还没打过来」——那才是不合理，不是她不该存在。
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
    const alreadyTalking = input.alreadyTalking === true || input.dialogueStarted === true;
    const lastUser = String(input.lastUserText || input.userText || '').trim();
    const senseDriven = input.senseDriven === true || input.eventDriven === true;
    const facePresent = input.facePresent === true;
    if (dnd) {
      return { shouldSpeak: false, action: ACTIONS.HOLD, reason: 'presence_or_dnd', nextCheckMs: 60000 };
    }
    // 感知在场：没看见人就不开 presence（强制观察除外）
    if (senseDriven && !facePresent && input.force !== true) {
      return { shouldSpeak: false, action: ACTIONS.HOLD, reason: 'no_face', nextCheckMs: 4000 };
    }
    // 同一次冲动不要连发；已在聊时冷却略短
    const noticeGap = alreadyTalking ? 28000 : 45000;
    const speakGap = alreadyTalking ? 12000 : 16000;
    if (sinceNotice < noticeGap || sinceSpoken < speakGap) {
      return {
        shouldSpeak: false,
        action: ACTIONS.HOLD,
        reason: 'presence_cooldown',
        nextCheckMs: Math.max(8000, Math.min(40000, noticeGap - sinceNotice)),
      };
    }

    if (input.pendingUserTurn === true || input.isThinking === true) {
      return {
        shouldSpeak: false,
        action: ACTIONS.HOLD,
        reason: 'pending_user_turn',
        nextCheckMs: 6000,
        phase: alreadyTalking ? 'copresence' : 'presence',
      };
    }

    const idleSinceUser = Math.max(0, Number(input.idleMsSinceUser) || 0);
    if (alreadyTalking && idleSinceUser < 120000) {
      const features = analyzeTurn(lastUser, '');
      if (!features.cold && !features.bored) {
        return {
          shouldSpeak: false,
          action: ACTIONS.HOLD,
          reason: 'active_chat_quiet',
          nextCheckMs: Math.max(8000, 120000 - idleSinceUser),
          phase: 'copresence',
        };
      }
    }

    // 按内容选动机，不用骰子抽 bag；配额不再挡「想说」
    void quotaBlocked;
    void sessionAge;
    void entropy;
    let action = ACTIONS.POKE;
    if (/无聊|没事干/.test(lastUser)) action = ACTIONS.TEASE;
    else if (alreadyTalking && lastUser.length > 20) action = ACTIONS.SHARE;
    else if (!alreadyTalking) action = ACTIONS.POKE;
    this.state.lastPresenceNoticeAt = now;

    if (alreadyTalking) {
      return this._speak(now, action, lastUser
        ? '你们已经在同一窗口里；她想再插一句/接一下场子，不是第一次发现他'
        : '你们已经在同一窗口里；她想发出一点存在感，不是打电话也不是刚察觉', {
        phase: 'copresence',
        entropy: stableEntropy(`${now}${action}copresence`),
        contextFresh: !!lastUser,
        useAnchor: !!lastUser,
        presence: true,
      });
    }

    return this._speak(now, action, facePresent
      ? '她正看着你，想先发出一点存在感'
      : '窗口打开了，她想先跟你打个照面', {
      phase: 'presence',
      entropy: stableEntropy(`${now}${action}presence`),
      contextFresh: false,
      useAnchor: false,
      presence: true,
    });
  }

  /**
   * 共在开口：内驱已经想说 + 读场允许 → 决定 action/phase。
   * 不再用「看见脸 / 看够 N 秒」当扳机。
   */
  decideBeside(input = {}) {
    const now = Number(input.now) || Date.now();
    this._expireThread(now);
    if (input.pendingUserTurn === true || input.isThinking === true) {
      return {
        shouldSpeak: false,
        action: ACTIONS.HOLD,
        reason: input.pendingUserTurn ? 'pending_user_turn' : 'is_thinking',
        nextCheckMs: 6000,
        phase: input.alreadyTalking ? 'copresence' : 'presence',
      };
    }
    if (input.dnd === true) {
      return {
        shouldSpeak: false,
        action: ACTIONS.HOLD,
        reason: 'presence_or_dnd',
        nextCheckMs: 60000,
        phase: 'copresence',
      };
    }
    if (input.autonomyShouldAct !== true) {
      return {
        shouldSpeak: false,
        action: ACTIONS.HOLD,
        reason: input.autonomyReason || 'no_urge_or_moment',
        nextCheckMs: Number(input.nextCheckMs) || 8000,
        phase: input.alreadyTalking ? 'copresence' : 'presence',
        speakHint: input.speakHint || '',
      };
    }

    const alreadyTalking = input.alreadyTalking === true || input.dialogueStarted === true;
    const idleMs = Math.max(0, Number(input.idleMs) || 0);
    const socialPre = input.social || {};
    // 正在正常聊天：别插队半截谜语；等他安静一会儿再说（冷感除外）
    if (alreadyTalking && idleMs < 75000 && !socialPre.cold && !socialPre.bored) {
      return {
        shouldSpeak: false,
        action: ACTIONS.HOLD,
        reason: 'active_chat_quiet',
        nextCheckMs: Math.max(8000, 75000 - idleMs),
        phase: 'copresence',
      };
    }

    const sinceSpoken = now - Number(this.state.lastSpokenAt || 0);
    if (sinceSpoken < 12000) {
      return {
        shouldSpeak: false,
        action: ACTIONS.HOLD,
        reason: 'just_spoke',
        nextCheckMs: 12000 - sinceSpoken + 1500,
        phase: 'copresence',
      };
    }

    const social = input.social || {};
    const lastUser = String(input.lastUserText || '').trim();
    const intentKey = String(input.urgeIntentKey || input.urgeIntent || '').toUpperCase();
    let action = ACTIONS.POKE;
    if (/PLAYFUL|TEASE|JAB/.test(intentKey)) action = ACTIONS.TEASE;
    else if (/ASK|PROBE|QUESTION|EXPLORE/.test(intentKey)) action = ACTIONS.PROBE;
    else if (/CARE|BOND|REACH|DEEPEN|CONNECTION/.test(intentKey)) action = ACTIONS.POKE;
    else if (/CREATE|SHARE|IDEA|MEANING|SELF/.test(intentKey)) action = ACTIONS.SHARE;
    else if (/STANCE|DEFEND/.test(intentKey)) action = ACTIONS.STANCE;

    let phase = 'presence';
    if (alreadyTalking) phase = social.cold || social.bored ? 'coldness' : 'copresence';
    else if (social.facePresent) phase = 'presence';

    const reason = input.speakHint
      || (alreadyTalking
        ? '人在旁边，心里有点想开口——不是第一次发现他'
        : '人在旁边，心里攒了一点想说的话');

    return this._speak(now, action, reason, {
      phase,
      entropy: stableEntropy(`${now}${action}beside`),
      contextFresh: alreadyTalking && !!lastUser,
      useAnchor: alreadyTalking && !!lastUser && (social.tension > 0.35 || action === ACTIONS.PROBE),
      presence: true,
    });
  }

  /**
   * 主体思维流开口：对话引擎只负责社交边界和冷却，不再从关键词猜一种
   * probe/poke/tease 动作。说什么来自 SoulRuntime 中已经持续存在的念头。
   */
  decideThought(input = {}) {
    const now = Number(input.now) || Date.now();
    this._expireThread(now);
    if (
      input.dnd === true
      || input.pendingUserTurn === true
      || input.isThinking === true
      || input.awaitingReply === true
    ) {
      return {
        shouldSpeak: false,
        action: ACTIONS.HOLD,
        reason: 'social_boundary',
        nextCheckMs: 60000,
      };
    }
    const thoughtId = String(input.thoughtId || '');
    const thought = compact(input.thought);
    if (!thoughtId || !thought) {
      return {
        shouldSpeak: false,
        action: ACTIONS.HOLD,
        reason: 'no_unfinished_thought',
        nextCheckMs: Math.max(60000, Number(input.nextCheckMs) || 90000),
      };
    }
    // 内在念头只是“想说”，不是“现在可以打断”。真正开口前必须经过
    // 读场：用户是否仍在说话、TTS 是否占用声道、沉默是否舒服、关系
    // 和情绪是否足以抵消插话成本。没有 social 时保留纯引擎调用的兼容性，
    // 但服务端的实际主动链路始终传入 social。
    if (input.social && input.socialGate !== false) {
      const gate = shouldSpeakNow(Number(input.tension) || 0, input.social, {
        relScore: input.relScore,
      });
      if (!gate.ok) {
        return {
          shouldSpeak: false,
          action: ACTIONS.HOLD,
          reason: `social_${gate.reason || 'not_now'}`,
          nextCheckMs: Math.max(8000, Number(input.nextCheckMs) || 12000),
          phase: String(input.phase || 'idle'),
        };
      }
    }
    const ignored = Math.max(0, Number(this.state.ignoredStreak) || 0);
    const gap = Math.min(8 * 60000, this.cooldownMs * 2 + ignored * 90000);
    const elapsed = now - Number(this.state.lastSpokenAt || 0);
    if (elapsed < gap) {
      return {
        shouldSpeak: false,
        action: ACTIONS.HOLD,
        reason: 'thought_refractory',
        nextCheckMs: Math.max(15000, gap - elapsed),
      };
    }
    return this._speak(now, 'thought', String(input.reason || thought), {
      phase: String(input.phase || 'idle'),
      contextFresh: input.contextFresh === true,
      useAnchor: false,
      thoughtId,
      thought,
      desire: compact(input.desire),
      score: Number(input.tension) || 0,
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
    if (input.dnd === true) {
      return { shouldSpeak: false, action: ACTIONS.HOLD, reason: 'presence_or_dnd', nextCheckMs: 90000 };
    }
    if (!features.cold && !features.bored) {
      return { shouldSpeak: false, action: ACTIONS.HOLD, reason: 'not_cold', nextCheckMs: 45000 };
    }
    if (now - this.state.lastSpokenAt < 20000) {
      return { shouldSpeak: false, action: ACTIONS.HOLD, reason: 'adaptive_cooldown', nextCheckMs: 20000 };
    }
    const action = features.bored ? ACTIONS.TEASE : ACTIONS.POKE;
    void entropy;
    const hasAnchor = String(input.lastUserText || input.userText || '').trim().length > 0;
    return this._speak(now, action, features.bored
      ? '他嫌无聊——接他处境给一点具体动静，禁止「又是这个话题吗/好无聊」收束'
      : '察觉到气氛有点冷/敷衍，想换轻松一点的动静把场子拉回来', {
      phase: 'coldness',
      entropy: stableEntropy(`${now}${action}cold`),
      // 有上一句就接上一句，禁止空降「刚注意到/打电话」
      contextFresh: hasAnchor,
      useAnchor: hasAnchor,
    });
  }

  /** 空闲主动：你还在，但一阵子没说话。 */
  decideIdle(input = {}) {
    const now = Number(input.now) || Date.now();
    this._expireThread(now);
    const idleMs = Math.max(0, Number(input.idleMs) || 0);
    const presenceBlocked = input.dnd === true;
    const ignored = Math.min(3, Math.max(0, Number(this.state.ignoredStreak) || 0));
    const adaptiveGap = Math.min(4 * 60000, this.cooldownMs + ignored * 28000);
    const minIdle = Math.min(2 * 60000, 22000 + ignored * 16000);
    // 冷却与勿扰仍尊重；配额与骰子不再挡
    if (presenceBlocked || now - this.state.lastSpokenAt < adaptiveGap || idleMs < minIdle) {
      return {
        shouldSpeak: false,
        action: ACTIONS.HOLD,
        reason: presenceBlocked ? 'presence_or_dnd' : 'adaptive_cooldown',
        nextCheckMs: Math.max(10000, Math.min(70000, Math.max(minIdle - idleMs, adaptiveGap - (now - this.state.lastSpokenAt)))),
      };
    }

    const lastUser = String(input.lastUserText || '');
    if (analyzeTurn(lastUser, '').cold && idleMs >= 25000) {
      return this.decideColdness({ ...input, now });
    }

    const contextFresh = input.contextFresh === true && input.topicContaminated !== true;
    const relScore = clamp(Number(input.relScore) || 0);
    let action = contextFresh ? ACTIONS.SHARE : ACTIONS.POKE;
    if (/无聊/.test(lastUser)) action = ACTIONS.TEASE;
    if (relScore < 0.15 && action === ACTIONS.TEASE) action = ACTIONS.POKE;

    return this._speak(now, action, contextFresh
      ? '他还在附近，她想随手接一下场子'
      : '安静太久，她想制造一点存在感', {
      phase: 'idle',
      entropy: 0.4,
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
    const thoughtId = String(input.thoughtId || this.state.activeThoughtId || '');
    if (thoughtId) this.state.activeThoughtId = thoughtId;
    this.state.sentCount += 1;
    this.state.activeThread = {
      id: `pro_${now}_${this.state.sentCount}`,
      sentAt: now,
      expiresAt: now + 10 * 60000,
      text: text.slice(0, 180),
      topicKey,
      replied: false,
      thoughtId,
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
      this.state.activeThoughtId = '';
    } else if (type === 'presence') {
      this.state.ignoredStreak = Math.max(0, this.state.ignoredStreak - 1);
      if (this.state.ignoredStreak === 0) this.state.engagedStreak = Math.min(8, this.state.engagedStreak + 1);
    } else if (type === 'dismissed' || type === 'ignored') {
      this.state.ignoredStreak = Math.min(4, this.state.ignoredStreak + 1);
      this.state.engagedStreak = 0;
      this.state.activeThread = null;
      this.state.activeThoughtId = '';
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
    if (extra.thoughtId) this.state.activeThoughtId = String(extra.thoughtId);
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
      thoughtId: extra.thoughtId || '',
      thought: extra.thought || '',
      desire: extra.desire || '',
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

  _deliveryFor(action, _entropy, phase) {
    const micro = action === ACTIONS.POKE || action === ACTIONS.TEASE;
    const presencePhase = phase === 'presence' || phase === 'copresence' || phase === 'coldness';
    const expansive = action === ACTIONS.CARE
      || action === ACTIONS.STANCE
      || action === ACTIONS.SHARE;
    return {
      style: action === ACTIONS.POKE ? 'poke' : action === ACTIONS.TEASE ? 'banter'
        : action === ACTIONS.CARE ? 'soft' : action === ACTIONS.STANCE ? 'opinion' : 'casual',
      bubbleCount: micro ? 2 : expansive ? 4 : 3,
      maxCharsPerBubble: micro ? 44 : 72,
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
