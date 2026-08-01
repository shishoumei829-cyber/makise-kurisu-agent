'use strict';

const ACTIONS = new Set(['care', 'probe', 'stance', 'tease', 'share', 'poke', 'thought']);
const FILLER = /(?:私|僕|俺|あなた|彼|彼女|これ|それ|あれ|今日|今|さっき|もう|やっと|本当|ちょっと|こと|もの|ため|から|ので|です|ます|だ|だった|でした|我|你|他|她|我们|今天|现在|刚才|终于|真的|有点|这个|那个|事情|已经|了|的|是|在|把|就|也)/g;

function compact(value) {
  return String(value || '').replace(/\s+/g, ' ').trim();
}

function topicTokens(value) {
  const text = compact(value)
    .toLowerCase()
    .replace(/[，。！？!?；;、…“”‘’（）()\[\]【】\s]/g, '')
    .replace(FILLER, '');
  const out = new Set();
  for (let size = 2; size <= Math.min(5, text.length); size += 1) {
    for (let i = 0; i <= text.length - size; i += 1) {
      const token = text.slice(i, i + size);
      if (!/^(?:する|した|です|ます|こと|もの|から|ので)$/u.test(token)) out.add(token);
    }
  }
  return [...out].sort((a, b) => b.length - a.length);
}

function isVulnerable(value) {
  return /累|疲|困|睡不着|难过|烦|焦虑|害怕|孤独|撑不住|想哭|失望|迷茫|空虚|疲れ|眠い|つら|しんど|悩|不安|怖/.test(String(value || ''));
}

function actionRule(action, vulnerable) {
  const rules = {
    care: vulnerable
      ? '相手の具体的な状態を一箇所だけ受け止める。助言や定型慰めで埋めない。'
      : '心配や体調を勝手に想定しない。具体的な事実への短い反応にする。',
    probe: '相手が言った一点だけを本当に知りたい時に聞く。事実をそのまま聞き返さない。',
    stance: 'その話について自分の判断か反論を一つ言う。質問で締めない。',
    tease: '共有された具体的な事実に軽く突っ込むか、少し意地悪な評価を一つ言う。質問で締めない。',
    share: '相手の話から生まれた自分の連想・見方を一つ共有する。新しい出来事は作らない。',
    poke: '文脈にある一点へ、短い存在感か反応を置く。安否確認や定時挨拶にしない。',
    thought: '持続していた自分の具体的な考えを言う。説明や質問へ逃がさない。',
  };
  return rules[action] || rules.poke;
}

/**
 * 主动发话先确定“这一次到底要表达什么”。这里不产出台词，
 * 只产出可追溯的内容核；没有内容核就不允许模型填空式开口。
 */
function buildProactiveContentPlan(input = {}) {
  const requestedAction = String(input.action || 'poke').toLowerCase();
  let action = ACTIONS.has(requestedAction) ? requestedAction : 'poke';
  const anchor = compact(input.anchor);
  const thought = compact(input.thought);
  const contextFresh = input.contextFresh === true;
  const vulnerable = isVulnerable(anchor);

  let origin = '';
  let subject = '';
  if (contextFresh && anchor) {
    origin = 'last_user_turn';
    subject = anchor;
  } else if (thought) {
    origin = 'persistent_thought';
    subject = thought;
    action = 'thought';
  }

  if (!subject) {
    return {
      shouldGenerate: false,
      reason: 'no_specific_content_core',
      action: 'hold',
      origin: '',
      subject: '',
      topicTokens: [],
      allowQuestion: false,
      rule: '',
    };
  }

  return {
    shouldGenerate: true,
    action,
    origin,
    subject,
    vulnerable,
    topicTokens: topicTokens(subject).slice(0, 12),
    allowQuestion: action === 'probe',
    rule: actionRule(action, vulnerable),
  };
}

function buildJapaneseContentPlanBlock(plan) {
  if (!plan?.shouldGenerate) return '【発話の核】今は具体的に言いたいことがない。必ず [SILENCE] だけを返す。';
  return [
    '【今回の発話の核】台詞を作る前に、これだけを自分の理由として受け取る。',
    `出所：${plan.origin === 'persistent_thought' ? '心に残っていた自分の考え' : '彼の直前の発言'}`,
    `具体的な対象：${plan.subject}`,
    `表現の方針：${plan.rule}`,
    plan.allowQuestion
      ? '一つだけ聞いてよい。ただし相手が断定した事実を確認し直さない。'
      : '質問形で終えない。相手に会話を続けさせるためだけの問いを置かない。',
    '対象と無関係な疲労・心配・電話・実験・予定を足さない。',
  ].join('\n');
}

function buildJapanesePlanRepair(plan) {
  return [
    '直前の文は発話の核を守れていない。台詞だけを書き直す。',
    `対象は「${plan.subject}」。${plan.rule}`,
    plan.allowQuestion ? '聞くなら一点だけ。' : '質問で終えない。',
    '対象にない心配、疲労、電話、予定、実験を足さない。できなければ [SILENCE]。',
  ].join('\n');
}

function validateProactiveContent(reply, plan) {
  const text = compact(reply);
  if (!plan?.shouldGenerate) return { ok: !text || /^\[?silence\]?$/i.test(text), reason: 'no_content_core' };
  if (!text || /^\[?silence\]?$/i.test(text)) return { ok: false, reason: 'empty' };
  // A question hidden before a second bubble/parenthesis is still a forced
  // follow-up.  Active speech may not smuggle the old "answer then ask" habit.
  if (!plan.allowQuestion && /[？?]/.test(text)) return { ok: false, reason: 'forced_question' };
  if (!plan.vulnerable && /疲れて|疲れた|無理しない|心配|大丈夫|眠|累了|担心|没事吧/.test(text)) {
    return { ok: false, reason: 'ungrounded_care' };
  }
  const compactSubject = compact(plan.subject).replace(/[，。！？!?、\s]/g, '');
  const compactReply = text.replace(/[，。！？!?、\s]/g, '');
  if (compactSubject.length >= 8 && compactReply === compactSubject) return { ok: false, reason: 'mere_repeat' };
  const tokens = Array.isArray(plan.topicTokens) ? plan.topicTokens : [];
  if (tokens.length && !tokens.some((token) => token.length >= 2 && compactReply.includes(token))) {
    // Human conversation often carries a just-mentioned concrete subject with
    // a natural anaphora ("that hurdle", "finally", "got out of it") instead
    // of repeating the noun.  Accept only narrow, explainable continuations;
    // do not turn this into a generic-care escape hatch.
    const anchoredContinuation = /(?:あの|その|やっと|終わ|片付|山場|這い出|例の|あれ|总算|终于|写完|报告)/.test(text);
    if (anchoredContinuation && !/疲れ|心配|大丈夫|累了|担心|没事吧/.test(text)) {
      return { ok: true, reason: 'anaphoric_subject_continuation' };
    }
    return { ok: false, reason: 'lost_subject' };
  }
  return { ok: true, reason: '' };
}

module.exports = {
  buildProactiveContentPlan,
  buildJapaneseContentPlanBlock,
  buildJapanesePlanRepair,
  validateProactiveContent,
  topicTokens,
};
