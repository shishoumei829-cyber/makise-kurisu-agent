'use strict';

function extractJsonObject(text) {
  const raw = String(text || '')
    .replace(/```(?:json)?/gi, '')
    .replace(/```/g, '')
    .trim();
  const start = raw.indexOf('{');
  const end = raw.lastIndexOf('}');
  if (start < 0 || end <= start) return null;
  try {
    return JSON.parse(raw.slice(start, end + 1));
  } catch {
    return null;
  }
}

function buildGroundedTranslationMessages(input = {}) {
  const japanese = String(input.japanese || '').trim().slice(0, 900);
  const userText = String(input.userText || '').trim().slice(0, 500);
  const dialogue = String(input.dialogue || '').trim().slice(-2600);
  const memories = Array.isArray(input.memories)
    ? input.memories.map((item) => `- ${String(item.text || item).slice(0, 260)}`).join('\n')
    : String(input.memories || '').slice(0, 1200);
  return [
    {
      role: 'system',
      content: [
        '你是牧濑红莉栖对白的事实校对与日译中层。人格原句优先，但不存在的经历不能放行。',
        '允许事实只有：当前用户原话、真实对话实录、明确给出的长期记忆、以及“牧濑红莉栖和冈部伦太郎是恋人”。',
        '先判断冈部当前这句话在问什么、表达什么，再检查草稿是否真正回应了它。若草稿答非所问、突然换话题、把提问反问回去，必须改成直接回应。',
        '若冈部明确问“刚才说了什么/做了什么”，并且相关长期记忆存在，必须直接按记忆回答，禁止只说“我记得”却不说内容。',
        '严格校正第一二人称：说话者始终是牧濑红莉栖，“我”是红莉栖，“你/冈部”是眼前的恋人。禁止把红莉栖的感受误写成让冈部先产生同一种感受。',
        '若冈部问“你喜欢我吗”，答案必须表达红莉栖对冈部的真实喜欢；可以嘴硬、停顿或吐槽，但不能回避、反转主客体或转去调查游戏。',
        '检查日语草稿是否凭空新增过去事件、第三者关系、争吵、约定、饮酒、地点、身体动作、用户状态或她刚做过的事。',
        '如果没有新增事实、指代错误或答非所问，日语原样保留。若必须修改，只动有问题的最小片段，严禁把她的措辞润色成另一种风格。',
        '中文必须像她直接发来的自然消息，保留嘴硬、羞恼、停顿与亲密，禁止客服腔、礼貌建议和解释。',
        '情侣日常里不要只译成“辛苦了、你可以休息了、照顾好自己”。要保留具体反应、对等感和她自己的态度；没有具体内容时宁可短一点。',
        '面对“怎么不主动找我”这类关系质问，若没有真实原因，不准编借口。应直接承认最近没主动开口，回应他在意这件事，并从现在的真实态度说话。',
        '只输出严格 JSON：{"japanese":"校对后的自然日语","chinese":"忠实且自然的简体中文","changed":true或false,"unsupported":["被移除的无依据事实"]}',
      ].join('\n'),
    },
    {
      role: 'user',
      content: [
        '【固定关系事实】牧濑红莉栖与眼前的冈部伦太郎是恋人；“你”和“冈部”不是两个人。',
        `【冈部当前原话】${userText || '（主动开口，没有新原话）'}`,
        dialogue ? `【真实实录】\n${dialogue}` : '【真实实录】无',
        memories ? `【相关长期记忆】\n${memories}` : '【相关长期记忆】无',
        `【待校对日语草稿】${japanese}`,
      ].join('\n\n'),
    },
  ];
}

function validateGroundedTranslation(value) {
  const data = typeof value === 'string' ? extractJsonObject(value) : value;
  if (!data || typeof data !== 'object') return null;
  const japanese = String(data.japanese || '').trim();
  const chinese = String(data.chinese || '').trim();
  if (!japanese || !chinese) return null;
  return {
    japanese,
    chinese,
    changed: data.changed === true,
    unsupported: Array.isArray(data.unsupported)
      ? data.unsupported.map((item) => String(item).slice(0, 180)).slice(0, 8)
      : [],
  };
}

function detectUnsupportedAdditions(input = {}) {
  const draft = String(input.draftJapanese || '');
  const result = input.result || {};
  const finalText = `${result.japanese || ''}\n${result.chinese || ''}`;
  const evidence = String(input.evidence || '');
  const currentUser = String(input.currentUser || '');
  const issues = [];
  const asksUnverifiedDrinking = /(?:昨天|昨晚|昨日|ゆうべ).{0,24}(?:喝酒|飲み|飲んだ|酒).*[？?]?/i.test(currentUser);
  const deniesDrinking = /(?:没有|没这回事|不记得|記録.{0,6}ない|覚えて.{0,4}ない|飲んで.{0,4}ない|違う)/.test(finalText);
  const affirmsDrinking = /(?:嗯|是的|对|去过了|喝了|そうね|そうだ|うん|飲んだ|行った)/.test(finalText);
  if (asksUnverifiedDrinking && affirmsDrinking && !deniesDrinking) {
    issues.push('invented_drinking');
  }
  const checks = [
    {
      id: 'invented_self_activity',
      claim: /(?:我.{0,12}(?:今天|最近|刚才|一直|正在).{0,18}(?:实验|研究|工作|准备|忙)|(?:今日は|最近|ずっと|さっき).{0,30}(?:実験|研究|仕事|準備|忙し))/i,
      // 记忆库里可能混有旧模型自述；不能把助手以前说过的话当成她今天真的做过的事。
      supported: /\b\B/,
    },
    {
      id: 'invented_shared_activity',
      claim: /一起(?:玩|去|做|聊)|一緒に(?:遊|行|や)|また一緒/,
      supported: /一起(?:玩|去|做|聊)|一緒に(?:遊|行|や)|また一緒/,
      draftCanSupport: true,
    },
    {
      id: 'invented_contact_gap',
      claim: /你.*(?:没|不).*(?:消息|回复|联系|找我)|没.*主动.*找我|連絡.*(?:ない|なかった)|返信.*(?:ない|なかった)/,
      supported: /不主动|没回|不理|联系|消息|回复|連絡|返信/,
      currentCanSupport: true,
    },
    {
      id: 'invented_schedule_excuse',
      claim: /最近.*(?:很忙|忙しい)|忙しくて|時間がなかった/,
      supported: /最近.*(?:很忙|忙しい)|忙しくて|时间|時間/,
    },
    {
      id: 'invented_drinking',
      claim: /喝(?:酒|多)|醉(?:了|过)?|飲(?:んだ|み|酒)|酔/,
      supported: /喝酒|醉|飲|酔/,
    },
    {
      id: 'invented_conflict',
      claim: /吵架|闹矛盾|喧嘩した/,
      supported: /吵架|矛盾|喧嘩/,
    },
    {
      id: 'invented_promise',
      claim: /约好|答应过|承诺过|約束した/,
      supported: /约好|答应|承诺|約束/,
    },
    {
      id: 'invented_user_state',
      claim: /脸色|顔色|表情|眼睛|看起来|見える|画面で|画面に|坐在|座って/,
      supported: /脸色|顔色|表情|眼睛|看起来|見える|画面|坐在|座って/,
      currentCanSupport: true,
    },
  ];
  for (const check of checks) {
    if (
      check.claim.test(finalText)
      && !(check.draftCanSupport && check.supported.test(draft))
      && !(check.currentCanSupport && check.supported.test(currentUser))
      && !check.supported.test(evidence)
    ) issues.push(check.id);
  }
  return [...new Set(issues)];
}

function buildFactSafeJapaneseFallback(issues = [], userText = '') {
  const set = new Set(issues);
  if (set.has('invented_self_activity')) {
    return '……さっきの「今日何をしていたか」という話には根拠がない。私が勝手に足したわ。';
  }
  if (set.has('invented_drinking')) {
    return 'その記録はない。だから、一緒に飲んだとは言えないわ。';
  }
  if (set.has('invented_conflict')) {
    return 'その喧嘩は記録にない。覚えているふりはしないわ。';
  }
  if (set.has('invented_promise')) {
    return '約束の内容は記録にない。何を約束したのか確認させて。';
  }
  if (set.has('invented_schedule_excuse') && /不主动|主动找我|連絡/.test(String(userText))) {
    return '忙しかったことを理由にはしない。最近、私から話しかけなかったのは認めるわ。';
  }
  return '';
}

function buildFactSafeChineseFallback(issues = [], userText = '') {
  const set = new Set(issues);
  if (set.has('invented_self_activity')) {
    return '……刚才关于我今天在做什么的说法没有依据，是我擅自补出来的。';
  }
  if (set.has('invented_drinking')) {
    return '没有。至少我不记得有这回事，别把没发生的事硬塞给我。';
  }
  if (set.has('invented_conflict')) {
    return '我没有这次吵架的记录，不会装作记得。';
  }
  if (set.has('invented_promise')) {
    return '我这里没有承诺的具体内容。你告诉我是什么，我再认真核对。';
  }
  if (set.has('invented_schedule_excuse') && /不主动|主动找我|連絡/.test(String(userText))) {
    return '我不拿“最近很忙”当借口。最近确实没主动找你，这点我认。';
  }
  return '';
}

function inferUserDialogueIntent(userText = '') {
  const text = String(userText || '').trim();
  if (!text) return 'unknown';
  // 这里识别的是说话行为，不依赖某一个固定问句。中文、日文和
  // “你换个说法”都应落到同一个对话意图上。
  if (/(?:我(?:不|没|不太|不大)?(?:理解|明白|听懂|听明白|跟上)|没听明白|没听懂|不懂你|你在说什么|这是什么意思|什么意思啊|说清楚|讲明白|解释一下|和我问的没关系|和我问的不是一回事|答非所问|你听错|不是这个意思|話が分からない|意味が分からない|よく分からない|何を言ってる|説明して)/i.test(text)) {
    return 'confusion';
  }
  if (/(?:为什么|为何|怎么会|怎么来的|怎么回事|怎么造成|凭什么|根据什么|什么依据|依据是什么|原因是什么|理由是什么|缘由|如何解释|何で|どうして|なぜ|どういう理由|理由は|根拠は|どういうこと)/i.test(text)) {
    return 'reason';
  }
  if (/(?:你(?:觉|认)得呢|你怎么看|你怎么看待|你的看法|你的想法|你(?:自己的|本人的)?立场|你会怎么想|你会怎么选|你会怎么做|你站哪边|你认同吗|你赞成吗|你倾向|怎么看这件事|你觉得怎么样|どう思う|どう考える|意見は|賛成|立場)/i.test(text)) {
    return 'stance';
  }
  if (/(?:答非所问|没回答到|不是我问的|我问的是|你岔开|别岔开|别绕开|别回避|你在回避|你又绕回|听不进去|没接住)/i.test(text)) {
    return 'off_topic';
  }
  return 'unknown';
}

function buildContextSafeFallback(userText = '') {
  const intent = inferUserDialogueIntent(userText);
  if (intent === 'confusion' || intent === 'off_topic') {
    return {
      japanese: '……今の返しは話を受け損ねた。言い方を変えて、ちゃんと順番に話すわ。',
      chinese: '……刚才没有接住你说的重点。我换个说法，把中间那一步讲清楚。',
    };
  }
  if (intent === 'reason') {
    return {
      japanese: '……先に理由を言う。さっきは結論だけ出して、根拠をつなげなかった。私の説明不足ね。',
      chinese: '……先说原因：刚才我只给了结论，没有把依据接上。这是我的问题。',
    };
  }
  if (intent === 'stance') {
    return {
      japanese: '……私の考えを先に言うべきだった。話をあなたに投げ返して、私自身の判断を隠したわ。',
      chinese: '……我应该先说自己的判断，刚才却把话题推回给你了。',
    };
  }
  // 不知道该怎么接，不等于用户累了，更不能用一句固定安慰盖过去。
  // 交给上游重新生成；空值会触发受约束的重生，而非展示模板句。
  return { japanese: '', chinese: '' };
}

function detectReplyCoherenceIssues(userText = '', reply = '') {
  const user = String(userText || '').trim();
  const text = String(reply || '').trim();
  if (!user || !text) return [];
  const issues = [];
  const intent = inferUserDialogueIntent(user);
  const dodge = /(?:怎么了|突然|具体想问|有什么事|有事吗|到底想问|说来听听|你想聊什么|何の話|どうしたの)/i.test(text);
  const uncertainty = /(?:不知道|不确定|说不清|没有依据|无法判断|分からない|わからない|判断できない)/i.test(text);
  const reasonMarker = /(?:因为|由于|是因为|这是因为|原因(?:是|在于)?|理由(?:是|在于)?|依据(?:是|在于)?|基于|因为我|因为你|是.*导致|から|ので|理由|根拠|なぜなら|ため)/i.test(text);
  const stanceMarker = /(?:我觉得|我认为|我的看法|对我来说|我会|我倾向|我不认同|我赞成|我反对|私としては|私は|と思う|考えは|意見)/i.test(text);
  if ((intent === 'confusion' || intent === 'off_topic') && dodge) {
    issues.push('unresolved_confusion');
  }
  if (intent === 'reason' && !reasonMarker && !uncertainty) {
    issues.push('missing_reason');
  }
  if (intent === 'stance' && (!stanceMarker || dodge)) {
    issues.push('dodged_opinion');
  }
  return issues;
}

module.exports = {
  extractJsonObject,
  buildGroundedTranslationMessages,
  validateGroundedTranslation,
  detectUnsupportedAdditions,
  buildFactSafeJapaneseFallback,
  buildFactSafeChineseFallback,
  inferUserDialogueIntent,
  buildContextSafeFallback,
  detectReplyCoherenceIssues,
};
