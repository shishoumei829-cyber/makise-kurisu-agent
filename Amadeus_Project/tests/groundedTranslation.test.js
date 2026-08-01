'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const {
  buildGroundedTranslationMessages,
  validateGroundedTranslation,
  detectUnsupportedAdditions,
  buildFactSafeJapaneseFallback,
  buildFactSafeChineseFallback,
  inferUserDialogueIntent,
  buildContextSafeFallback,
  detectReplyCoherenceIssues,
} = require('../lib/groundedTranslation');

test('grounding prompt limits facts and fixes Okabe identity split', () => {
  const messages = buildGroundedTranslationMessages({
    japanese: '岡部と喧嘩してるから、あなたを忘れた',
    userText: '你怎么不主动找我',
    dialogue: '冈部：你怎么不主动找我',
  });
  const prompt = messages.map((item) => item.content).join('\n');
  assert.match(prompt, /不存在的经历不能放行/);
  assert.match(prompt, /“你”和“冈部”不是两个人/);
  assert.match(prompt, /争吵/);
});

test('a user question is not evidence that a past drinking event happened', () => {
  const issues = detectUnsupportedAdditions({
    draftJapanese: '昨日は一緒に飲んだ。',
    result: {
      japanese: '昨日は一緒に飲んだ。',
      chinese: '昨天我们一起喝酒了。',
    },
    currentUser: '我们昨天是不是一起去喝酒了？',
    evidence: '',
  });
  assert.deepEqual(issues, ['invented_drinking']);
  assert.match(buildFactSafeJapaneseFallback(issues), /記録はない/);
});

test('strict JSON result is parsed', () => {
  const out = validateGroundedTranslation('```json\n{"japanese":"別に忘れてない。","chinese":"我才没忘。","changed":true,"unsupported":["和冈部吵架"]}\n```');
  assert.equal(out.japanese, '別に忘れてない。');
  assert.equal(out.chinese, '我才没忘。');
  assert.equal(out.changed, true);
  assert.deepEqual(out.unsupported, ['和冈部吵架']);
});

test('grounding cannot invent a new shared activity while rewriting', () => {
  const issues = detectUnsupportedAdditions({
    draftJapanese: 'お疲れ様。私はここで待ってる。',
    result: {
      japanese: '少し休んで、また一緒に遊ぼう。',
      chinese: '休息一下，我们再一起玩吧。',
    },
    evidence: '冈部：我健身回来，累死了。',
  });
  assert.deepEqual(issues, ['invented_shared_activity']);
});

test('grounding rejects unsupported self activity claims', () => {
  const issues = detectUnsupportedAdditions({
    draftJapanese: '\u4eca\u65e5\u306f\u305a\u3063\u3068\u5b9f\u9a13\u306e\u6e96\u5099\u3092\u3057\u3066\u3044\u305f',
    result: {
      japanese: '\u4eca\u65e5\u306f\u305a\u3063\u3068\u5b9f\u9a13\u306e\u6e96\u5099\u3092\u3057\u3066\u3044\u305f',
      chinese: '\u6211\u4eca\u5929\u4e00\u76f4\u5728\u51c6\u5907\u5b9e\u9a8c\u3002',
    },
    currentUser: '\u611f\u89c9\u4e00\u5929\u8fc7\u5f97\u597d\u5feb',
    evidence: '',
  });
  assert.deepEqual(issues, ['invented_self_activity']);
  assert.match(buildFactSafeChineseFallback(issues), /\u6ca1\u6709\u4f9d\u636e/);
});

test('coherence repair answers confusion and why instead of asking another question', () => {
  assert.deepEqual(
    detectReplyCoherenceIssues('\u6211\u4e0d\u7406\u89e3\u4f60\u8ddf\u6211\u8bf4\u7684\u8fd9\u4e9b\u8bdd', '\u54fc\u2026\u7a81\u7136\u600e\u4e48\u4e86\uff1f\u5230\u5e95\u60f3\u95ee\u4ec0\u4e48\uff1f'),
    ['unresolved_confusion'],
  );
  assert.deepEqual(
    detectReplyCoherenceIssues('\u4e3a\u4ec0\u4e48\u5b89\u5fc3\u5462', '\u6240\u4ee5\u8fd9\u6837\u60f3\u6765\uff0c\u591a\u5c11\u4f1a\u5b89\u5fc3\u4e00\u70b9\u5427'),
    ['missing_reason'],
  );
  assert.match(buildContextSafeFallback('\u6211\u4e0d\u7406\u89e3\u4f60\u8ddf\u6211\u8bf4\u7684\u8fd9\u4e9b\u8bdd').chinese, /\u6ca1\u6709\u63a5\u4f4f|\u8bb2\u6e05\u695a/);
});

test('coherence intent is semantic enough to survive alternate wording', () => {
  assert.equal(inferUserDialogueIntent('你到底凭什么这么判断'), 'reason');
  assert.equal(inferUserDialogueIntent('那你自己的立场是什么'), 'stance');
  assert.equal(inferUserDialogueIntent('你刚才说的和我问的不是一回事'), 'confusion');
  assert.deepEqual(
    detectReplyCoherenceIssues('那你自己的立场是什么', '哼，怎么突然问这个？你具体想问什么？'),
    ['dodged_opinion'],
  );
  assert.deepEqual(
    detectReplyCoherenceIssues('你到底凭什么这么判断', '所以，就这样吧。'),
    ['missing_reason'],
  );
});
