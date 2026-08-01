'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const {
  getReplyLanguageMode,
  validateJapaneseOutput,
  extractJapaneseBody,
  stripModelDecorations,
  buildLiteralJpToCnMessages,
  alignLiteralCnToJapanese,
  LITERAL_JP_TO_CN_SYSTEM,
} = require('../lib/replyLanguage');
const { buildPrompt } = require('../cognitive/prompts');

test('Japanese validation accepts natural Japanese', () => {
  assert.equal(validateJapaneseOutput('いるわよ。何か用？').ok, true);
});

test('Japanese validation rejects Chinese-only and Cyrillic output', () => {
  assert.deepEqual(validateJapaneseOutput('我在这里').issues, ['missing-kana']);
  assert.ok(validateJapaneseOutput('Привет、いるよ').issues.includes('cyrillic'));
});

test('extractJapaneseBody removes labels and Chinese tail', () => {
  assert.equal(
    extractJapaneseBody('JP: そんな名前で呼ばないで。\nCN: 别那么叫我。'),
    'そんな名前で呼ばないで。',
  );
});

test('stripModelDecorations keeps mixed JP+CN body for literal display source', () => {
  const raw = 'えと、今何してるの？… 分心啊';
  assert.equal(stripModelDecorations(raw), raw);
});

test('display JP→CN messages require faithful readable Chinese', () => {
  const msgs = buildLiteralJpToCnMessages('えと、今何してるの？… 分心啊');
  assert.equal(msgs.length, 2);
  assert.equal(msgs[0].role, 'system');
  assert.equal(msgs[0].content, LITERAL_JP_TO_CN_SYSTEM);
  assert.match(msgs[0].content, /忠实|可读/);
  assert.match(msgs[0].content, /不要脑补|不增删事实/);
  assert.match(msgs[0].content, /谜语|断句乱码/);
  assert.match(msgs[0].content, /もう≠再三/);
  assert.equal(msgs[1].content, 'えと、今何してるの？… 分心啊');
});

test('alignLiteralCnToJapanese fixes もう→再三 mistranslation', () => {
  assert.equal(
    alignLiteralCnToJapanese('もう？ まだ心配してるの', '再三？还在担心吗？'),
    '又？还在担心吗？',
  );
  assert.equal(
    alignLiteralCnToJapanese('もう？まだ心配してるの', '又？还在担心吗？'),
    '又？还在担心吗？',
  );
});

test('stripConsciousnessEcho removes leaked intention tags', () => {
  const { stripConsciousnessEcho } = require('../lib/replyLanguage');
  assert.equal(
    stripConsciousnessEcho('[打算] 我会先找话题——然后就闲聊。'),
    '',
  );
  assert.equal(
    stripConsciousnessEcho('哦？什么时候开始在意了。\n[打算] 我会先找话题——然后就闲聊。'),
    '哦？什么时候开始在意了。',
  );
});

test('Kurisu version models default to Japanese mode', () => {
  const oldLanguage = process.env.AMADEUS_REPLY_LANGUAGE;
  const oldModel = process.env.AMADEUS_CHAT_MODEL;
  delete process.env.AMADEUS_REPLY_LANGUAGE;
  process.env.AMADEUS_CHAT_MODEL = 'kurisu-v4-candidate';
  assert.equal(getReplyLanguageMode(), 'ja');
  if (oldLanguage == null) delete process.env.AMADEUS_REPLY_LANGUAGE;
  else process.env.AMADEUS_REPLY_LANGUAGE = oldLanguage;
  if (oldModel == null) delete process.env.AMADEUS_CHAT_MODEL;
  else process.env.AMADEUS_CHAT_MODEL = oldModel;
});

test('Swallow Japanese prompt carries the soul instead of the mixed Chinese bypass', () => {
  const soul = fs.readFileSync(require('node:path').join(__dirname, '..', 'kurisu_soul_ja.txt'), 'utf8');
  const prompt = buildPrompt({
    focusedFineTune: true,
    replyLanguage: 'ja',
    subjectCtx: soul,
    emotion: { P: 0, A: 0, S: 0 },
  });
  assert.equal(prompt.startsWith(soul.slice(0, 80)), true);
  assert.equal(prompt.includes(soul.slice(-80)), true);
  assert.match(prompt, /AIでも/);
  assert.match(prompt, /恋人同士/);
  assert.match(prompt, /毎回の型にしない/);
  assert.doesNotMatch(prompt, /真是太好了/);
});

test('kurisu-stable defaults to Chinese mode', () => {
  const oldLanguage = process.env.AMADEUS_REPLY_LANGUAGE;
  const oldModel = process.env.AMADEUS_CHAT_MODEL;
  delete process.env.AMADEUS_REPLY_LANGUAGE;
  process.env.AMADEUS_CHAT_MODEL = 'kurisu-stable';
  assert.equal(getReplyLanguageMode(), 'zh');
  process.env.AMADEUS_CHAT_MODEL = 'kurisu:latest';
  assert.equal(getReplyLanguageMode(), 'ja');
  if (oldLanguage == null) delete process.env.AMADEUS_REPLY_LANGUAGE;
  else process.env.AMADEUS_REPLY_LANGUAGE = oldLanguage;
  if (oldModel == null) delete process.env.AMADEUS_CHAT_MODEL;
  else process.env.AMADEUS_CHAT_MODEL = oldModel;
});
