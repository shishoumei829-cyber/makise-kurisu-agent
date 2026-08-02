'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const {
  detectReplyingToHerThread,
  buildProactiveReplyFocus,
  replyLooksLikeAutonomyFabrication,
} = require('../cognitive/turnContinuity');
const { stripOrphanClosingSentence } = require('../cognitive/replyAlign');

test('detectReplyingToHerThread merges consecutive assistant bubbles', () => {
  const dialogue = [
    { role: 'assistant', content: '第一句主动。' },
    { role: 'assistant', content: '第二句补充。' },
    { role: 'user', content: '我在呢' },
  ];
  const out = detectReplyingToHerThread(dialogue);
  assert.equal(out.active, true);
  assert.match(out.anchor, /第一句主动/);
  assert.match(out.anchor, /第二句补充/);
});

test('buildProactiveReplyFocus notes multi-bubble anchor', () => {
  const block = buildProactiveReplyFocus('嗯', '你好。在干嘛？');
  assert.match(block, /连发了两句/);
});

test('stripOrphanClosingSentence keeps short two-sentence IM replies', () => {
  const raw = '行啊。那你说说看。';
  const out = stripOrphanClosingSentence(raw, '随便', '');
  assert.equal(out, raw);
});

test('autonomy fabrication catches invented phone-call presence', () => {
  assert.equal(
    replyLooksLikeAutonomyFabrication(
      '好无聊好无聊',
      '你们明明还没有打电话过来，还是刚刚才注意到吧？',
      { alreadyTalking: true },
    ),
    true,
  );
  assert.equal(
    replyLooksLikeAutonomyFabrication('好无聊', '无聊就找点事做，别光打字。'),
    false,
  );
});

test('autonomy fabrication catches generic care unrelated to the current topic', () => {
  assert.equal(
    replyLooksLikeAutonomyFabrication(
      '我今天终于把报告写完了',
      '……疲れているなら、そう言えばいい。決まり文句で慰めるつもりはないわ。',
      { alreadyTalking: true },
    ),
    true,
  );
  assert.equal(
    replyLooksLikeAutonomyFabrication('今天真的累坏了', '疲れているなら、そう言えばいい。', { alreadyTalking: true }),
    false,
  );
});
