'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const {
  detectReplyingToHerThread,
  buildProactiveReplyFocus,
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
