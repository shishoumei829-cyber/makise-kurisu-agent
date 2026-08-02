'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const { isRecentAssistantDuplicate } = require('../lib/replyContinuity');

test('rejects an exact assistant reply repeated across adjacent user turns', () => {
  const entries = [
    { role: 'assistant', text: '哎呀，怎么了？是不是有什么麻烦的事情吗？' },
  ];
  assert.equal(isRecentAssistantDuplicate('哎呀，怎么了？是不是有什么麻烦的事情吗？', entries), true);
  assert.equal(isRecentAssistantDuplicate('无聊啊。那就先陪我说点具体的。', entries), false);
});
