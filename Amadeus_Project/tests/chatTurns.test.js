'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');

const {
  parseIncomingChat,
  capDialogue,
  buildOllamaMessages,
  estimateMessageChars,
} = require('../cognitive/chatTurns');

test('parseIncomingChat should preserve dialogue roles and infer last user', () => {
  const body = {
    messages: [
      { role: 'system', content: 'sys A' },
      { role: 'user', content: '你好' },
      { role: 'assistant', content: '你好\nJP: hello' },
      { role: 'user', content: '你是谁' },
    ],
  };
  const out = parseIncomingChat(body);
  assert.equal(out.clientSystem, 'sys A');
  assert.equal(out.lastUser, '你是谁');
  assert.equal(out.dialogue.length, 3);
  assert.equal(out.dialogue[1].content, '你好');
});

test('capDialogue should limit message count', () => {
  const d = Array.from({ length: 8 }, (_, i) => ({
    role: i % 2 ? 'assistant' : 'user',
    content: `m${i}`,
  }));
  const out = capDialogue(d, 4);
  assert.equal(out.length, 4);
  assert.equal(out[0].content, 'm4');
});

test('buildOllamaMessages should keep system message and minimum turns', () => {
  const d = [
    { role: 'user', content: 'u1' },
    { role: 'assistant', content: 'a1' },
    { role: 'user', content: 'u2' },
    { role: 'assistant', content: 'a2' },
    { role: 'user', content: 'u3' },
    { role: 'assistant', content: 'a3' },
  ];
  const msgs = buildOllamaMessages('system', d, 20, 2);
  assert.equal(msgs[0].role, 'system');
  assert.ok(msgs.length >= 3);
});

test('estimateMessageChars should return non-negative integer-like value', () => {
  const n = estimateMessageChars([
    { role: 'system', content: 'abc' },
    { role: 'user', content: '12345' },
  ]);
  assert.equal(n, 8);
});
