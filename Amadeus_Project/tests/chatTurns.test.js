'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');

const {
  parseIncomingChat,
  capDialogue,
  buildOllamaMessages,
  estimateMessageChars,
  resolvePromptCharBudget,
  calculatePromptCharBudget,
  isFastConversationTurn,
} = require('../cognitive/chatTurns');
const { stripRoleplayActions, hasRoleplayActions } = require('../cognitive/replyAlign');

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

test('resolvePromptCharBudget should cap chars to num_ctx budget', () => {
  const cap = resolvePromptCharBudget({ numCtx: 2048, maxTok: 384, envMaxChars: 6000 });
  assert.ok(cap <= 6000);
  assert.ok(cap < 3500);
  const wide = resolvePromptCharBudget({ numCtx: 8192, maxTok: 384, envMaxChars: 6000 });
  assert.equal(wide, 6000);
});

test('estimateMessageChars should return non-negative integer-like value', () => {
  const n = estimateMessageChars([
    { role: 'system', content: 'abc' },
    { role: 'user', content: '12345' },
  ]);
  assert.equal(n, 8);
});

test('roleplay cleanup preserves formulas and removes actual actions', () => {
  assert.equal(stripRoleplayActions('x+(x+1)=1.10（叹气）'), 'x+(x+1)=1.10');
  assert.equal(hasRoleplayActions('x+(x+1)=1.10'), false);
  assert.equal(hasRoleplayActions('（轻笑）知道了'), true);
});

test('calculatePromptCharBudget should reserve room for output tokens', () => {
  const small = calculatePromptCharBudget(2048, 384, 8000);
  const large = calculatePromptCharBudget(8192, 384, 8000);
  assert.ok(small < 2500);
  assert.ok(large > small);
  assert.ok(calculatePromptCharBudget(2048, 384, 1200) <= 1200);
});

test('fast conversation path only accepts small social turns', () => {
  assert.equal(isFastConversationTurn('谢谢你。'), true);
  assert.equal(isFastConversationTurn('こんにちは'), true);
  assert.equal(isFastConversationTurn('帮我创建一个明天的提醒'), false);
  assert.equal(isFastConversationTurn('为什么这个方案会失败？'), false);
  assert.equal(isFastConversationTurn('好吧', { useLongTermMemory: true }), false);
});
