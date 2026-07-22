'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const { ChatTurnRegistry } = require('../lib/chatTurnRegistry');

test('new chat turn cancels the previous turn in the same conversation', () => {
  const registry = new ChatTurnRegistry();
  const first = registry.begin('window-a', 'turn-1');
  const second = registry.begin('window-a', 'turn-2');
  assert.equal(first.signal.aborted, true);
  assert.equal(first.isCurrent(), false);
  assert.equal(second.isCurrent(), true);
});

test('different conversations do not cancel each other', () => {
  const registry = new ChatTurnRegistry();
  const first = registry.begin('window-a', 'turn-1');
  const second = registry.begin('window-b', 'turn-1');
  assert.equal(first.signal.aborted, false);
  assert.equal(second.signal.aborted, false);
});

test('completed turn cannot later own a dialogue commit', () => {
  const registry = new ChatTurnRegistry();
  const lease = registry.begin('window-a', 'turn-1');
  lease.complete();
  assert.equal(lease.isCurrent(), false);
});
