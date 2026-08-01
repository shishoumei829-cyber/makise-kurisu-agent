'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const {
  extractDeferredSpeak,
  looksLikeFutureSpeakRequest,
} = require('../cognitive/deferredSpeak');

test('wake request is deferred speak', () => {
  const now = Date.parse('2026-07-26T01:00:00+08:00');
  const hit = extractDeferredSpeak('早上8点能叫我起床吗', now);
  assert.ok(hit);
  assert.ok(hit.dueAt > now);
  assert.match(hit.content, /起床|叫/);
});

test('check-in later is deferred speak without alarm words', () => {
  const now = Date.parse('2026-07-26T12:00:00+08:00');
  const hit = extractDeferredSpeak('下午3点记得回来找我说话', now);
  assert.ok(hit);
  assert.ok(looksLikeFutureSpeakRequest('下午3点记得回来找我说话'));
});

test('mere schedule fact is not deferred speak', () => {
  assert.equal(extractDeferredSpeak('明天早上8点有课'), null);
});
