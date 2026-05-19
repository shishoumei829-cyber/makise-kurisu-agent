'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');

const {
  hasModel,
  summarizeHealth,
  buildUserHints,
} = require('../lib/startupCheck');

test('hasModel should match exact and base names', () => {
  const models = ['kurisu:latest', 'nomic-embed-text:latest'];
  assert.equal(hasModel(models, 'kurisu:latest'), true);
  assert.equal(hasModel(models, 'kurisu'), true);
  assert.equal(hasModel(models, 'llama3.2'), false);
});

test('summarizeHealth should mark ready when required checks pass', () => {
  const checks = [
    { id: 'ollama', required: true, ok: true, message: 'ok' },
    { id: 'chat_model', required: true, ok: true, message: 'ok' },
    { id: 'tts', required: false, ok: false, message: 'missing' },
  ];
  const out = summarizeHealth(checks);
  assert.equal(out.ready, true);
  assert.equal(out.warnings.length, 1);
});

test('buildUserHints should include ollama guidance for blockers', () => {
  const summary = summarizeHealth([
    { id: 'ollama', required: true, ok: false, message: 'down' },
  ]);
  const hints = buildUserHints(summary);
  assert.ok(hints.some((h) => /Ollama/i.test(h)));
});
