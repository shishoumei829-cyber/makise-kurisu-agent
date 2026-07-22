'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');

const {
  hasModel,
  summarizeHealth,
  buildUserHints,
  runStartupChecks,
} = require('../lib/startupCheck');

test('hasModel should match exact and base names', () => {
  const models = ['kurisu-v4-candidate:latest', 'nomic-embed-text:latest'];
  assert.equal(hasModel(models, 'kurisu-v4-candidate'), true);
  assert.equal(hasModel(models, 'kurisu-v4-candidate:latest'), true);
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

test('runStartupChecks should expose optional vision capability without blocking chat', async () => {
  const fakeFetch = async (url) => {
    if (String(url).endsWith('/api/tags')) {
      return { ok: true, status: 200, async json() { return { models: [{ name: 'kurisu:latest' }, { name: 'llama3.2-vision:latest' }] }; } };
    }
    throw new Error('optional service offline');
  };
  const report = await runStartupChecks({
    fetchFn: fakeFetch,
    ragIndexed: false,
    visionModels: ['llama3.2-vision:latest'],
  });
  assert.equal(report.ready, true);
  assert.equal(report.visionModel, 'llama3.2-vision:latest');
  assert.equal(report.capabilities.visionUnderstanding, true);
  assert.equal(report.capabilities.tts, false);
  assert.equal(report.capabilities.proactiveDialogue, true);
});
