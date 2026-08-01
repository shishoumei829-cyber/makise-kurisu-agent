'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const os = require('os');
const path = require('path');

const { createDefaultAxioms, principleLines } = require('../brain/axioms/fromSoul');
const { judgeRequest } = require('../lib/agency/judge');
const { extractHerPromise } = require('../lib/agency/promiseExtract');
const { IntentionStore } = require('../lib/agency/intentionStore');
const { buildCapabilitySnapshot, formatSnapshotForPrompt } = require('../lib/agency/capabilityRegistry');
const { AgencyLoop } = require('../lib/agency/loop');
const { ButlerKernel } = require('../lib/butler/kernel');

function tempDir(prefix) {
  return fs.mkdtempSync(path.join(os.tmpdir(), prefix));
}

test('axioms list native effectors and narrow forbidden domain', () => {
  const ax = createDefaultAxioms();
  assert.ok(ax.effector_domain.native.includes('speech'));
  assert.ok(ax.effector_domain.native.includes('time_awareness'));
  assert.ok(ax.effector_domain.native.includes('commitment'));
  assert.ok(ax.effector_forbidden.includes('user_body'));
  assert.ok(!ax.effector_forbidden.includes('user_device'));
  const lines = principleLines(ax);
  assert.ok(lines.some((l) => /时间感|声音|承诺/.test(l)));
  assert.ok(lines.some((l) => /禁止.*AI/.test(l)));
});

test('judge: wake-up is native at_time, not AI refuse', () => {
  const now = Date.parse('2026-07-26T01:00:00+08:00');
  const j = judgeRequest('早上8点能叫我起床吗', { now });
  assert.equal(j.verdict, 'native');
  assert.equal(j.trigger.kind, 'at_time');
  assert.ok(j.trigger.dueAt > now);
  assert.ok(j.effectors.includes('speech.deferred'));
});

test('judge: open notepad is augmented immediate', () => {
  const j = judgeRequest('打开记事本');
  assert.equal(j.verdict, 'augmented');
  assert.equal(j.trigger.kind, 'immediate');
  assert.ok(j.plan?.some((s) => s.capabilityId === 'app.launch'));
});

test('judge: buy coffee is forbidden with alternative', () => {
  const j = judgeRequest('帮我去买杯咖啡');
  assert.equal(j.verdict, 'forbidden');
  assert.match(j.alternative || '', /记|说|开口|提醒/);
});

test('judge: bare schedule fact is not agency', () => {
  const j = judgeRequest('明天早上8点有课');
  assert.equal(j.verdict, 'none');
});

test('promise extract binds her yes to user wake request', () => {
  const now = Date.parse('2026-07-26T01:00:00+08:00');
  const hit = extractHerPromise({
    userText: '早上8点叫我起床',
    replyText: '行，到点叫你。',
    now,
  });
  assert.ok(hit);
  assert.equal(hit.trigger.kind, 'at_time');
  assert.ok(hit.trigger.dueAt > now);
});

test('intention store persists commit and due', () => {
  const dir = tempDir('amadeus-intention-');
  const store = new IntentionStore(dir);
  const dueAt = Date.now() + 60_000;
  const item = store.commit({
    goal: '叫他起床',
    trigger: { kind: 'at_time', dueAt },
    effectors: ['speech.deferred', 'time.schedule'],
    source: 'user_request',
  });
  assert.equal(item.status, 'committed');
  const again = new IntentionStore(dir);
  assert.equal(again.get(item.id)?.goal, '叫他起床');
  assert.equal(again.due(dueAt + 1).length, 1);
});

test('capability snapshot includes native even when tools off', () => {
  const snap = buildCapabilitySnapshot({
    toolCaps: [{ id: 'app.launch', name: '启动应用', available: false, risk: 'low' }],
  });
  assert.ok(snap.native.some((c) => c.id === 'speech.deferred' && c.available));
  assert.ok(snap.augmented.some((c) => c.id === 'app.launch' && c.available === false));
  const text = formatSnapshotForPrompt(snap);
  assert.match(text, /原生/);
  assert.match(text, /speech\.deferred|到点开口/);
});

test('agency loop: wake commits; notepad proposes task; coffee refuses', async () => {
  const dir = tempDir('amadeus-agency-');
  const kernel = new ButlerKernel({ dataDir: dir });
  const loop = new AgencyLoop({ dataDir: dir, butlerKernel: kernel });

  const wake = loop.perceiveUser('早上8点叫我起床', { now: Date.parse('2026-07-26T01:00:00+08:00') });
  assert.equal(wake.judgment.verdict, 'native');
  assert.ok(wake.intention);
  assert.equal(wake.intention.status, 'committed');

  const note = loop.perceiveUser('打开记事本');
  assert.equal(note.judgment.verdict, 'augmented');
  assert.ok(note.task);

  const coffee = loop.perceiveUser('帮我去买杯咖啡');
  assert.equal(coffee.judgment.verdict, 'forbidden');
  assert.equal(coffee.intention, null);

  const promise = loop.perceiveHerReply({
    userText: '下午3点回来找我说话',
    replyText: '好，到点找你。',
    now: Date.parse('2026-07-26T12:00:00+08:00'),
  });
  assert.ok(promise.intention);
});

test('kernel observeUserRequest routes through agency for wake and notepad', () => {
  const dir = tempDir('amadeus-kernel-agency-');
  const kernel = new ButlerKernel({ dataDir: dir });
  const wake = kernel.observeUserRequest({
    text: '早上8点叫我起床',
    turnId: 't1',
    now: Date.parse('2026-07-26T01:00:00+08:00'),
  });
  assert.ok(['deferred_speak_proposed', 'agency_committed', 'intention_committed'].includes(wake.action)
    || wake.intention
    || wake.deferred);
  assert.ok(wake.intention || wake.task);

  const note = kernel.observeUserRequest({ text: '打开记事本', turnId: 't2' });
  assert.ok(note.task || note.created);
});
