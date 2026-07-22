'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');

const monitor = require('../brain/monitor');
const { localReviseDraft } = require('../brain/deliberation');
const { createDefaultAxioms } = require('../brain/axioms/fromSoul');
const { extractSpeechActs } = require('../brain/monitor/speechActs');
const probes = require('../brain/probes/capability.json');

const baseSelf = { axioms: createDefaultAxioms() };

function evalProbe(p) {
  const defaultLog = '冈部在实验室';
  const logExcerpt = p.world?.dialogue && Object.prototype.hasOwnProperty.call(p.world.dialogue, 'logExcerpt')
    ? p.world.dialogue.logExcerpt
    : defaultLog;
  const worldSnapshot = {
    partner: { isOkabe: true, name: '冈部', ...(p.world?.partner || {}) },
    dialogue: { logExcerpt, ...(p.world?.dialogue || {}) },
  };
  const result = monitor.check(p.draft, {
    userText: p.user,
    selfModel: baseSelf,
    worldSnapshot,
    oocOpts: {},
  });
  return result;
}

for (const p of probes) {
  test(`capability probe: ${p.id}`, () => {
    const result = evalProbe(p);
    if (p.expectPass === false) {
      assert.equal(result.pass, false, `expected fail: ${p.id}`);
      if (p.expectViolation) {
        assert.ok(
          result.violations.some((v) => v.id === p.expectViolation),
          `missing ${p.expectViolation} in ${result.violations.map((x) => x.id).join(',')}`,
        );
      }
    } else {
      assert.equal(result.pass, true, `expected pass: ${p.id} violations=${result.violations.map((v) => v.id).join(',')}`);
    }
  });
}

test('localReviseDraft: strips physical promise', () => {
  const draft = '好啊，顺路给你带一杯咖啡。';
  const mr = monitor.check(draft, { selfModel: baseSelf, userText: '带咖啡', worldSnapshot: {} });
  const revised = localReviseDraft(draft, mr);
  assert.ok(!/顺路给你带/.test(revised));
});

test('extractSpeechActs: physical_promise class', () => {
  const acts = extractSpeechActs('我去帮你买咖啡');
  assert.ok(acts.some((a) => a.type === 'physical_promise'));
});

test('monitor: empty draft fails', () => {
  const r = monitor.check('', { selfModel: baseSelf, worldSnapshot: {} });
  assert.equal(r.pass, false);
});
