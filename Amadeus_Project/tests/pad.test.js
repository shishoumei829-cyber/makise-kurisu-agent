'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');

const {
  PAD_BASE,
  updatePAD,
  inferMainEventFromInput,
  clamp,
} = require('../cognitive/pad');

test('clamp should bound value within range', () => {
  assert.equal(clamp(2, -1, 1), 1);
  assert.equal(clamp(-2, -1, 1), -1);
  assert.equal(clamp(0.5, -1, 1), 0.5);
});

test('updatePAD should keep values in expected ranges', () => {
  const next = updatePAD(
    { P: 0, A: 0, D: 0, S: 0.5 },
    { P: 0.2, A: -0.1, D: 0.4, S: 0.1 },
    0.7,
  );
  assert.ok(next.P >= -1 && next.P <= 1);
  assert.ok(next.A >= -1 && next.A <= 1);
  assert.ok(next.D >= -1 && next.D <= 1);
  assert.ok(next.S >= 0 && next.S <= 1);
});

test('inferMainEventFromInput should fallback to neutral on unknown text', () => {
  const event = inferMainEventFromInput('今天天气还行', { S: 0.1 });
  assert.equal(event.type, 'neutral');
  assert.ok(event.delta.S > 0);
});

test('inferMainEventFromInput should detect intimate signal', () => {
  const event = inferMainEventFromInput('谢谢你，一直很温柔', { S: PAD_BASE.S });
  assert.equal(event.type, 'intimate');
  assert.ok(event.delta.P > 0);
});
