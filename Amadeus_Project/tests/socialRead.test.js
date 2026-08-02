'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readSocialField, shouldSpeakNow, nextSenseMs } = require('../cognitive/socialRead');

test('comfortable silence raises the speak threshold', () => {
  const social = readSocialField({
    facePresent: true,
    faceMs: 60000,
    quietMs: 90000,
    alreadyTalking: true,
    lastUserText: '嗯，这个实验挺有意思的',
    relScore: 0.6,
    pad: { P: 0.2, S: 0.5 },
  });
  assert.ok(social.comfortableSilence > 0.4);
  assert.equal(social.floorOpen, true);
  const gate = shouldSpeakNow(0.35, social, { relScore: 0.6 });
  assert.equal(gate.ok, false);
  assert.equal(gate.reason, 'urge_not_ripe');
});

test('cold short reply creates tension and can let a ripe urge through', () => {
  const social = readSocialField({
    facePresent: true,
    faceMs: 40000,
    quietMs: 25000,
    alreadyTalking: true,
    lastUserText: '嗯',
    relScore: 0.4,
    pad: { P: 0, S: 0.3 },
  });
  assert.ok(social.tension > 0.4);
  assert.ok(social.comfortableSilence < 0.35);
  const gate = shouldSpeakNow(0.55, social, { relScore: 0.4 });
  assert.equal(gate.ok, true);
});

test('thinking or TTS closes the floor', () => {
  const social = readSocialField({
    facePresent: true,
    isThinking: true,
    quietMs: 20000,
  });
  assert.equal(social.floorOpen, false);
  assert.equal(shouldSpeakNow(0.9, social).ok, false);
});

test('next sense slows down when quiet together', () => {
  const quiet = nextSenseMs({ facePresent: true, comfortableSilence: 0.75 }, 0.2);
  const eager = nextSenseMs({ facePresent: true, comfortableSilence: 0.1 }, 0.7);
  assert.ok(quiet > eager);
});
