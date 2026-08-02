'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const {
  detectContentBand,
  applyAfterglow,
  EmotionalBandwidthEngine,
  BANDS,
} = require('../cognitive/emotionalBandwidth');

test('priority: confession is shocked not tsundere', () => {
  assert.equal(detectContentBand('我喜欢你', { relScore: 0.8, pad: { S: 0.7 } }), BANDS.SHOCKED);
});

test('priority: distress is tender over soft', () => {
  assert.equal(
    detectContentBand('我好难受，陪我一下', { relScore: 0.8, pad: { S: 0.7 } }),
    BANDS.TENDER,
  );
});

test('nickname is tsundere', () => {
  assert.equal(detectContentBand('克里斯蒂娜'), BANDS.TSUNDERE);
});

test('PAD alone does not force shocked', () => {
  assert.notEqual(
    detectContentBand('今天天气还行', { pad: { A: 0.9, P: 0, S: 0.2 }, relScore: 0.2 }),
    BANDS.SHOCKED,
  );
});

test('after shock, casual content becomes aftermath not playful', () => {
  const band = applyAfterglow(BANDS.PLAYFUL, {
    kind: BANDS.SHOCKED,
    until: Date.now() + 60000,
  });
  assert.equal(band, BANDS.AFTERMATH);
});

test('after shock, tender content still wins', () => {
  const band = applyAfterglow(BANDS.TENDER, {
    kind: BANDS.SHOCKED,
    until: Date.now() + 60000,
  });
  assert.equal(band, BANDS.TENDER);
});

test('engine resolves one band and registers afterglow', () => {
  const e = new EmotionalBandwidthEngine();
  const r = e.resolve({
    userText: '什么？！世界线真的变了',
    relScore: 0.6,
    pad: { S: 0.5, A: 0.2, P: 0 },
  });
  assert.equal(r.band, BANDS.SHOCKED);
  assert.match(r.block, /情感带宽/);
  e.registerSpoken(r.band, 100000);
  const next = e.resolve({
    now: 100000 + 5000,
    userText: '在吗',
    relScore: 0.6,
    pad: { S: 0.5 },
  });
  assert.equal(next.band, BANDS.AFTERMATH);
});
