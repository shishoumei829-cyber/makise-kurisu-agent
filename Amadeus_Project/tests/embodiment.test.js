'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');

const { EnvironmentSense } = require('../digital_life/embodiment/environment');
const { TimeSense } = require('../digital_life/embodiment/time_perception');
const { ExpressionMapper } = require('../digital_life/embodiment/expression_mapper');
const { SpritePolicy } = require('../digital_life/embodiment/sprite_policy');
const { EmbodimentSubsystem } = require('../digital_life/embodiment');

test('EnvironmentSense: infers busy focused state', () => {
  const env = new EnvironmentSense();
  const scene = env.understandScene('他在专注工作打字');
  assert.ok(scene.tags.includes('focused'));
  assert.equal(env.userState.availability, 'busy');
});

test('TimeSense: weekend and late night hints', () => {
  const t = new TimeSense();
  const ctx = t.understandTime(new Date('2026-06-28T02:30:00'));
  assert.equal(ctx.isWeekend, true);
  const hint = t.influenceBehavior(ctx);
  assert.ok(hint.includes('夜深') || hint.includes('周末'));
});

test('ExpressionMapper: maps PAD to preset and sprite', () => {
  const ex = new ExpressionMapper();
  const warm = ex.mapFromPad({ P: 0.5, A: 0.2, D: -0.2, S: 0.6 });
  assert.ok(warm.preset);
  assert.ok(warm.spriteIndex >= 0);
  const cold = ex.mapFromPad({ P: -0.5, A: -0.1, D: 0.4, S: 0.5 });
  assert.equal(cold.preset, 'cold');
});

test('SpritePolicy: blocks unapproved expressive assets', () => {
  const policy = new SpritePolicy({ minHoldMs: 0 });
  const out = policy.decide({
    userText: '你这样有点可爱，害羞了？',
    pad: { P: 0.3, A: 0.4, D: -0.2, S: 0.6 },
    preset: 'shy',
    now: 1000,
  });
  assert.equal(out.spriteId, 'neutral');
  assert.equal(out.hold, true);
  assert.equal(out.blockedSpriteId, 'shy_denial');
  assert.equal(out.changeReason, 'asset_not_approved');
});

test('ExpressionMapper: returns sprite policy metadata', () => {
  const ex = new ExpressionMapper();
  const out = ex.mapFromPad(
    { P: 0.2, A: 0.4, D: -0.4, S: 0.6 },
    { userText: '别露出那种害羞表情' }
  );
  assert.equal(out.assetApproved, true);
  assert.equal(out.spriteId, 'neutral');
  assert.equal(out.blockedSpriteId, 'shy_denial');
});

test('EmbodimentSubsystem: vision and conversation turn', () => {
  const emb = new EmbodimentSubsystem();
  const scene = emb.onVision('看起来疲惫在玩手机');
  assert.ok(scene.tags.includes('tired') || scene.tags.includes('distracted'));
  const out = emb.onConversationTurn({
    pad: { P: 0.3, A: 0.4, D: 0, S: 0.5 },
    userModel: { model: { patterns: { active_hours: { 22: 5 } } } },
    idleMs: 0,
  });
  assert.ok(out.expression.preset);
  assert.ok(out.timeLine.length > 0);
});
