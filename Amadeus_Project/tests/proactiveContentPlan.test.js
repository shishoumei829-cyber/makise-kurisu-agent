'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const {
  buildProactiveContentPlan,
  validateProactiveContent,
} = require('../lib/proactiveContentPlan');

test('proactive content needs a concrete anchor or persistent thought', () => {
  const plan = buildProactiveContentPlan({ action: 'poke', contextFresh: false });
  assert.equal(plan.shouldGenerate, false);
});

test('tease plan stays tied to the concrete latest turn and rejects a forced question', () => {
  const plan = buildProactiveContentPlan({
    action: 'tease',
    contextFresh: true,
    anchor: 'レポートを書き終えた',
  });
  assert.equal(plan.shouldGenerate, true);
  assert.equal(plan.allowQuestion, false);
  assert.equal(validateProactiveContent('疲れているなら、そう言えばいい。', plan).ok, false);
  assert.equal(validateProactiveContent('レポート、終わったの？', plan).ok, false);
  assert.equal(validateProactiveContent('レポートは終わったのね。今回は締切に追われずに済んだわけ。', plan).ok, true);
});

test('persistent thought can be a content core without a fresh user turn', () => {
  const plan = buildProactiveContentPlan({
    action: 'poke',
    contextFresh: false,
    thought: '昨日の話で残った、彼の進路への考えを聞きたい',
  });
  assert.equal(plan.shouldGenerate, true);
  assert.equal(plan.origin, 'persistent_thought');
});

test('specific active content may use a narrow natural anaphora instead of noun repetition', () => {
  const plan = buildProactiveContentPlan({
    action: 'tease',
    contextFresh: true,
    anchor: '今日はついにレポートを終わらせた',
  });
  const checked = validateProactiveContent('ふん、やっとあの山場から這い出したな。', plan);
  assert.equal(checked.ok, true);
  assert.equal(checked.reason, 'anaphoric_subject_continuation');
});

test('a question in a later bubble is still rejected when the intent forbids asking', () => {
  const plan = buildProactiveContentPlan({ action: 'tease', contextFresh: true, anchor: '报告终于写完了' });
  assert.equal(validateProactiveContent('总算写完了。现在要我夸你？', plan).reason, 'forced_question');
});
