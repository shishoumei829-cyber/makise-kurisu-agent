'use strict';

const { describe, it } = require('node:test');
const assert = require('node:assert/strict');
const { extractSpeechCandidate, salvageAssistantReply } = require('../lib/salvageReply');
const { gateAssistantReply } = require('../lib/generationGate');

describe('salvage extractSpeechCandidate', () => {
  it('pulls spoken tail from cot-heavy draft', () => {
    const dirty = [
      '好的，用户现在提到熬夜。首先需要分析需求和对话历史。',
      '凌晨三点才睡，身体会垮的。',
    ].join('\n');
    const got = extractSpeechCandidate(dirty);
    assert.ok(got.includes('凌晨三点') || got.includes('身体'));
    assert.equal(gateAssistantReply(got).action !== 'drop', true);
  });

  it('drops pure identity collapse drafts', () => {
    assert.equal(extractSpeechCandidate('作为有逻辑的人工智能，这种信息应该记住。'), '');
  });

  it('salvage uses extract before llm', async () => {
    const out = await salvageAssistantReply({}, {
      userText: '我喜欢喝什么',
      previousDraft: '用户突然提问。\n胡椒博士啊，别装忘。',
    });
    assert.match(out, /胡椒博士/);
  });
});

it('style-criticism salvage never reuses a fixed self-defence', async () => {
  let system = '';
  const out = await salvageAssistantReply({
    ollamaChatOnce: async (_model, messages) => {
      system = messages[0].content;
      return '我不想只是给你一个千篇一律的回答。';
    },
  }, {
    model: 'test',
    userText: '你又在说固定回复，很假。',
    previousDraft: '我不会那些客服式的套话敷衍你。',
  });
  assert.equal(out, '');
  assert.doesNotMatch(system, /客服|模板|AI/);
  assert.match(system, /不要解释、保证、辩护/);
});

it('unsafe observation discards the original draft and regenerates from user facts', async () => {
  let system = '';
  const out = await salvageAssistantReply({
    ollamaChatOnce: async (_model, messages) => {
      system = messages[0].content;
      return '今天烦什么，直接说。';
    },
  }, {
    model: 'test',
    userText: '今天有点烦。',
    reasons: ['invented_user_state'],
    previousDraft: '你脸色看起来很差，肯定累坏了。',
  });
  assert.equal(out, '今天烦什么，直接说。');
  assert.match(system, /不能沿用/);
  assert.match(system, /不描述他的脸色/);
});
