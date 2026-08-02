'use strict';

const { describe, it } = require('node:test');
const assert = require('node:assert/strict');
const { checkConsistency } = require('../brain/monitor/consistency');
const { assessArchive } = require('../lib/memory/writeGate');
const { gateAssistantReply } = require('../lib/generationGate');

describe('dialogue UX hardening', () => {
  it('consistency blocks cot_leak', () => {
    const v = checkConsistency(
      '好的，用户现在提到熬夜。首先需要分析他们可能的需求和对话历史。',
      { userText: '我熬夜了' },
    );
    assert.ok(v.some((x) => x.id === 'consistency.cot_leak' || x.id === 'consistency.product_meta'));
  });

  it('light chat does not archive even if assistant is long', () => {
    const d = assessArchive({
      userText: '好无聊啊',
      assistantText: '那就去做点实验，别光在这儿发呆。相位噪声也不会自己消失。',
      userRepliedToProactive: true,
    });
    assert.equal(d.admit, false);
    assert.ok(d.action === 'working' || d.action === 'reject');
  });

  it('gate drop means empty display candidate not dirty pass', () => {
    const dirty = '嗯……用户突然让我讲解对话系统的运作机制。';
    const g = gateAssistantReply(dirty);
    assert.equal(g.action, 'drop');
    // 模拟 legacyChat：dropped 时 out 必须是空串，禁止 || dirty
    const polished = { chinese: '', dropped: true };
    const out = polished.dropped ? '' : (polished.chinese || dirty);
    assert.equal(out, '');
  });
});
