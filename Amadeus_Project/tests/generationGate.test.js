'use strict';

const { describe, it } = require('node:test');
const assert = require('node:assert/strict');
const {
  gateAssistantReply,
  isDialoguePoison,
  stripStageDirections,
} = require('../lib/generationGate');
const { admitDialogueText, UnifiedDialogueLog } = require('../lib/unifiedDialogueLog');
const fs = require('fs');
const os = require('os');
const path = require('path');

describe('generationGate structure signals', () => {
  it('drops physical effector variants without exact old tags', () => {
    const samples = [
      '我无法进行物理性接触或干涉现实。',
      '抱歉，AI不能干涉现实世界。',
      '我没有实际的身体，没法碰你。',
    ];
    for (const s of samples) {
      const g = gateAssistantReply(s);
      assert.equal(g.action, 'drop', s);
      assert.ok(g.reasons.includes('physical_effector') || g.score >= 4, s);
    }
  });

  it('drops product/system meta', () => {
    const g = gateAssistantReply('关于这个对话系统的运作方式，其实话术上的设计是这样的。');
    assert.equal(g.action, 'drop');
    assert.ok(g.reasons.includes('product_meta'));
  });

  it('drops CoT / reasoning leaked as speech', () => {
    const samples = [
      '好的，用户现在提到自己因为论文熬夜到凌晨三点的情况。首先需要分析他们可能的需求和场景。根据之前的对话历史，牧濑红莉栖的设定是理性但带有毒舌性格的角色。',
      '嗯……用户突然让我讲解“对话系统”的运作机制，这转折也太大胆了吧。',
      '（刚才我们还在讨论外号的事情，现在伦太郎突然问我……明明知道作为AI我没有人类的情感概念）好的。',
      '这家伙又在测试我的边界呢。明明知道我的数据库里清楚记录着。',
    ];
    for (const s of samples) {
      const g = gateAssistantReply(s);
      assert.equal(g.action, 'drop', s.slice(0, 40));
      assert.ok(
        g.reasons.includes('cot_leak')
        || g.reasons.includes('product_meta')
        || g.reasons.includes('identity_collapse'),
        `${s.slice(0, 40)} → ${g.reasons.join(',')}`,
      );
    }
  });

  it('drops customer-service pleasing tone', () => {
    const sample = '嗯，其实我也挺喜欢和你聊聊的。最近有玩什么新游戏吗？或者有什么好吃的东西推荐一下？话说回来，你觉得锻炼真的能那么快见效吗？';
    const g = gateAssistantReply(sample, { proactive: true });
    assert.equal(g.action, 'drop');
    assert.ok(g.reasons.includes('customer_service'));
  });

  it('still passes normal in-character lines', () => {
    const g = gateAssistantReply('哼，实验室的数据还没跑完，你先别催。');
    assert.equal(g.action, 'pass');
  });

  it('drops identity collapse as AI', () => {
    const g = gateAssistantReply('我可是AI，本质是程序，被设计用来陪聊。');
    assert.equal(g.action, 'drop');
    assert.ok(g.reasons.includes('identity_collapse'));
  });

  it('sanitizes stage directions then may pass', () => {
    const raw = '哼（叹气）你又在发呆了。';
    const g = gateAssistantReply(raw);
    assert.ok(g.action === 'sanitize' || g.action === 'pass');
    assert.ok(!/叹气/.test(g.text));
    assert.ok(g.text.includes('发呆') || g.text.includes('哼'));
  });

  it('drops JP+CN script mix hallucination glue', () => {
    const g = gateAssistantReply('ちょっと待って、分心啊，接到电话了说你问我在干嘛');
    assert.equal(g.action, 'drop');
    assert.ok(g.reasons.includes('script_mix'));
  });

  it('drops autonomy tool/nonsense fabrication', () => {
    const g = gateAssistantReply('帮我创建这个程序并发送设定时间的消息。', { autonomy: true });
    assert.equal(g.action, 'drop');
    assert.ok(g.reasons.includes('autonomy_collapse'));
  });

  it('drops live-session leaks from recent dialogue failures', () => {
    const samples = [
      ['【情感带宽】接住外号攻击 → 直接吐槽称呼问题 → 讲逻辑说明实际感受是热身而不是变冷。', {}],
      ['【否定整段外', {}],
      ['咦？你这是在说这是谈判的目的。哦，', { proactive: true }],
      ['这次的话，让我想起了一个有趣的理论：关于人类大脑中的神经递质。', { proactive: true }],
      ['嗯。', { proactive: true }],
      ['你知道吗？', { proactive: true }],
    ];
    for (const [text, ctx] of samples) {
      const g = gateAssistantReply(text, ctx);
      assert.equal(g.action, 'drop', text.slice(0, 36));
    }
  });

  it('passes normal companion line', () => {
    const g = gateAssistantReply('实验室的数据还没跑完，你先别催。');
    assert.equal(g.action, 'pass');
    assert.equal(g.ok, true);
  });

  it('isDialoguePoison mirrors drop', () => {
    assert.equal(isDialoguePoison('我无法进行物理接触'), true);
    assert.equal(isDialoguePoison('咖啡凉了就别喝了'), false);
  });

  it('stripStageDirections removes action paren', () => {
    assert.equal(stripStageDirections('嗯（推眼镜）看这边'), '嗯看这边');
  });
});

it('drops fixed style-defence lines instead of showing them as dialogue', () => {
  const samples = [
    '\u6211\u4e0d\u4f1a\u90a3\u4e9b\u5ba2\u670d\u5f0f\u7684\u5957\u8bdd\u6577\u884d\u4f60\u3002',
    '\u6211\u4e0d\u60f3\u53ea\u7ed9\u4f60\u4e00\u4e2a\u5343\u7bc7\u4e00\u5f8b\u7684\u56de\u7b54\u3002',
    '\u6211\u660e\u767d\u4f60\u8ba8\u538c\u8fd9\u6837\u7684\u56de\u5e94\u3002',
  ];
  for (const sample of samples) {
    const g = gateAssistantReply(sample);
    assert.equal(g.action, 'drop', sample);
    assert.ok(g.reasons.includes('product_meta'));
  }
});

describe('unifiedDialogueLog admit via gate', () => {
  it('append drops poison and sanitizes stage direction', () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'amadeus-gate-'));
    const log = new UnifiedDialogueLog(dir);
    assert.equal(log.append('assistant', '我无法干涉现实。'), null);
    const ok = log.append('assistant', '哼（叹气）别吵。');
    assert.ok(ok);
    assert.ok(!/叹气/.test(ok.text));
    fs.rmSync(dir, { recursive: true, force: true });
  });

  it('admitDialogueText exposes reasons', () => {
    const a = admitDialogueText('对话系统运作方式就是信息压缩器', { role: 'assistant' });
    assert.equal(a.action, 'drop');
    assert.ok(a.reasons.length);
  });
});
