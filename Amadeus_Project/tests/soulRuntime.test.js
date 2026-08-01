'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { SoulRuntime } = require('../lib/soulRuntime');

function fresh() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'amadeus-soul-'));
  return {
    dir,
    runtime: new SoulRuntime({ statePath: path.join(dir, 'subject_state.json') }),
  };
}

test('relationship is a durable lover premise, not a keyword mode', () => {
  const { runtime } = fresh();
  assert.equal(runtime.state.relationship.type, 'lovers');
  assert.equal(runtime.state.relationship.partnerName, '冈部伦太郎');
  assert.match(runtime.promptBlock('今天吃什么'), /已经是恋人/);
  assert.match(runtime.promptBlock('今天吃什么'), /不存在另一个/);
});

test('short meaningful life events are written and recalled', () => {
  const { runtime } = fresh();
  runtime.observeUserTurn('准备去健身房了');
  runtime.observeUserTurn('健身回来咯，累死了');
  runtime.observeUserTurn('流鼻血了');
  assert.equal(runtime.state.memories.length, 3);
  assert.ok(runtime.recall('健身累不累', 3).some((item) => /健身回来/.test(item.text)));
  assert.ok(runtime.recall('身体不舒服', 5).some((item) => /鼻血/.test(item.text)));
});

test('unrelated relationship utterances are not injected into every turn', () => {
  const { runtime } = fresh();
  runtime.observeUserTurn('你最近怎么都不主动找我？');
  const recalled = runtime.recall('我健身回来了，腿很累', 5);
  assert.doesNotMatch(recalled.map((item) => item.text).join('\n'), /不主动找我/);
});

test('memory recall ignores filler-word overlap between unrelated proactive topics', () => {
  const { runtime } = fresh();
  runtime.observeUserTurn('我今天有点累，但还不想听客服式安慰');
  const recalled = runtime.recall('我今天终于把报告写完了', 5);
  assert.doesNotMatch(recalled.map((item) => item.text).join('\n'), /累|客服/);
});

test('a shared-past question is stored as an inquiry, never as an event fact', () => {
  const { runtime } = fresh();
  const memory = runtime.observeUserTurn('我们昨天是不是一起去喝酒了？');
  assert.equal(memory.type, 'inquiry');
  assert.match(memory.text, /只是提问，不代表其中的事件发生过/);
});

test('recall questions do not outrank the fact they are asking about', () => {
  const { runtime } = fresh();
  runtime.observeUserTurn('我健身回来咯，累死了');
  runtime.observeUserTurn('我刚才干嘛回来？');
  const recalled = runtime.recall('我刚才干嘛回来？', 5);
  assert.ok(recalled.some((item) => /健身回来/.test(item.text)));
  assert.ok(recalled.every((item) => item.type !== 'inquiry'));
});

test('acknowledgements do not become durable memory', () => {
  const { runtime } = fresh();
  runtime.observeUserTurn('嗯');
  runtime.observeUserTurn('知道了');
  assert.equal(runtime.state.memories.length, 0);
});

test('an expressed thought is consumed and cannot repeat', () => {
  const { runtime } = fresh();
  const thought = runtime.addThought({
    content: '我还是想知道他健身后有没有好好休息',
    tension: 0.8,
    delayMs: 0,
  });
  thought.earliestSpeakAt = 0;
  const decision = runtime.selectInitiative({ lastSpokenAt: 0, now: Date.now() });
  assert.equal(decision.shouldSpeak, true);
  runtime.markThoughtExpressed(decision.thoughtId, '累的话就先坐一会儿。');
  const again = runtime.selectInitiative({ lastSpokenAt: 0, now: Date.now() });
  assert.equal(again.shouldSpeak, false);
});

test('reflection updates affect, memory and an unfinished thought', async () => {
  const { runtime } = fresh();
  runtime.observeUserTurn('我健身回来咯，累死了');
  await runtime.reflectAfterTurn({
    userText: '我健身回来咯，累死了',
    assistantText: '先喝水。别一回来就瘫着。',
    ollamaChat: async () => JSON.stringify({
      emotion: '在意',
      valence: 0.3,
      arousal: 0.4,
      vulnerability: 0.5,
      appraisal: '他累成这样，我有点担心。',
      keepMemory: false,
      memory: '',
      memoryType: 'wellbeing',
      openThought: '等一会儿想确认他有没有补水',
      desire: '确认他在照顾身体',
      tension: 0.72,
    }),
  });
  assert.equal(runtime.state.affect.emotion, '在意');
  assert.ok(runtime.state.memories.some((item) => /健身回来/.test(item.text) && item.source === 'user'));
  assert.ok(runtime.state.thoughts.some((item) => /补水/.test(item.content)));
});
