'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const os = require('os');
const path = require('path');

const { assessArchive, classifyRoom, compressDeterministic } = require('../lib/memory/writeGate');
const { MemoryPalaceStore } = require('../lib/memory/palaceStore');
const { MemoryAdmissionPolicy } = require('../lib/memoryAdmission');

function tmpDir() {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'amadeus-palace-'));
}

test('casual chatter does not archive to palace', () => {
  const d = assessArchive({
    userText: '哼',
    assistantText: '怎么了？',
  });
  assert.equal(d.admit, false);
  assert.equal(d.allowPalace, false);
  assert.equal(d.reason, 'short_casual');
});

test('substantive turn archives to a room', () => {
  const d = assessArchive({
    userText: '我最近在写一篇关于量子纠缠的论文，进度卡住了。',
    assistantText: '哪一段卡住了？公式还是实验设计？',
  });
  assert.equal(d.admit, true);
  assert.equal(d.allowPalace, true);
  assert.equal(d.room, 'lab');
});

test('unanswered proactive cannot archive', () => {
  const d = assessArchive({
    userText: '',
    assistantText: '你还在吗？',
    proactive: true,
    userRepliedToProactive: false,
  });
  assert.equal(d.admit, false);
  assert.equal(d.reason, 'proactive_unanswered');
});

test('identity-collapse assistant blocks palace write', () => {
  const d = assessArchive({
    userText: '我是谁',
    assistantText: '你是眼前这位同学吧？',
  });
  assert.equal(d.admit, false);
  assert.ok(['identity_collapse', 'poison_or_internal'].includes(d.reason));
});

test('explicit remember allows profile flag', () => {
  const admission = new MemoryAdmissionPolicy().assessUserText('请记住我喜欢胡椒博士', { source: 'user' });
  const d = assessArchive({
    userText: '请记住我喜欢胡椒博士',
    assistantText: '知道了，胡椒博士。',
    userAdmission: admission,
  });
  assert.equal(d.admit, true);
  assert.equal(d.allowProfile, true);
  assert.ok(d.room === 'hall' || d.room === 'cafe' || d.room === 'forbidden');
});

test('derived-style meta assistant is rejected', () => {
  const d = assessArchive({
    userText: '微调后语言更差了',
    assistantText: '我们这边已经好好调整了，下次对话模式可以柔和一些。',
  });
  assert.equal(d.admit, false);
});

test('palace store persists archive and navigate', () => {
  const dir = tmpDir();
  const palace = new MemoryPalaceStore(dir);
  const r = palace.archiveTurn({
    userText: '我今天好困，想早点睡。',
    assistantText: '那就去睡，别硬撑。',
  });
  assert.equal(r.ok, true);
  assert.equal(r.room, 'cafe');
  assert.ok(r.node.text.includes('→') || r.node.text.length > 2);

  const nav = palace.navigate('我困了想睡觉');
  assert.ok(nav.excerpt.includes('CAFÉ') || nav.excerpt.includes('cafe') || nav.excerpt.length > 0);
  assert.ok(palace.counts().total >= 1);

  const palace2 = new MemoryPalaceStore(dir);
  assert.ok(palace2.counts().total >= 1);
});

test('proactive buffer promotes only after user reply path', () => {
  const dir = tmpDir();
  const palace = new MemoryPalaceStore(dir);
  palace.bufferProactive('程序代码在舞蹈'); // will be rejected on archive by poison if used as assistant
  palace.bufferProactive('你还在忙吗？');
  assert.equal(palace.hasPendingProactive(), true);

  const blocked = palace.archiveTurn({
    userText: '',
    assistantText: '你还在忙吗？',
    proactive: true,
    userRepliedToProactive: false,
  });
  assert.equal(blocked.ok, false);

  const pending = palace.consumeProactiveOnUserReply();
  assert.ok(pending.length >= 1);

  const ok = palace.archiveTurn({
    userText: '还在，刚忙完。今天实验数据有点烦。',
    assistantText: '哪组数据不对？拿来我对一对。',
    userRepliedToProactive: true,
  });
  assert.equal(ok.ok, true);
});

test('classifyRoom and compressDeterministic basics', () => {
  assert.equal(classifyRoom('量子论文实验'), 'lab');
  assert.equal(classifyRoom('好困想睡觉'), 'cafe');
  const c = compressDeterministic('我喜欢胡椒博士', '记下了');
  assert.match(c, /胡椒|记下/);
});

test('clear empties palace', () => {
  const dir = tmpDir();
  const palace = new MemoryPalaceStore(dir);
  palace.archiveTurn({
    userText: '我决定明天开始整理实验室笔记。',
    assistantText: '行，我到时候问你进度。',
  });
  assert.ok(palace.counts().total >= 1);
  palace.clear();
  assert.equal(palace.counts().total, 0);
});
