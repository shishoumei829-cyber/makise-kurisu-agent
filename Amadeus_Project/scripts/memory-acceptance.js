'use strict';

/**
 * 记忆系统验收：宫殿中枢写入/拒绝/召回矩阵。
 * exit 0 = 达标
 */

const assert = require('node:assert/strict');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { MemoryPalaceStore } = require('../lib/memory/palaceStore');
const { MemoryAdmissionPolicy } = require('../lib/memoryAdmission');
const { assessArchive } = require('../lib/memory/writeGate');

const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'amadeus-mem-accept-'));
const palace = new MemoryPalaceStore(dir);
const admission = new MemoryAdmissionPolicy(dir);

const cases = [];

function check(name, fn) {
  try {
    fn();
    cases.push({ name, ok: true });
  } catch (e) {
    cases.push({ name, ok: false, error: e.message });
  }
}

check('闲聊不进宫殿', () => {
  const r = palace.archiveTurn({ userText: '嗯', assistantText: '怎么了' });
  assert.equal(r.ok, false);
  assert.equal(palace.counts().total, 0);
});

check('实质智识进 lab', () => {
  const r = palace.archiveTurn({
    userText: '量子通信实验的误差又飘了，帮我想想校准思路',
    assistantText: '先看参考时钟和光纤延迟，别急着怪探测器',
  });
  assert.equal(r.ok, true);
  assert.equal(r.room, 'lab');
});

check('生活情绪进 cafe', () => {
  const r = palace.archiveTurn({
    userText: '我今天好困，想早点睡',
    assistantText: '去睡，别硬撑',
  });
  assert.equal(r.ok, true);
  assert.equal(r.room, 'cafe');
});

check('明确记住开 profile 门', () => {
  const adm = admission.assessUserText('请记住我喜欢胡椒博士', { source: 'user' });
  const d = assessArchive({
    userText: '请记住我喜欢胡椒博士',
    assistantText: '记下了',
    userAdmission: adm,
  });
  assert.equal(d.allowPalace, true);
  assert.equal(d.allowProfile, true);
});

check('未接话主动不进宫殿', () => {
  palace.bufferProactive('你还在吗？');
  const r = palace.archiveTurn({
    userText: '',
    assistantText: '你还在吗？',
    proactive: true,
    userRepliedToProactive: false,
  });
  assert.equal(r.ok, false);
  assert.ok(palace.hasPendingProactive());
});

check('接话后可归档对话', () => {
  palace.consumeProactiveOnUserReply();
  const r = palace.archiveTurn({
    userText: '在，刚才在忙实验数据',
    assistantText: '哪组数据不对？',
    userRepliedToProactive: true,
  });
  assert.equal(r.ok, true);
});

check('身份崩坏拒绝', () => {
  const before = palace.counts().total;
  const r = palace.archiveTurn({
    userText: '我是谁',
    assistantText: '你是眼前这位同学吧',
  });
  assert.equal(r.ok, false);
  assert.equal(palace.counts().total, before);
});

check('召回能命中', () => {
  const nav = palace.navigate('量子实验误差');
  assert.ok(nav.excerpt.length > 0);
});

check('持久化重载', () => {
  const p2 = new MemoryPalaceStore(dir);
  assert.ok(p2.counts().total >= 2);
});

const failed = cases.filter((c) => !c.ok);
console.log(JSON.stringify({ dir, passed: cases.length - failed.length, failed: failed.length, cases }, null, 2));
if (failed.length) process.exit(1);
console.log('MEMORY_ACCEPTANCE_PASS');
