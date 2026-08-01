'use strict';

const { describe, it } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { UnifiedDialogueLog, needsConversationRecall } = require('../lib/unifiedDialogueLog');
const { buildRecallFactAnchor, needsFactRecall } = require('../lib/recallFactAnchor');
const { MemoryPalaceStore } = require('../lib/memory/palaceStore');

describe('recall fact anchor', () => {
  it('detects preference recall questions', () => {
    assert.equal(needsFactRecall('我喜欢喝什么来着？'), true);
    assert.equal(needsConversationRecall('我喜欢喝什么来着？'), true);
    assert.equal(needsFactRecall('实验室数据跑完了吗'), false);
  });

  it('pins user remember lines into anchor block', () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'amadeus-recall-'));
    const log = new UnifiedDialogueLog(dir);
    log.append('user', '请记住：我平时最喜欢胡椒博士，不喝咖啡。');
    log.append('assistant', '知道了。');
    log.append('user', '以后叫我阿万音铃羽就行，记住。');
    log.append('assistant', '行。');
    const block = buildRecallFactAnchor({
      userText: '我喜欢喝什么？你还记得吗',
      dialogueLog: log,
    });
    assert.ok(block.includes('胡椒博士'));
    assert.ok(block.includes('禁止张冠李戴'));
    fs.rmSync(dir, { recursive: true, force: true });
  });

  it('includes palace excerpt when available', () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'amadeus-recall-p-'));
    const palace = new MemoryPalaceStore(dir);
    palace.archiveTurn({
      userText: '请记住我对花生过敏',
      assistantText: '记下了，花生不行。',
      userRepliedToProactive: true,
    });
    const block = buildRecallFactAnchor({
      userText: '我有什么过敏',
      dialogueLog: { getRecent: () => [] },
      memoryPalace: palace,
    });
    assert.ok(/花生|过敏/.test(block));
    fs.rmSync(dir, { recursive: true, force: true });
  });
});
