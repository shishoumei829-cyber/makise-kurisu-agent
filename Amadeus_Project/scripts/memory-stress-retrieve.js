'use strict';

/**
 * 大量对话压力下的宫殿检索验收。
 *
 * 设计意图：
 * - 先写入少量「目标事实」
 * - 再灌入大量干扰对话，把目标事实挤出短上下文窗口（模拟 toOllamaDialogue maxMsgs）
 * - 断言：短上下文里已经看不到目标事实，但宫殿 retrieve/navigate 仍能命中
 *
 * 用法: node scripts/memory-stress-retrieve.js
 * 通过打印 MEMORY_STRESS_PASS
 */

const assert = require('node:assert/strict');
const fs = require('fs');
const os = require('os');
const path = require('path');

const { MemoryPalaceStore } = require('../lib/memory/palaceStore');
const { UnifiedDialogueLog } = require('../lib/unifiedDialogueLog');

const DISTRACTOR_ROUNDS = 120;
const CONTEXT_MAX_MSGS = 24;

const SEEDS = [
  {
    id: 'pepper',
    user: '请记住，我最喜欢喝的是胡椒博士，别给我推咖啡。',
    assistant: '知道了，你点名要胡椒博士，我记下了。',
    queries: ['我喜欢喝什么', '胡椒博士', '别推咖啡记得吗'],
    mustInclude: ['胡椒博士'],
  },
  {
    id: 'quantum',
    user: '量子纠缠实验的误差又飘了，光纤延迟可能有问题。',
    assistant: '先核对参考时钟和光纤延迟，别急着怪探测器。',
    queries: ['量子纠缠实验怎么样了', '光纤延迟', '实验误差飘了'],
    mustInclude: ['量子', '光纤'],
  },
  {
    id: 'alarm',
    user: '明天早上七点半用电脑提醒叫醒我，别忘了。',
    assistant: '七点半我会到点开口喊你，先记在约定里。',
    queries: ['明早几点叫醒我', '七点半提醒', '闹钟约定'],
    mustInclude: ['七点半', '叫醒'],
  },
];

const DISTRACTOR_TOPICS = [
  ['我觉得今天天气一般，出门有点闷，不想多走。', '嗯，外面看起来有点闷，别硬撑着逛街。'],
  ['我最近代码又报错了，堆栈看不懂，帮我想想怎么查。', '把关键栈贴出来，我们按调用链往上追。'],
  ['中午我吃了面，其实还想再加点菜，但又有点腻。', '那还行，别只喝汤，下午容易饿。'],
  ['排位又输了，我觉得自己操作变形，有点烦。', '别上头，休息一轮再打，状态回来再说。'],
  ['实验室空调好冷，我坐着写数据手都僵了。', '披件外套，别硬扛，冻感冒更麻烦。'],
  ['我想刷会儿视频放松一下，因为刚才太紧张了。', '看完记得回来，别拖太晚。'],
  ['论文引言好难写，我打算先搁着改方法部分。', '可以，先写方法段落，引言最后补。'],
  ['地铁好挤，我差点没挤上车，现在还有点喘。', '到站了跟我说一声，别边走边生气。'],
  ['耳机没电了，我又不能在图书馆外放，好烦。', '插线先用着，别耽误你听课。'],
  ['我想喝奶茶，不过担心糖分太高影响晚上状态。', '少糖，别灌太多，喝完继续干活。'],
  ['我发现邻座一直在打电话，根本没法专心看书。', '换个位置，或者戴降噪，别跟噪音较劲。'],
  ['我计划周末把房间收拾一遍，不然东西找不到。', '行，先清桌面，别一上来就翻柜子。'],
];

function tmpDir() {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'amadeus-mem-stress-'));
}

function archivePair(palace, log, user, assistant, ts) {
  log.append('user', user, { ts, source: 'stress' });
  log.append('assistant', assistant, { ts: ts + 1, source: 'stress' });
  return palace.archiveTurn({ userText: user, assistantText: assistant });
}

function contextContainsAny(dialogue, needles) {
  const blob = dialogue.map((m) => m.content).join('\n');
  return needles.some((n) => blob.includes(n));
}

function excerptHits(excerpt, mustInclude) {
  const text = String(excerpt || '');
  return mustInclude.some((n) => text.includes(n));
}

function main() {
  const dir = tmpDir();
  const palace = new MemoryPalaceStore(dir);
  const log = new UnifiedDialogueLog(dir);
  const report = { dir, seeds: [], distractors: DISTRACTOR_ROUNDS, failures: [] };

  let ts = Date.now() - (DISTRACTOR_ROUNDS + 20) * 60000;

  // 1) 早期写入目标事实
  for (const seed of SEEDS) {
    const r = archivePair(palace, log, seed.user, seed.assistant, ts);
    ts += 60000;
    if (!r.ok) {
      report.failures.push({ stage: 'seed_archive', id: seed.id, reason: r.reason });
    }
    report.seeds.push({ id: seed.id, archived: !!r.ok, room: r.room || null });
  }

  // 2) 大量干扰对话（宫殿 + 对话实录）——必须能归档，才是真·噪声库
  let archivedDistractors = 0;
  for (let i = 0; i < DISTRACTOR_ROUNDS; i++) {
    const [u, a] = DISTRACTOR_TOPICS[i % DISTRACTOR_TOPICS.length];
    const user = `${u} 另外补充一下，这是干扰轮 ${i + 1}，我最近状态一般。`;
    const asst = `${a} 好，第 ${i + 1} 轮我先接住，不扯无关旧事。`;
    const r = archivePair(palace, log, user, asst, ts);
    if (r.ok) archivedDistractors += 1;
    ts += 45000;
  }
  report.archivedDistractors = archivedDistractors;
  if (archivedDistractors < DISTRACTOR_ROUNDS * 0.8) {
    report.failures.push({
      stage: 'distractor_archive',
      detail: `干扰轮归档过少 ${archivedDistractors}/${DISTRACTOR_ROUNDS}`,
    });
  }
  if (palace.counts().total < 40) {
    report.failures.push({
      stage: 'palace_volume',
      detail: `宫殿节点过少 ${palace.counts().total}，不足以证明噪声下检索`,
    });
  }

  // 3) 短上下文应已挤掉种子事实
  const recent = log.toOllamaDialogue({ maxMsgs: CONTEXT_MAX_MSGS });
  const seedNeedles = SEEDS.flatMap((s) => s.mustInclude);
  const stillInContext = contextContainsAny(recent, seedNeedles);
  if (stillInContext) {
    // 再灌一轮直到挤出，或失败
    for (let i = 0; i < 40 && contextContainsAny(log.toOllamaDialogue({ maxMsgs: CONTEXT_MAX_MSGS }), seedNeedles); i++) {
      archivePair(
        palace,
        log,
        `继续闲聊填充 ${i}`,
        `收到，填充对话 ${i}，聊聊天气和代码。`,
        ts,
      );
      ts += 30000;
    }
  }
  const contextAfter = log.toOllamaDialogue({ maxMsgs: CONTEXT_MAX_MSGS });
  const leaked = contextContainsAny(contextAfter, seedNeedles);
  if (leaked) {
    report.failures.push({
      stage: 'context_isolation',
      detail: '目标事实仍出现在短上下文，无法证明宫殿独立召回',
    });
  }

  // 4) 宫殿检索必须命中（与短上下文无关）
  const retrieval = [];
  for (const seed of SEEDS) {
    for (const q of seed.queries) {
      const nav = palace.navigate(q, { topK: 6, minScore: 2.0 });
      const ok = excerptHits(nav.excerpt, seed.mustInclude)
        || (nav.hits || []).some((h) => excerptHits(h.detail, seed.mustInclude));
      retrieval.push({
        seed: seed.id,
        query: q,
        ok,
        top: (nav.hits || []).slice(0, 3).map((h) => ({ score: h.score, detail: String(h.detail || '').slice(0, 60) })),
      });
      if (!ok) {
        report.failures.push({ stage: 'retrieve', seed: seed.id, query: q, excerpt: nav.excerpt.slice(0, 200) });
      }
    }
  }

  // 5) 负例：无关查询不应把胡椒博士顶到第一
  const neg = palace.navigate('地铁挤不挤', { topK: 3, minScore: 2.0 });
  const topDetail = String(neg.hits?.[0]?.detail || '');
  if (topDetail.includes('胡椒博士') && (neg.hits?.[0]?.score || 0) > 8) {
    report.failures.push({ stage: 'negative', detail: '无关查询误把胡椒博士排第一' });
  }

  report.palaceCounts = palace.counts();
  report.dialogueCount = log.entriesCount;
  report.contextMsgs = contextAfter.length;
  report.contextLeakedSeed = leaked;
  report.retrieval = retrieval;
  report.passed = report.failures.length === 0;

  console.log(JSON.stringify(report, null, 2));
  if (!report.passed) {
    console.error('MEMORY_STRESS_FAIL');
    process.exit(1);
  }
  console.log('MEMORY_STRESS_PASS');
}

main();
