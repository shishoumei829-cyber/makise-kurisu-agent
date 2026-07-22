'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');

const { ButlerKernel } = require('../lib/butler/kernel');
const { UnifiedDialogueLog } = require('../lib/unifiedDialogueLog');
const { collectCandidates } = require('../brain/workspace');
const { init: initUserModel, UserModel } = require('../user_model');

function makeKernel(options = {}) {
  const dataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'amadeus-butler-'));
  return { dataDir, kernel: new ButlerKernel({ dataDir, ...options }) };
}

test('butler keeps casual conversation out of the task queue', () => {
  const { kernel } = makeKernel();
  const result = kernel.observeUserRequest({ text: '今天有点累', turnId: 'casual-1' });
  assert.equal(result.intent.actionable, false);
  assert.equal(result.task, null);
  assert.equal(kernel.tasks.listTasks().length, 0);
});

test('explicit requests become persistent tasks and retry is idempotent', () => {
  const { dataDir, kernel } = makeKernel();
  const first = kernel.observeUserRequest({ text: '帮我整理下载文件夹', turnId: 'turn-1' });
  const second = kernel.observeUserRequest({ text: '帮我整理下载文件夹', turnId: 'turn-1' });
  assert.equal(first.created, true);
  assert.equal(second.created, false);
  assert.equal(first.task.id, second.task.id);
  assert.equal(kernel.tasks.listTasks().length, 1);

  const reloaded = new ButlerKernel({ dataDir });
  assert.equal(reloaded.tasks.listTasks().length, 1);
  assert.equal(reloaded.tasks.getTask(first.task.id).title, '帮我整理下载文件夹');
});

test('high-risk requests require a concrete plan before explicit confirmation', async () => {
  const { kernel } = makeKernel();
  const result = kernel.observeUserRequest({ text: '删除下载目录里的所有文件', turnId: 'danger-1' });
  assert.equal(result.task.risk, 'high');
  assert.equal(result.task.status, 'proposed');
  assert.equal(result.task.requiresConfirmation, true);
  assert.equal(result.task.confirmation, null);
});

test('a task cannot be completed without successful evidence and verification', () => {
  const { kernel } = makeKernel();
  const { task } = kernel.observeUserRequest({ text: '帮我创建一个测试文件', turnId: 'proof-1' });
  kernel.tasks.transition(task.id, 'planned', { plan: ['创建文件', '重新读取文件'] });
  kernel.tasks.transition(task.id, 'ready');
  kernel.tasks.transition(task.id, 'running');
  assert.throws(() => kernel.tasks.transition(task.id, 'completed'), /requires verifyTask/);
  assert.throws(() => kernel.tasks.verifyTask(task.id, { passed: true }), /requires successful execution evidence/);

  kernel.tasks.addEvidence(task.id, {
    ok: true,
    kind: 'file_stat',
    capability: 'file.local',
    summary: '文件存在且可重新读取',
    artifact: 'C:\\Temp\\amadeus-proof.txt',
  });
  kernel.tasks.transition(task.id, 'verifying');
  const completed = kernel.tasks.verifyTask(task.id, {
    passed: true,
    checkedBy: 'file-verifier',
    summary: '路径、大小和内容均符合要求',
  });
  assert.equal(completed.status, 'completed');
  assert.equal(completed.verification.passed, true);
});

test('capability execution closes the observe-act-verify loop', async () => {
  const { kernel } = makeKernel();
  const { task } = kernel.observeUserRequest({ text: '帮我检查电脑运行状态', turnId: 'inspect-1' });
  kernel.tasks.transition(task.id, 'planned', { plan: ['读取状态', '验证返回结构'] });
  kernel.tasks.transition(task.id, 'ready');
  const result = await kernel.executeTask(task.id, 'system.inspect');
  assert.equal(result.task.status, 'completed');
  assert.equal(result.result.ok, true);
  assert.equal(result.evidence.capability, 'system.inspect');
  assert.equal(result.verification.passed, true);
});

test('reminder request is planned, persisted, executed and verified', async () => {
  const { dataDir, kernel } = makeKernel();
  const { task } = kernel.observeUserRequest({ text: '提醒我10分钟后喝水', turnId: 'reminder-1' });
  const planned = await kernel.planTask(task.id);
  assert.equal(planned.plan.source, 'heuristic');
  assert.equal(planned.task.status, 'ready');
  assert.equal(planned.task.plan[0].capabilityId, 'reminder.create');

  const executed = await kernel.runPlan(task.id);
  assert.equal(executed.task.status, 'completed');
  assert.equal(kernel.reminders.list().length, 1);
  assert.equal(kernel.reminders.list()[0].content, '喝水');

  const reloaded = new ButlerKernel({ dataDir });
  assert.equal(reloaded.reminders.list().length, 1);
  assert.equal(reloaded.reminders.list()[0].status, 'scheduled');
});

test('file search stays inside allowed roots and verifies every result', async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'amadeus-files-'));
  fs.writeFileSync(path.join(root, 'project-report-unique.txt'), 'proof', 'utf8');
  const { kernel } = makeKernel({ allowedRoots: [root] });
  const { task } = kernel.observeUserRequest({ text: '帮我找一下 project-report-unique 文件', turnId: 'file-1' });
  const planned = await kernel.planTask(task.id);
  assert.equal(planned.task.plan[0].capabilityId, 'file.search');
  planned.task.plan[0].args.root = root;
  const executed = await kernel.runPlan(task.id);
  assert.equal(executed.task.status, 'completed');
  assert.match(executed.steps[0].result.artifact, /project-report-unique\.txt/);

  const { task: outsideTask } = kernel.observeUserRequest({ text: '帮我找一下 forbidden 文件', turnId: 'file-2' });
  await kernel.planTask(outsideTask.id);
  outsideTask.plan[0].args.root = path.parse(root).root;
  const blocked = await kernel.runPlan(outsideTask.id);
  assert.equal(blocked.task.status, 'blocked');
  assert.match(blocked.task.blockedReason, /outside allowed roots/);
});

test('unsupported tasks are planned by the isolated strong work brain', async () => {
  let calls = 0;
  const { kernel } = makeKernel({
    reasoner: async ({ task, capabilities }) => {
      calls += 1;
      assert.match(task.description, /制定提升专注力的方案/);
      assert.ok(capabilities.some((item) => item.id === 'system.inspect'));
      return JSON.stringify({
        summary: '先读取系统状态作为分析依据。',
        confidence: 0.9,
        needsClarification: false,
        clarificationQuestion: '',
        steps: [{ id: 'inspect', capabilityId: 'system.inspect', args: {}, successCriteria: '取得状态' }],
      });
    },
  });
  const { task } = kernel.observeUserRequest({ text: '帮我制定提升专注力的方案', turnId: 'strong-1' });
  assert.equal(await kernel.planTask(task.id, { allowStrong: false }), null);
  const planned = await kernel.planTask(task.id, { allowStrong: true });
  assert.equal(calls, 1);
  assert.equal(planned.plan.source, 'work_brain');
  assert.equal(planned.task.status, 'ready');
});

test('work brain reports a capability gap instead of choosing an unrelated tool', async () => {
  const { kernel } = makeKernel({
    reasoner: async () => JSON.stringify({
      summary: '当前没有邮件发送能力。',
      confidence: 0.96,
      canExecute: false,
      blockedReason: '需要先接入邮件账户与发送能力。',
      needsClarification: false,
      clarificationQuestion: '',
      steps: [],
    }),
  });
  const { task } = kernel.observeUserRequest({ text: '帮我发送一封邮件', turnId: 'gap-1' });
  const planned = await kernel.planTask(task.id);
  assert.equal(planned.plan.canExecute, false);
  assert.equal(planned.task.status, 'blocked');
  assert.match(planned.task.blockedReason, /邮件账户/);
});

test('operator executes only safe ready work and leaves confirmation tasks alone', async () => {
  const { kernel } = makeKernel();
  const { task } = kernel.observeUserRequest({ text: '帮我检查电脑运行状态', turnId: 'operator-1' });
  await kernel.planTask(task.id);
  const tick = await kernel.operatorTick();
  assert.equal(tick.acted, true);
  assert.equal(kernel.tasks.getTask(task.id).status, 'completed');

  const risky = kernel.observeUserRequest({ text: '删除所有下载文件', turnId: 'operator-risk-1' }).task;
  const quiet = await kernel.operatorTick();
  assert.equal(quiet.acted, false);
  assert.equal(kernel.tasks.getTask(risky.id).status, 'proposed');
});

test('trusted Windows app launch uses an allowlist and returns verifiable dispatch evidence', async () => {
  const launches = [];
  const { kernel } = makeKernel({
    windowsOptions: {
      allowTestPlatform: true,
      launcher: (exe, args) => { launches.push({ exe, args }); return 4242; },
    },
  });
  const { task } = kernel.observeUserRequest({ text: '打开记事本', turnId: 'app-1' });
  const planned = await kernel.planTask(task.id);
  assert.equal(planned.task.plan[0].capabilityId, 'app.launch');
  const executed = await kernel.runPlan(task.id);
  assert.equal(executed.task.status, 'completed');
  assert.deepEqual(launches, [{ exe: 'notepad.exe', args: [] }]);
  assert.match(executed.steps[0].verification.summary, /启动请求/);
});

test('file mutation is no-overwrite, confirmation-gated and reversible', async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'amadeus-mutate-'));
  const { kernel } = makeKernel({ allowedRoots: [root] });

  const create = kernel.tasks.createTask({ title: '创建文件', description: '创建文件', type: 'file', requestKey: 'create-1' }).task;
  kernel.tasks.setPlan(create.id, [{
    id: 'create', capabilityId: 'file.create_text', args: { directory: root, name: 'proof.txt', content: 'hello' }, successCriteria: 'hash matches',
  }], { summary: '创建且复查', source: 'test', confidence: 1 });
  kernel.tasks.transition(create.id, 'planned');
  kernel.tasks.transition(create.id, 'ready');
  const created = await kernel.runPlan(create.id);
  const target = path.join(root, 'proof.txt');
  assert.equal(created.task.status, 'completed');
  assert.equal(fs.readFileSync(target, 'utf8'), 'hello');

  const overwrite = kernel.tasks.createTask({ title: '再次创建', description: '再次创建', type: 'file', requestKey: 'create-2' }).task;
  kernel.tasks.setPlan(overwrite.id, [{
    id: 'overwrite', capabilityId: 'file.create_text', args: { directory: root, name: 'proof.txt', content: 'changed' }, successCriteria: 'no overwrite',
  }], { summary: '不能覆盖', source: 'test', confidence: 1 });
  kernel.tasks.transition(overwrite.id, 'planned');
  kernel.tasks.transition(overwrite.id, 'ready');
  const refused = await kernel.runPlan(overwrite.id);
  assert.equal(refused.task.status, 'blocked');
  assert.equal(fs.readFileSync(target, 'utf8'), 'hello');

  const trash = kernel.tasks.createTask({
    title: '删除 proof.txt', description: '删除 proof.txt', type: 'file', risk: 'high', requiresConfirmation: true, requestKey: 'trash-1',
  }).task;
  kernel.tasks.setPlan(trash.id, [{
    id: 'trash', capabilityId: 'file.trash', args: { path: target }, successCriteria: 'recoverable copy exists',
  }], { summary: '移入可恢复区', source: 'test', confidence: 1, requiresConfirmation: true });
  kernel.tasks.transition(trash.id, 'planned');
  kernel.tasks.transition(trash.id, 'waiting_confirmation');
  const confirmed = kernel.observeUserRequest({ text: '确认', turnId: 'confirm-trash-1' });
  assert.equal(confirmed.action, 'confirmation_approved');
  assert.equal(confirmed.task.id, trash.id);
  const trashed = await kernel.runPlan(trash.id);
  assert.equal(trashed.task.status, 'completed');
  assert.equal(fs.existsSync(target), false);
  const trashId = trashed.steps[0].result.data.entry.id;

  const restore = kernel.tasks.createTask({ title: '恢复文件', description: '恢复文件', type: 'file', requestKey: 'restore-1' }).task;
  kernel.tasks.setPlan(restore.id, [{
    id: 'restore', capabilityId: 'file.restore', args: { trashId }, successCriteria: 'original exists',
  }], { summary: '恢复原路径', source: 'test', confidence: 1 });
  kernel.tasks.transition(restore.id, 'planned');
  kernel.tasks.transition(restore.id, 'ready');
  const restored = await kernel.runPlan(restore.id);
  assert.equal(restored.task.status, 'completed');
  assert.equal(fs.readFileSync(target, 'utf8'), 'hello');
});

test('confirmation text never approves a task before its plan exists', () => {
  const { kernel } = makeKernel();
  const pending = kernel.observeUserRequest({ text: '删除下载目录里的所有文件', turnId: 'unplanned-risk' }).task;
  const result = kernel.observeUserRequest({ text: '确认', turnId: 'early-confirm' });
  assert.equal(result.task, null);
  assert.equal(pending.status, 'proposed');
  assert.equal(pending.confirmation, null);
});

test('background plans produce a follow-up when user confirmation is required', async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'amadeus-followup-'));
  const target = path.join(root, 'remove-me.txt');
  fs.writeFileSync(target, 'safe', 'utf8');
  const { kernel } = makeKernel({
    allowedRoots: [root],
    reasoner: async () => JSON.stringify({
      summary: '把指定文件移入可恢复区。',
      confidence: 0.99,
      canExecute: true,
      blockedReason: '',
      needsClarification: false,
      clarificationQuestion: '',
      steps: [{ id: 'trash', capabilityId: 'file.trash', args: { path: target }, successCriteria: '恢复副本存在' }],
    }),
  });
  const before = Date.now() - 1;
  const { task } = kernel.observeUserRequest({ text: `删除 ${target}`, turnId: 'followup-1' });
  const planned = await kernel.planTask(task.id, { allowStrong: true, source: 'background' });
  assert.equal(planned.task.status, 'waiting_confirmation');
  const updates = kernel.updatesSince(before);
  assert.equal(updates.length, 1);
  assert.equal(updates[0].status, 'waiting_confirmation');
  assert.match(updates[0].text, /明确说“确认”/);

  kernel.observeUserRequest({ text: '确认', turnId: 'followup-confirm' });
  const tick = await kernel.operatorTick();
  assert.equal(tick.execution.task.status, 'completed');
  const done = kernel.updatesSince(updates[0].ts);
  assert.ok(done.some((item) => item.status === 'completed'));
});

test('background planner failure becomes a visible blocked task', () => {
  const { kernel } = makeKernel();
  const { task } = kernel.observeUserRequest({ text: '帮我完成一个复杂任务', turnId: 'plan-fail-1' });
  const before = Date.now() - 1;
  const blocked = kernel.failPlanning(task.id, new Error('provider unavailable'));
  assert.equal(blocked.status, 'blocked');
  assert.match(blocked.blockedReason, /provider unavailable/);
  assert.ok(kernel.updatesSince(before).some((item) => /无法继续/.test(item.text)));
});

test('operational preferences survive restart and constrain later plans', async () => {
  const { dataDir, kernel } = makeKernel();
  kernel.observeUserRequest({ text: '以后文件默认都放桌面', turnId: 'preference-1' });
  assert.equal(kernel.userWorld.snapshot().operationalPreferences.defaultDirectory, 'desktop');

  const { task } = kernel.observeUserRequest({ text: '创建一个 note.txt 内容是实验记录', turnId: 'preference-task-1' });
  const planned = await kernel.planTask(task.id);
  assert.equal(planned.task.plan[0].capabilityId, 'file.create_text');
  assert.equal(planned.task.plan[0].args.directory, 'desktop');

  const reloaded = new ButlerKernel({ dataDir });
  assert.equal(reloaded.userWorld.snapshot().operationalPreferences.defaultDirectory, 'desktop');
});

test('planner user context is privacy-cropped and contains task outcome learning', async () => {
  let observedContext = null;
  const { kernel } = makeKernel({
    reasoner: async ({ userContext }) => {
      observedContext = userContext;
      return JSON.stringify({
        summary: '没有对应能力', confidence: 0.9, canExecute: false,
        blockedReason: '缺少能力', needsClarification: false, clarificationQuestion: '', steps: [],
      });
    },
  });
  kernel.setUserContextProvider(() => ({
    userModel: {
      preferences: { response_length: 'short', topics: { technical: 9 } },
      name: 'PRIVATE_NAME',
      rawDialogue: 'PRIVATE_DIALOGUE',
    },
  }));
  const done = kernel.observeUserRequest({ text: '帮我检查电脑运行状态', turnId: 'world-outcome-1' }).task;
  await kernel.planTask(done.id);
  await kernel.runPlan(done.id);

  const unknown = kernel.observeUserRequest({ text: '帮我处理一个未知外部服务', turnId: 'world-plan-1' }).task;
  await kernel.planTask(unknown.id);
  assert.equal(observedContext.interactionTendencies.responseLength, 'short');
  assert.deepEqual(observedContext.interactionTendencies.topTopics, ['technical']);
  assert.ok(observedContext.taskOutcomes.completed >= 1);
  const serialized = JSON.stringify(observedContext);
  assert.doesNotMatch(serialized, /PRIVATE_NAME|PRIVATE_DIALOGUE/);
  assert.equal(observedContext.privacy.containsIdentity, false);
});

test('legacy user model actually reloads persisted understanding after restart', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'amadeus-user-model-'));
  fs.writeFileSync(path.join(dir, 'user_model.json'), JSON.stringify({
    stats: { total_messages: 42 },
    preferences: { response_length: 'short', topics: { technical: 7 } },
    relationship: { stage: 'close', trust_level: 0.8, closeness: 0.75 },
    _log: [{ text: 'saved', timestamp: 1 }],
  }), 'utf8');
  initUserModel(dir);
  const model = new UserModel();
  assert.equal(model.model.stats.total_messages, 42);
  assert.equal(model.model.preferences.response_length, 'short');
  assert.equal(model.model.preferences.topics.technical, 7);
  assert.equal(model.model.relationship.stage, 'close');
  assert.equal('_log' in model.model, false);
});

test('kernel exposes active (non-terminal) tasks for the brain and chat', async () => {
  const { kernel } = makeKernel();
  const { task } = kernel.observeUserRequest({ text: '帮我检查电脑运行状态', turnId: 'active-1' });
  assert.ok(kernel.getActiveTasks().some((item) => item.id === task.id));
  await kernel.planTask(task.id);
  await kernel.runPlan(task.id);
  assert.equal(kernel.getActiveTasks().some((item) => item.id === task.id), false);
});

test('unified journal filters by source and notifies subscribers in real time', () => {
  const { kernel } = makeKernel();
  const seen = [];
  const unsubscribe = kernel.journal.subscribe((event) => seen.push(event.type));
  kernel.journal.append('emotion.pad_shifted', { after: { P: 0.2 } }, { source: 'emotion', actor: 'amadeus' });
  unsubscribe();
  kernel.journal.append('emotion.pad_shifted', {}, { source: 'emotion' });
  assert.deepEqual(seen, ['emotion.pad_shifted']);

  const emotionEvents = kernel.journal.recent(50, { source: 'emotion' });
  assert.equal(emotionEvents.length, 2);
  assert.ok(kernel.journal.recent(50, { source: 'butler' }).every((event) => event.source === 'butler'));
});

test('dialogue turns flow into the unified event journal', () => {
  const { dataDir, kernel } = makeKernel();
  const dialogue = new UnifiedDialogueLog(dataDir);
  dialogue.setEventSink((entry) => {
    kernel.journal.append('dialogue.turn', { role: entry.role, text: entry.text }, {
      actor: entry.role === 'user' ? 'user' : 'amadeus',
      source: 'dialogue',
      ts: entry.ts,
    });
  });
  dialogue.append('user', '今晚记得整理下载文件夹');
  const events = kernel.journal.recent(10, { source: 'dialogue' });
  assert.equal(events.length, 1);
  assert.equal(events[0].type, 'dialogue.turn');
  assert.match(events[0].payload.text, /整理下载文件夹/);
});

test('a blocked task recovers automatically by replanning with failure evidence', async () => {
  let observedFailureContext = null;
  const { kernel } = makeKernel({
    reasoner: async ({ failureContext }) => {
      observedFailureContext = failureContext;
      return {
        summary: '避开失败原因后重试。',
        confidence: 0.9,
        canExecute: true,
        needsClarification: false,
        clarificationQuestion: '',
        steps: [{ id: 'retry', capabilityId: 'test.flaky', args: {}, successCriteria: '返回 ok' }],
      };
    },
  });
  let calls = 0;
  kernel.registerCapability({ id: 'test.flaky', name: '不稳定能力', available: true, risk: 'low' }, {
    execute: async () => {
      calls += 1;
      if (calls === 1) return { ok: false, summary: '第一次执行失败：目标被占用' };
      return { ok: true, summary: '第二次执行成功', artifact: 'test://ok' };
    },
    verify: async (result) => ({ passed: result.ok === true, summary: '结果结构验证通过' }),
  });

  const { task } = kernel.tasks.createTask({ title: '恢复测试', description: '恢复测试', requestKey: 'recover-1' });
  kernel.tasks.setPlan(task.id, [{ id: 'step_1', capabilityId: 'test.flaky', args: {}, successCriteria: 'ok' }], {
    summary: '首跑', source: 'test', confidence: 1,
  });
  kernel.tasks.transition(task.id, 'planned');
  kernel.tasks.transition(task.id, 'ready');
  const first = await kernel.runPlan(task.id);
  assert.equal(first.task.status, 'blocked');

  const before = Date.now() - 1;
  const recovery = await kernel.attemptRecovery(task.id);
  assert.equal(recovery.recovered, true);
  assert.equal(recovery.task.status, 'completed');
  assert.equal(recovery.task.recovery.replans, 1);
  assert.ok(observedFailureContext.some((line) => /目标被占用/.test(line)));
  assert.ok(kernel.updatesSince(before).some((item) => item.status === 'completed'));
  assert.ok(kernel.journal.recent(20, { type: 'task.recovery_started' }).length >= 1);
});

test('recovery respects the replan budget and finally reports back to the user', async () => {
  const { kernel } = makeKernel({
    reasoner: async () => ({
      summary: '再试一次。',
      confidence: 0.7,
      canExecute: true,
      needsClarification: false,
      clarificationQuestion: '',
      steps: [{ id: 'retry', capabilityId: 'test.broken', args: {}, successCriteria: 'ok' }],
    }),
  });
  kernel.registerCapability({ id: 'test.broken', name: '始终失败', available: true, risk: 'low' }, {
    execute: async () => ({ ok: false, summary: '设备不可用' }),
    verify: async () => ({ passed: false, summary: '无结果' }),
  });
  const { task } = kernel.tasks.createTask({ title: '预算测试', description: '预算测试', requestKey: 'budget-1' });
  kernel.tasks.setPlan(task.id, [{ id: 'step_1', capabilityId: 'test.broken', args: {}, successCriteria: 'ok' }], {
    summary: '首跑', source: 'test', confidence: 1,
  });
  kernel.tasks.transition(task.id, 'planned');
  kernel.tasks.transition(task.id, 'ready');
  await kernel.runPlan(task.id);

  const before = Date.now() - 1;
  // operatorTick 会自动挑选可恢复的阻塞任务
  const tick1 = await kernel.operatorTick();
  assert.equal(tick1.acted, true);
  const tick2 = await kernel.operatorTick();
  assert.equal(tick2.acted, true);
  assert.equal(kernel.tasks.getTask(task.id).recovery.replans, 2);

  // 预算（默认 2）耗尽：不再恢复，且已经回访用户
  const exhausted = await kernel.attemptRecovery(task.id);
  assert.equal(exhausted.recovered, false);
  assert.equal(exhausted.reason, 'not_recoverable');
  assert.ok(kernel.updatesSince(before).some((item) => item.status === 'blocked'));
});

test('butler tasks enter the consciousness workspace as intentions', () => {
  const items = collectCandidates({
    perceived: { userContent: '随便聊聊' },
    butler: {
      activeTasks: [
        { id: 't1', title: '整理下载文件夹', status: 'waiting_confirmation' },
        { id: 't2', title: '查资料', status: 'blocked', blockedReason: '网络能力未接入' },
      ],
      updates: [{ taskId: 't3', status: 'completed', text: '“设置提醒”已经完成并验证通过。' }],
    },
  });
  const butlerItems = items.filter((item) => item.source === 'butler');
  assert.equal(butlerItems.length, 3);
  assert.ok(butlerItems.some((item) => /等他确认/.test(item.content)));
  assert.ok(butlerItems.some((item) => /卡住了/.test(item.content)));
  assert.ok(butlerItems.some((item) => /刚办完一件事/.test(item.content)));
});

test('product charter makes the Jarvis direction an executable contract', () => {
  const charter = fs.readFileSync(path.join(__dirname, '..', 'PRODUCT_CHARTER.md'), 'utf8');
  const server = fs.readFileSync(path.join(__dirname, '..', 'server.js'), 'utf8');
  assert.match(charter, /贾维斯感/);
  assert.match(charter, /没有工具结果和验证证据/);
  assert.match(server, /app\.get\('\/butler\/status'/);
  assert.match(server, /app\.post\('\/butler\/tasks\/:id\/verify'/);
  assert.match(server, /app\.post\('\/butler\/tasks\/:id\/execute'/);
  assert.match(server, /butlerKernel\.observeUserRequest/);
});
