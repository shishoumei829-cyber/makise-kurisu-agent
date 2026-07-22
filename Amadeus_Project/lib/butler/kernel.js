'use strict';

const os = require('os');
const path = require('path');
const { ButlerEventJournal } = require('./eventJournal');
const { ButlerTaskStore } = require('./taskStore');
const { extractUserText, classifyButlerIntent } = require('./intent');
const { ButlerPlanner } = require('./planner');
const { ReminderStore } = require('./reminderStore');
const { registerLocalCapabilities } = require('./localCapabilities');
const { FileUndoStore } = require('./fileUndoStore');
const { registerFileMutationCapabilities } = require('./fileMutationCapabilities');
const { registerWindowsCapabilities } = require('./windowsCapabilities');
const { ButlerUserWorld } = require('./userWorld');

class ButlerKernel {
  constructor(options = {}) {
    this.journal = options.journal || new ButlerEventJournal(options.dataDir);
    this.tasks = options.tasks || new ButlerTaskStore(options.dataDir, this.journal);
    this.reminders = options.reminders || new ReminderStore(options.dataDir, this.journal);
    this.fileUndo = options.fileUndo || new FileUndoStore(options.dataDir, this.journal);
    this.userWorld = options.userWorld || new ButlerUserWorld(options.dataDir, this.journal);
    this.planner = options.planner || new ButlerPlanner({ reasoner: options.reasoner });
    this.capabilities = new Map();
    this._operatorBusy = false;
    this._registerFoundationCapabilities();
    const allowedRoots = options.allowedRoots || [
      path.join(os.homedir(), 'Downloads'),
      os.homedir(),
      options.rootPath,
    ];
    registerLocalCapabilities(this, { allowedRoots });
    registerFileMutationCapabilities(this, { allowedRoots });
    registerWindowsCapabilities(this, { allowedRoots, ...(options.windowsOptions || {}) });
  }

  _registerFoundationCapabilities() {
    this.registerCapability({ id: 'task.manage', name: '目标与任务管理', available: true, risk: 'low' });
    this.registerCapability({ id: 'task.verify', name: '执行证据与完成验证', available: true, risk: 'low' });
    this.registerCapability({
      id: 'system.inspect',
      name: '读取本机运行状态',
      available: true,
      risk: 'low',
      description: '只读获取平台、运行时间、内存和负载，用于诊断。',
    }, {
      execute: async () => ({
        ok: true,
        summary: `系统=${os.platform()} ${os.release()}，运行时间=${Math.floor(os.uptime())}秒，可用内存=${os.freemem()}字节`,
        artifact: 'system://runtime-status',
      }),
      verify: async (result) => ({
        passed: result.ok === true && /^系统=/.test(result.summary || ''),
        summary: '已重新检查系统状态结果的数据结构。',
      }),
    });
    this.registerCapability({ id: 'file.local', name: '本地文件操作', available: false, risk: 'medium' });
    this.registerCapability({ id: 'system.desktop', name: '桌面与应用控制', available: false, risk: 'medium' });
    this.registerCapability({ id: 'web.research', name: '网页研究', available: false, risk: 'medium' });
    this.registerCapability({ id: 'reminder.local', name: '本地提醒', available: false, risk: 'low' });
  }

  registerCapability(definition = {}, implementation = {}) {
    const id = String(definition.id || '').trim();
    if (!id) throw new Error('capability id required');
    this.capabilities.set(id, {
      id,
      name: definition.name || id,
      available: definition.available === true,
      risk: definition.risk || 'low',
      description: definition.description || '',
      inputSchema: definition.inputSchema || {},
      execute: typeof implementation.execute === 'function' ? implementation.execute : null,
      verify: typeof implementation.verify === 'function' ? implementation.verify : null,
    });
  }

  setUserContextProvider(provider) {
    this.userWorld.setContextProvider(provider);
  }

  capabilitySnapshot() {
    return [...this.capabilities.values()].map((item) => ({
      id: item.id,
      name: item.name,
      available: item.available,
      executable: typeof item.execute === 'function',
      risk: item.risk,
      description: item.description,
      inputSchema: item.inputSchema,
    }));
  }

  async executeTask(taskId, capabilityId, args = {}, context = {}) {
    const task = this.tasks.getTask(taskId);
    if (!task) throw new Error('task not found');
    const capability = this.capabilities.get(capabilityId);
    if (!capability || !capability.available || typeof capability.execute !== 'function') {
      throw new Error(`capability unavailable: ${capabilityId}`);
    }
    if (task.requiresConfirmation && task.confirmation?.approved !== true) {
      throw new Error('task requires explicit confirmation');
    }
    if (task.status === 'blocked') this.tasks.transition(task.id, 'ready', { reason: 'retry execution' });
    if (task.status !== 'ready') throw new Error(`task must be ready before execution, current=${task.status}`);

    this.tasks.transition(task.id, 'running', {
      actor: 'executor',
      nextAction: `调用 ${capability.id}`,
    });
    this.journal.append('execution.started', { taskId, capabilityId, args }, {
      actor: 'executor', correlationId: taskId,
    });

    try {
      const raw = await capability.execute(args, { ...context, task: this.tasks.getTask(taskId) });
      const result = {
        ok: raw?.ok === true,
        summary: String(raw?.summary || '').trim() || '执行器没有返回说明',
        artifact: String(raw?.artifact || '').trim(),
        data: raw?.data && typeof raw.data === 'object' ? raw.data : null,
      };
      const evidence = this.tasks.addEvidence(taskId, {
        ok: result.ok,
        kind: 'capability_result',
        capability: capability.id,
        summary: result.summary,
        artifact: result.artifact,
        details: result.data,
      });
      if (!result.ok) {
        const blocked = this.tasks.transition(taskId, 'blocked', {
          actor: 'executor',
          blockedReason: result.summary,
          nextAction: '分析执行失败原因并重新规划',
        });
        this.userWorld.observeTaskOutcome(blocked);
        return { task: blocked, result, evidence, verification: null };
      }

      this.tasks.transition(taskId, 'verifying', { actor: 'executor', nextAction: '验证执行结果' });
      const check = capability.verify
        ? await capability.verify(result, args, { ...context, task: this.tasks.getTask(taskId) })
        : { passed: false, summary: '该能力没有验证器，不能宣布完成。' };
      const verified = this.tasks.verifyTask(taskId, {
        passed: check?.passed === true,
        summary: String(check?.summary || '').trim(),
        checkedBy: `${capability.id}:verifier`,
        nextAction: '修正结果后重新验证',
      });
      this.userWorld.observeTaskOutcome(verified);
      return { task: verified, result, evidence, verification: verified.verification };
    } catch (error) {
      const current = this.tasks.getTask(taskId);
      if (current && ['running', 'verifying'].includes(current.status)) {
        this.tasks.addEvidence(taskId, {
          ok: false,
          kind: 'execution_error',
          capability: capability.id,
          summary: error.message,
        });
        const blocked = this.tasks.transition(taskId, 'blocked', {
          actor: 'executor', blockedReason: error.message, nextAction: '分析错误并选择恢复方案',
        });
        this.userWorld.observeTaskOutcome(blocked);
      }
      throw error;
    }
  }

  async planTask(taskId, options = {}) {
    let task = this.tasks.getTask(taskId);
    if (!task) throw new Error('task not found');
    if (!['proposed', 'waiting_confirmation', 'blocked', 'failed'].includes(task.status)) {
      throw new Error(`task cannot be planned from state=${task.status}`);
    }
    const plan = await this.planner.plan(task, this.capabilitySnapshot(), {
      ...options,
      userContext: options.userContext || this.userWorld.toPlannerContext(),
    });
    if (!plan) return null;
    const planNeedsConfirmation = task.requiresConfirmation || plan.steps.some((step) => {
      const capability = this.capabilities.get(step.capabilityId);
      return capability?.risk === 'high';
    });
    this.tasks.setPlan(taskId, plan.steps, {
      summary: plan.summary,
      source: plan.source,
      confidence: plan.confidence,
      requiresConfirmation: planNeedsConfirmation,
      nextAction: plan.needsClarification
        ? plan.clarificationQuestion
        : planNeedsConfirmation ? '等待用户确认计划' : '准备执行计划',
    });
    task = this.tasks.getTask(taskId);

    if (plan.canExecute === false) {
      if (task.status === 'proposed') this.tasks.transition(taskId, 'planned', { actor: 'planner' });
      task = this.tasks.getTask(taskId);
      if (task.status === 'planned') {
        task = this.tasks.transition(taskId, 'blocked', {
          actor: 'planner',
          blockedReason: plan.blockedReason || '当前没有完成该任务所需的能力。',
          nextAction: plan.blockedReason || '安装或实现所需能力',
        });
      } else if (task.status === 'waiting_confirmation') {
        task = this.tasks.transition(taskId, 'blocked', {
          actor: 'planner',
          blockedReason: plan.blockedReason || '当前没有完成该任务所需的能力。',
          nextAction: plan.blockedReason || '安装或实现所需能力',
        });
      }
      if (options.source === 'background') this._queueFollowup(task, 'blocked');
      return { task, plan };
    }

    if (plan.needsClarification) {
      if (task.status === 'proposed') this.tasks.transition(taskId, 'planned', { actor: 'planner' });
      task = this.tasks.getTask(taskId);
      if (task.status === 'planned') {
        task = this.tasks.transition(taskId, 'blocked', {
          actor: 'planner',
          blockedReason: plan.clarificationQuestion,
          nextAction: plan.clarificationQuestion,
        });
      } else if (task.status === 'waiting_confirmation') {
        task = this.tasks.transition(taskId, 'blocked', {
          actor: 'planner',
          blockedReason: plan.clarificationQuestion,
          nextAction: plan.clarificationQuestion,
        });
      }
      if (options.source === 'background') this._queueFollowup(task, 'blocked');
      return { task, plan };
    }

    if (task.status === 'proposed') task = this.tasks.transition(taskId, 'planned', { actor: 'planner' });
    if (planNeedsConfirmation) {
      if (task.status === 'planned') task = this.tasks.transition(taskId, 'waiting_confirmation', { actor: 'planner' });
      else if (['blocked', 'failed'].includes(task.status)) task = this.tasks.transition(taskId, 'waiting_confirmation', { actor: 'planner' });
      if (options.source === 'background') this._queueFollowup(task, 'waiting_confirmation');
      return { task, plan };
    }
    if (['blocked', 'failed'].includes(task.status)) task = this.tasks.transition(taskId, 'ready', { actor: 'planner' });
    else if (task.status === 'planned') task = this.tasks.transition(taskId, 'ready', { actor: 'planner' });
    return { task, plan };
  }

  /** 非终态任务（最近更新在前）；legacyChat 与意识管线共用 */
  getActiveTasks() {
    return this.tasks.listTasks()
      .filter((task) => !['completed', 'failed', 'cancelled'].includes(task.status));
  }

  _queueFollowup(task, kind) {
    if (!task) return null;
    return this.journal.append('task.followup_required', {
      taskId: task.id,
      kind,
      status: task.status,
    }, { actor: 'operator', correlationId: task.id });
  }

  /**
   * 失败恢复判定：只有「执行确实失败留下证据」的阻塞才自动恢复；
   * 等待用户澄清/确认的阻塞不属于恢复范围。
   */
  _isRecoverable(task) {
    if (!task || task.status !== 'blocked') return false;
    const lastEvidence = task.evidence?.[task.evidence.length - 1];
    if (!lastEvidence || lastEvidence.ok !== false) return false;
    const max = Number(process.env.AMADEUS_BUTLER_MAX_REPLANS) || 2;
    return (task.recovery?.replans || 0) < max;
  }

  /**
   * 恢复循环：带失败证据 replan → 就绪则重跑 → 预算耗尽才回访用户。
   */
  async attemptRecovery(taskId, options = {}) {
    let task = this.tasks.getTask(taskId);
    if (!this._isRecoverable(task)) return { recovered: false, reason: 'not_recoverable', task };
    task = this.tasks.noteReplan(taskId);
    const failureContext = (task.evidence || [])
      .filter((item) => item.ok === false)
      .slice(-3)
      .map((item) => `${item.capability || item.kind}: ${item.summary}`);
    this.journal.append('task.recovery_started', {
      taskId,
      attempt: task.recovery.replans,
      failureContext,
    }, { actor: 'operator', correlationId: taskId });

    let planned = null;
    try {
      planned = await this.planTask(taskId, {
        ...options,
        failureContext,
        source: options.source || 'recovery',
      });
    } catch (error) {
      this.failPlanning(taskId, error, { followup: true });
      return { recovered: false, reason: 'replan_failed', task: this.tasks.getTask(taskId) };
    }

    task = this.tasks.getTask(taskId);
    if (task.status === 'waiting_confirmation') {
      this._queueFollowup(task, 'waiting_confirmation');
      return { recovered: false, reason: 'needs_confirmation', task, plan: planned?.plan || null };
    }
    if (task.status !== 'ready') {
      if (!this._isRecoverable(task)) this._queueFollowup(task, 'blocked');
      return { recovered: false, reason: `not_ready_after_replan(${task.status})`, task, plan: planned?.plan || null };
    }

    const execution = await this.runPlan(taskId, { source: 'recovery' });
    task = execution.task;
    if (task.status === 'completed') {
      this._queueFollowup(task, 'completed');
    } else if (task.status === 'blocked' && !this._isRecoverable(task)) {
      this._queueFollowup(task, 'blocked');
    }
    return { recovered: task.status === 'completed', task, execution };
  }

  failPlanning(taskId, error, options = {}) {
    let task = this.tasks.getTask(taskId);
    if (!task) return null;
    const reason = `执行脑规划失败：${String(error?.message || error || 'unknown error').slice(0, 600)}`;
    if (task.status === 'proposed') task = this.tasks.transition(taskId, 'planned', { actor: 'planner' });
    if (['planned', 'waiting_confirmation'].includes(task.status)) {
      task = this.tasks.transition(taskId, 'blocked', {
        actor: 'planner', blockedReason: reason, nextAction: '检查执行脑连接后重新规划',
      });
    }
    this.journal.append('planning.failed', { taskId, reason }, { actor: 'planner', correlationId: taskId });
    if (options.followup !== false) this._queueFollowup(task, 'blocked');
    return task;
  }

  updatesSince(since = 0, limit = 30) {
    const after = Math.max(0, Number(since) || 0);
    return this.journal.recent(1000, { type: 'task.followup_required' })
      .filter((event) => event.ts > after)
      .slice(-Math.min(100, Math.max(1, Number(limit) || 30)))
      .map((event) => {
        const task = this.tasks.getTask(event.payload?.taskId);
        if (!task) return null;
        const status = task.status;
        let text = '';
        if (status === 'waiting_confirmation') {
          const steps = (task.plan || []).map((step) => {
            const cap = this.capabilities.get(step.capabilityId);
            return cap?.name || step.capabilityId;
          }).join('、');
          text = `“${task.title}”的执行计划已经准备好了：${task.planSummary || steps || '高风险操作'}。需要你明确说“确认”后我才会执行。`;
        } else if (status === 'completed') {
          const evidence = task.evidence[task.evidence.length - 1];
          text = `“${task.title}”已经完成并验证通过。${evidence?.summary || task.verification?.summary || ''}`.trim();
        } else if (status === 'blocked' || status === 'failed') {
          text = `“${task.title}”现在无法继续：${task.blockedReason || '缺少必要条件'}。`;
        }
        if (!text) return null;
        return { id: event.id, ts: event.ts, taskId: task.id, status, text };
      })
      .filter(Boolean);
  }

  taskPromptBlock(taskId, execution = null) {
    const task = typeof taskId === 'string' ? this.tasks.getTask(taskId) : taskId;
    if (!task) return '';
    const lines = [
      '【智能管家任务事实】',
      `任务=${task.title}`,
      `任务ID=${task.id}，状态=${task.status}，风险=${task.risk}。`,
    ];
    if (task.planSummary) lines.push(`计划=${task.planSummary}`);
    if (task.nextAction) lines.push(`下一步=${task.nextAction}`);
    if (task.blockedReason) lines.push(`阻塞原因=${task.blockedReason}`);
    if (execution?.steps?.length) {
      lines.push(`执行结果=${execution.steps.map((item) => item.result?.summary).filter(Boolean).join('；')}`);
    }
    if (task.verification?.passed) lines.push(`验证=${task.verification.summary}`);
    lines.push(task.status === 'completed'
      ? '该任务已有执行证据并验证完成，可以如实汇报。'
      : '登记或计划不等于执行；没有工具证据与验证结果，绝不能声称已经完成。');
    return lines.join('\n');
  }

  async runPlan(taskId, context = {}) {
    let task = this.tasks.getTask(taskId);
    if (!task) throw new Error('task not found');
    if (task.requiresConfirmation && task.confirmation?.approved !== true) throw new Error('task requires explicit confirmation');
    if (task.status !== 'ready') throw new Error(`task must be ready before plan execution, current=${task.status}`);
    if (!Array.isArray(task.plan) || !task.plan.length) throw new Error('task has no executable plan');
    task = this.tasks.transition(taskId, 'running', { actor: 'executor', nextAction: '执行任务计划' });
    const stepResults = [];

    try {
      for (const step of task.plan) {
        const capability = this.capabilities.get(step.capabilityId);
        if (!capability || !capability.available || typeof capability.execute !== 'function') {
          throw new Error(`capability unavailable: ${step.capabilityId}`);
        }
        this.journal.append('execution.step_started', { taskId, step }, {
          actor: 'executor', correlationId: taskId,
        });
        const raw = await capability.execute(step.args || {}, { ...context, task: this.tasks.getTask(taskId), step });
        const result = {
          ok: raw?.ok === true,
          summary: String(raw?.summary || '').trim() || '执行器没有返回说明',
          artifact: String(raw?.artifact || '').trim(),
          data: raw?.data && typeof raw.data === 'object' ? raw.data : null,
        };
        const verification = capability.verify
          ? await capability.verify(result, step.args || {}, { ...context, task: this.tasks.getTask(taskId), step })
          : { passed: false, summary: '该能力没有验证器。' };
        const evidence = this.tasks.addEvidence(taskId, {
          ok: result.ok && verification?.passed === true,
          kind: 'plan_step_result',
          capability: capability.id,
          summary: `${result.summary} 验证：${String(verification?.summary || '')}`.trim(),
          artifact: result.artifact,
          details: result.data,
        });
        stepResults.push({ step, result, verification, evidence });
        if (!result.ok || verification?.passed !== true) {
          const reason = !result.ok ? result.summary : String(verification?.summary || '步骤验证失败');
          task = this.tasks.transition(taskId, 'blocked', {
            actor: 'executor', blockedReason: reason, nextAction: '分析失败步骤并重新规划',
          });
          return { task, steps: stepResults, verification: null };
        }
      }

      this.tasks.transition(taskId, 'verifying', { actor: 'executor', nextAction: '核对全部步骤' });
      task = this.tasks.verifyTask(taskId, {
        passed: stepResults.length === task.plan.length && stepResults.every((item) => item.verification?.passed === true),
        checkedBy: 'plan-verifier',
        summary: `全部 ${stepResults.length} 个计划步骤均执行并验证通过。`,
      });
      this.userWorld.observeTaskOutcome(task);
      return { task, steps: stepResults, verification: task.verification };
    } catch (error) {
      const current = this.tasks.getTask(taskId);
      let blocked = current;
      if (current && ['running', 'verifying'].includes(current.status)) {
        this.tasks.addEvidence(taskId, {
          ok: false, kind: 'execution_error', capability: 'plan', summary: error.message,
        });
        blocked = this.tasks.transition(taskId, 'blocked', {
          actor: 'executor', blockedReason: error.message, nextAction: '分析错误并重新规划',
        });
        this.userWorld.observeTaskOutcome(blocked);
      }
      return {
        task: blocked,
        steps: stepResults,
        verification: null,
        error: { message: error.message },
      };
    }
  }

  async operatorTick() {
    if (this._operatorBusy) return { acted: false, reason: 'busy' };
    this._operatorBusy = true;
    try {
      const ready = this.tasks.listTasks({ status: 'ready' }).find((task) => {
        const authorized = task.risk === 'low' && !task.requiresConfirmation
          || task.requiresConfirmation && task.confirmation?.approved === true;
        if (!authorized) return false;
        return Array.isArray(task.plan) && task.plan.length > 0 && task.plan.every((step) => {
          const capability = this.capabilities.get(step.capabilityId);
          if (!capability?.available || typeof capability.execute !== 'function') return false;
          return capability.risk === 'low' || task.confirmation?.approved === true;
        });
      });
      if (!ready) {
        const recoverable = this.tasks.listTasks({ status: 'blocked' }).find((task) => (
          this._isRecoverable(task)
          && (!task.requiresConfirmation || task.confirmation?.approved === true)
        ));
        if (!recoverable) return { acted: false, reason: 'no_safe_ready_task' };
        const recovery = await this.attemptRecovery(recoverable.id, { allowStrong: true });
        return { acted: true, taskId: recoverable.id, recovery };
      }
      const execution = await this.runPlan(ready.id, { source: 'operator' });
      if (execution?.task?.status === 'completed') this._queueFollowup(execution.task, 'completed');
      else if (execution?.task?.status === 'blocked') {
        // 留给下一个 tick 的恢复循环；预算耗尽才回访用户
        if (!this._isRecoverable(execution.task)) this._queueFollowup(execution.task, 'blocked');
      }
      return { acted: true, taskId: ready.id, execution };
    } finally {
      this._operatorBusy = false;
    }
  }

  observeUserRequest(input = {}) {
    const text = extractUserText(input.body || input.text || input);
    if (!text) return { text: '', intent: classifyButlerIntent(''), task: null, created: false, promptBlock: '' };
    const source = input.source || 'chat';
    const requestKey = String(input.turnId || input.requestKey || '').trim();
    const event = this.journal.append('user.request_observed', { text }, {
      actor: 'user', source, correlationId: requestKey,
    });
    this.userWorld.observeUserStatement(text);
    const confirmation = /^(?:确认|我确认|可以执行|执行吧|继续执行|同意)[。！!\s]*$/.test(text);
    const rejection = /^(?:取消|不要执行|别执行|拒绝|算了)[。！!\s]*$/.test(text);
    if (confirmation || rejection) {
      const pending = this.tasks.listTasks({ status: 'waiting_confirmation' })[0] || null;
      if (pending) {
        if (rejection) {
          const task = this.tasks.confirm(pending.id, false, { note: text });
          return { event, text, intent: { actionable: true, category: 'confirmation' }, task, created: false, action: 'confirmation_rejected', promptBlock: this.taskPromptBlock(task) };
        }
        if (!Array.isArray(pending.plan) || !pending.plan.length) {
          return { event, text, intent: { actionable: true, category: 'confirmation' }, task: pending, created: false, action: 'confirmation_waiting_for_plan', promptBlock: this.taskPromptBlock(pending) };
        }
        const task = this.tasks.confirm(pending.id, true, { note: text });
        return { event, text, intent: { actionable: true, category: 'confirmation' }, task, created: false, action: 'confirmation_approved', promptBlock: this.taskPromptBlock(task) };
      }
    }
    const intent = classifyButlerIntent(text);
    let task = null;
    let created = false;
    if (intent.actionable) {
      const result = this.tasks.createTask({
        title: intent.title,
        description: text,
        type: intent.category,
        risk: intent.risk,
        requiresConfirmation: intent.requiresConfirmation,
        source,
        requestKey,
        nextAction: '等待执行脑生成可验证计划',
      });
      task = result.task;
      created = result.created;
    }
    return {
      event,
      text,
      intent,
      task,
      created,
      promptBlock: task ? this.taskPromptBlock(task) : '',
    };
  }

  status() {
    const tasks = this.tasks.listTasks();
    const counts = {};
    for (const task of tasks) counts[task.status] = (counts[task.status] || 0) + 1;
    return {
      ok: true,
      product: 'Amadeus Butler Kernel',
      phase: 'operator_v2',
      objective: '长期理解用户，主动接住目标，执行现实任务，验证结果并持续跟进。',
      taskCounts: counts,
      activeTasks: tasks.filter((task) => !['completed', 'failed', 'cancelled'].includes(task.status)).slice(0, 20),
      capabilities: this.capabilitySnapshot(),
      completionRule: 'successful evidence + passed verification',
      userWorld: this.userWorld.snapshot(),
    };
  }
}

module.exports = { ButlerKernel };
