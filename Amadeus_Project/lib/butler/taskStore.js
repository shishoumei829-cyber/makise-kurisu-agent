'use strict';

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

const TASK_STATES = Object.freeze([
  'proposed',
  'planned',
  'ready',
  'waiting_confirmation',
  'running',
  'blocked',
  'verifying',
  'completed',
  'failed',
  'cancelled',
]);

const TRANSITIONS = Object.freeze({
  proposed: ['planned', 'waiting_confirmation', 'cancelled'],
  planned: ['ready', 'waiting_confirmation', 'blocked', 'cancelled'],
  ready: ['running', 'waiting_confirmation', 'blocked', 'cancelled'],
  waiting_confirmation: ['ready', 'blocked', 'cancelled'],
  running: ['blocked', 'verifying', 'failed', 'cancelled'],
  blocked: ['ready', 'running', 'waiting_confirmation', 'cancelled', 'failed'],
  verifying: ['running', 'blocked', 'failed'],
  completed: [],
  failed: ['ready', 'waiting_confirmation', 'cancelled'],
  cancelled: [],
});

function now() { return Date.now(); }
function makeId(prefix) { return `${prefix}_${crypto.randomUUID()}`; }

class ButlerTaskStore {
  constructor(dataDir, journal) {
    this.dir = path.join(dataDir, 'butler');
    this.path = path.join(this.dir, 'state.json');
    this.journal = journal;
    fs.mkdirSync(this.dir, { recursive: true });
    this.state = this._load();
  }

  _empty() {
    return { version: 1, savedAt: now(), goals: [], tasks: [] };
  }

  _load() {
    try {
      if (!fs.existsSync(this.path)) return this._empty();
      const parsed = JSON.parse(fs.readFileSync(this.path, 'utf8'));
      return {
        version: 1,
        savedAt: Number(parsed.savedAt) || now(),
        goals: Array.isArray(parsed.goals) ? parsed.goals : [],
        tasks: Array.isArray(parsed.tasks) ? parsed.tasks : [],
      };
    } catch {
      return this._empty();
    }
  }

  _save() {
    this.state.savedAt = now();
    const temp = `${this.path}.tmp`;
    fs.writeFileSync(temp, JSON.stringify(this.state, null, 2), 'utf8');
    fs.renameSync(temp, this.path);
  }

  ensureInboxGoal() {
    let goal = this.state.goals.find((item) => item.systemKey === 'user_requests');
    if (goal) return goal;
    goal = this.createGoal({
      title: '接住并完成用户交代的事情',
      description: '所有明确委托先进入这里，随后规划、执行、验证和跟进。',
      source: 'system',
      systemKey: 'user_requests',
    });
    return goal;
  }

  createGoal(input = {}) {
    const title = String(input.title || '').trim();
    if (!title) throw new Error('goal title required');
    const goal = {
      id: makeId('goal'),
      title: title.slice(0, 240),
      description: String(input.description || '').trim().slice(0, 1600),
      status: 'active',
      priority: input.priority || 'normal',
      source: input.source || 'user',
      systemKey: input.systemKey || '',
      createdAt: now(),
      updatedAt: now(),
    };
    this.state.goals.push(goal);
    this._save();
    this.journal?.append('goal.created', { goal }, { actor: input.source || 'user', correlationId: goal.id });
    return goal;
  }

  listGoals(filter = {}) {
    return this.state.goals.filter((goal) => !filter.status || goal.status === filter.status);
  }

  createTask(input = {}) {
    const title = String(input.title || '').trim();
    if (!title) throw new Error('task title required');
    const requestKey = String(input.requestKey || '').trim();
    if (requestKey) {
      const existing = this.state.tasks.find((task) => task.requestKey === requestKey);
      if (existing) return { task: existing, created: false };
    }
    const goalId = input.goalId || this.ensureInboxGoal().id;
    const task = {
      id: makeId('task'),
      goalId,
      parentTaskId: input.parentTaskId || '',
      title: title.slice(0, 300),
      description: String(input.description || '').trim().slice(0, 2400),
      type: input.type || 'general',
      status: 'proposed',
      priority: input.priority || 'normal',
      risk: input.risk || 'low',
      requiresConfirmation: Boolean(input.requiresConfirmation),
      confirmation: null,
      plan: Array.isArray(input.plan) ? input.plan : [],
      evidence: [],
      verification: null,
      blockedReason: '',
      recovery: { replans: 0, lastReplanAt: 0 },
      nextAction: String(input.nextAction || '等待规划').slice(0, 500),
      source: input.source || 'user',
      requestKey,
      createdAt: now(),
      updatedAt: now(),
    };
    this.state.tasks.push(task);
    this._save();
    this.journal?.append('task.created', { task }, { actor: task.source, correlationId: task.id });
    return { task, created: true };
  }

  getTask(id) {
    return this.state.tasks.find((task) => task.id === id) || null;
  }

  listTasks(filter = {}) {
    return this.state.tasks
      .filter((task) => !filter.status || task.status === filter.status)
      .filter((task) => !filter.goalId || task.goalId === filter.goalId)
      .slice()
      .sort((a, b) => b.updatedAt - a.updatedAt);
  }

  setPlan(id, plan, meta = {}) {
    const task = this.getTask(id);
    if (!task) throw new Error('task not found');
    if (['completed', 'cancelled'].includes(task.status)) throw new Error('terminal task cannot be replanned');
    task.plan = Array.isArray(plan) ? plan.slice(0, 40) : [];
    task.planSummary = String(meta.summary || '').trim().slice(0, 1200);
    task.planSource = String(meta.source || '').trim().slice(0, 80);
    task.planConfidence = Math.max(0, Math.min(1, Number(meta.confidence) || 0));
    if (meta.nextAction != null) task.nextAction = String(meta.nextAction).slice(0, 500);
    if (meta.requiresConfirmation === true) task.requiresConfirmation = true;
    task.updatedAt = now();
    this._save();
    this.journal?.append('task.planned', {
      taskId: task.id,
      summary: task.planSummary,
      source: task.planSource,
      confidence: task.planConfidence,
      steps: task.plan,
    }, { actor: meta.actor || 'planner', correlationId: task.id });
    return task;
  }

  transition(id, nextStatus, meta = {}) {
    if (!TASK_STATES.includes(nextStatus)) throw new Error(`unknown task state: ${nextStatus}`);
    if (nextStatus === 'completed') throw new Error('completed requires verifyTask with successful evidence');
    const task = this.getTask(id);
    if (!task) throw new Error('task not found');
    if (task.status === nextStatus) return task;
    if (!(TRANSITIONS[task.status] || []).includes(nextStatus)) {
      throw new Error(`invalid task transition: ${task.status} -> ${nextStatus}`);
    }
    const previous = task.status;
    task.status = nextStatus;
    task.updatedAt = now();
    if (meta.nextAction != null) task.nextAction = String(meta.nextAction).slice(0, 500);
    if (meta.blockedReason != null) task.blockedReason = String(meta.blockedReason).slice(0, 1000);
    if (meta.plan && Array.isArray(meta.plan)) task.plan = meta.plan.slice(0, 40);
    this._save();
    this.journal?.append('task.transitioned', {
      taskId: task.id, previous, current: task.status, reason: meta.reason || '',
    }, { actor: meta.actor || 'system', correlationId: task.id });
    return task;
  }

  noteReplan(id) {
    const task = this.getTask(id);
    if (!task) throw new Error('task not found');
    task.recovery = {
      replans: (task.recovery?.replans || 0) + 1,
      lastReplanAt: now(),
    };
    task.updatedAt = now();
    this._save();
    return task;
  }

  confirm(id, approved, meta = {}) {
    const task = this.getTask(id);
    if (!task) throw new Error('task not found');
    if (task.status !== 'waiting_confirmation') throw new Error('task is not waiting for confirmation');
    task.confirmation = { approved: approved === true, ts: now(), note: String(meta.note || '').slice(0, 500) };
    this._save();
    return this.transition(id, approved === true ? 'ready' : 'cancelled', {
      actor: 'user', reason: approved === true ? 'approved' : 'rejected',
    });
  }

  addEvidence(id, evidence = {}) {
    const task = this.getTask(id);
    if (!task) throw new Error('task not found');
    const item = {
      id: makeId('evidence'),
      ts: now(),
      kind: String(evidence.kind || 'tool_result').slice(0, 80),
      ok: evidence.ok === true,
      summary: String(evidence.summary || '').trim().slice(0, 1200),
      artifact: String(evidence.artifact || '').trim().slice(0, 1200),
      capability: String(evidence.capability || '').trim().slice(0, 120),
      details: evidence.details && typeof evidence.details === 'object' ? evidence.details : null,
    };
    if (!item.summary) throw new Error('evidence summary required');
    task.evidence.push(item);
    task.updatedAt = now();
    this._save();
    this.journal?.append('task.evidence_added', { taskId: task.id, evidence: item }, {
      actor: 'executor', correlationId: task.id,
    });
    return item;
  }

  verifyTask(id, result = {}) {
    const task = this.getTask(id);
    if (!task) throw new Error('task not found');
    if (!['running', 'verifying'].includes(task.status)) throw new Error('task must be running or verifying');
    const passed = result.passed === true;
    if (passed && !task.evidence.some((item) => item.ok)) {
      throw new Error('successful verification requires successful execution evidence');
    }
    const previous = task.status;
    task.verification = {
      passed,
      ts: now(),
      summary: String(result.summary || '').trim().slice(0, 1200),
      checkedBy: String(result.checkedBy || 'system').slice(0, 100),
    };
    task.status = passed ? 'completed' : 'blocked';
    task.nextAction = passed ? '' : String(result.nextAction || '分析失败原因并重新规划').slice(0, 500);
    task.blockedReason = passed ? '' : task.verification.summary;
    task.updatedAt = now();
    this._save();
    this.journal?.append('task.verified', {
      taskId: task.id, previous, current: task.status, verification: task.verification,
    }, { actor: task.verification.checkedBy, correlationId: task.id });
    return task;
  }
}

module.exports = { ButlerTaskStore, TASK_STATES, TRANSITIONS };
