'use strict';

const fs = require('fs');
const path = require('path');

class ButlerUserWorld {
  constructor(dataDir, journal) {
    this.dir = path.join(dataDir, 'butler');
    this.path = path.join(this.dir, 'user_world.json');
    this.journal = journal;
    this.contextProvider = null;
    fs.mkdirSync(this.dir, { recursive: true });
    this.state = this._load();
  }

  _empty() {
    return {
      version: 1,
      updatedAt: Date.now(),
      operationalPreferences: {
        defaultDirectory: '',
        defaultApp: '',
      },
      explicitRules: [],
      taskOutcomes: {
        completed: 0,
        blocked: 0,
        failed: 0,
        byCapability: {},
      },
    };
  }

  _load() {
    try {
      const parsed = JSON.parse(fs.readFileSync(this.path, 'utf8'));
      const base = this._empty();
      return {
        ...base,
        ...parsed,
        operationalPreferences: { ...base.operationalPreferences, ...(parsed.operationalPreferences || {}) },
        taskOutcomes: {
          ...base.taskOutcomes,
          ...(parsed.taskOutcomes || {}),
          byCapability: { ...(parsed.taskOutcomes?.byCapability || {}) },
        },
        explicitRules: Array.isArray(parsed.explicitRules) ? parsed.explicitRules : [],
      };
    } catch { return this._empty(); }
  }

  _save() {
    this.state.updatedAt = Date.now();
    const temp = `${this.path}.tmp`;
    fs.writeFileSync(temp, JSON.stringify(this.state, null, 2), 'utf8');
    fs.renameSync(temp, this.path);
  }

  setContextProvider(provider) {
    this.contextProvider = typeof provider === 'function' ? provider : null;
  }

  observeUserStatement(text) {
    const value = String(text || '').trim();
    if (!value || !/(?:以后|默认|总是|每次|不要|别再|习惯)/.test(value)) return null;
    let changed = false;
    const directory = value.match(/(?:以后|默认).{0,20}(?:文件|东西|结果|输出).{0,12}(?:放|存|保存|写入).{0,8}(桌面|文档|下载)/)?.[1];
    if (directory) {
      const aliases = { 桌面: 'desktop', 文档: 'documents', 下载: 'downloads' };
      this.state.operationalPreferences.defaultDirectory = aliases[directory];
      changed = true;
    }
    const app = value.match(/(?:默认|以后).{0,12}(?:用|使用)(记事本|计算器|画图|终端)/)?.[1];
    if (app) {
      const apps = { 记事本: 'notepad', 计算器: 'calculator', 画图: 'paint', 终端: 'terminal' };
      this.state.operationalPreferences.defaultApp = apps[app];
      changed = true;
    }
    if (!this.state.explicitRules.some((item) => item.text === value)) {
      this.state.explicitRules.push({ text: value.slice(0, 300), createdAt: Date.now() });
      this.state.explicitRules = this.state.explicitRules.slice(-40);
      changed = true;
    }
    if (!changed) return null;
    this._save();
    this.journal?.append('user_world.preference_updated', {
      rule: value.slice(0, 300),
      operationalPreferences: this.state.operationalPreferences,
    }, { actor: 'user', correlationId: 'user_world' });
    return this.snapshot();
  }

  observeTaskOutcome(task) {
    if (!task || !['completed', 'blocked', 'failed'].includes(task.status)) return;
    const outcomes = this.state.taskOutcomes;
    outcomes[task.status] = (outcomes[task.status] || 0) + 1;
    for (const evidence of task.evidence || []) {
      const id = String(evidence.capability || '').trim();
      if (!id) continue;
      const row = outcomes.byCapability[id] || { success: 0, failure: 0 };
      if (evidence.ok) row.success += 1;
      else row.failure += 1;
      outcomes.byCapability[id] = row;
    }
    this._save();
  }

  snapshot() {
    let external = {};
    try { external = this.contextProvider?.() || {}; } catch { external = {}; }
    const userModel = external.userModel || {};
    const topics = Object.entries(userModel.preferences?.topics || {})
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
      .map(([topic]) => topic);
    return {
      operationalPreferences: { ...this.state.operationalPreferences },
      explicitRules: this.state.explicitRules.slice(-12).map((item) => item.text),
      interactionTendencies: {
        responseLength: userModel.preferences?.response_length || 'medium',
        topTopics: topics,
      },
      taskOutcomes: JSON.parse(JSON.stringify(this.state.taskOutcomes)),
      privacy: {
        containsIdentity: false,
        containsRawDialogue: false,
        note: '身份、姓名、基础个人资料和原始对话不会发送给执行规划器。',
      },
    };
  }

  toPlannerContext() {
    return this.snapshot();
  }
}

module.exports = { ButlerUserWorld };
