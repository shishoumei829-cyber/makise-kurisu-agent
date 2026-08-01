'use strict';

function parseJsonObject(raw) {
  const text = String(raw || '').trim();
  if (!text) return null;
  const clean = text.replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/i, '').trim();
  try { return JSON.parse(clean); } catch { /* try extracting one object */ }
  const start = clean.indexOf('{');
  const end = clean.lastIndexOf('}');
  if (start < 0 || end <= start) return null;
  try { return JSON.parse(clean.slice(start, end + 1)); } catch { return null; }
}

function parseDueAt(text, now = Date.now()) {
  const value = String(text || '');
  let match = value.match(/(\d+)\s*分钟后/);
  if (match) return now + Number(match[1]) * 60000;
  match = value.match(/(\d+(?:\.\d+)?)\s*小时后/);
  if (match) return now + Number(match[1]) * 3600000;
  match = value.match(/(\d+)\s*天后/);
  if (match) return now + Number(match[1]) * 86400000;

  match = value.match(/(今天|明天|后天)?\s*(?:上午|早上|下午|晚上)?\s*(\d{1,2})\s*[点时](?:(\d{1,2})\s*分?)?/);
  if (!match) return null;
  const d = new Date(now);
  const offset = match[1] === '后天' ? 2 : match[1] === '明天' ? 1 : 0;
  d.setDate(d.getDate() + offset);
  let hour = Number(match[2]);
  if (/下午|晚上/.test(match[0]) && hour < 12) hour += 12;
  d.setHours(hour, Number(match[3]) || 0, 0, 0);
  if (!match[1] && d.getTime() <= now) d.setDate(d.getDate() + 1);
  return d.getTime();
}

function reminderContent(text) {
  return String(text || '')
    .replace(/^(?:请|麻烦)?\s*(?:你)?\s*(?:到时候)?\s*(?:提醒我|记得提醒我|别忘了提醒我)/, '')
    .replace(/(?:\d+\s*分钟后|\d+(?:\.\d+)?\s*小时后|\d+\s*天后|(?:今天|明天|后天)?\s*(?:上午|早上|下午|晚上)?\s*\d{1,2}\s*[点时](?:\d{1,2}\s*分?)?)/g, '')
    .replace(/(?:能|可以)?(?:叫我起床|叫醒我|叫我起来)(?:吗|嘛)?/g, '起床')
    .replace(/(?:设|定)(?:个|一个)?闹钟/g, '闹钟')
    .replace(/^[，,。？?\s]+|[，,。？?\s]+$/g, '')
    .trim() || '提醒事项';
}

function fileQuery(text) {
  const quoted = String(text || '').match(/[“"']([^”"']{1,120})[”"']/);
  if (quoted) return quoted[1].trim();
  return String(text || '')
    .replace(/^(?:请|麻烦|帮我|替我|给我)?\s*/, '')
    .replace(/(?:在|从)?(?:电脑|本机|下载(?:文件夹|目录)?|文件夹|目录)?(?:里|中)?/g, '')
    .replace(/(?:找一下|找找|查找|搜索|找|查)(?:一个|一下)?/g, '')
    .replace(/文件|文档/g, '')
    .replace(/[，,。？?\s]+/g, ' ')
    .trim()
    .slice(0, 120);
}

function heuristicPlan(task, now = Date.now(), userContext = {}) {
  const text = String(task.description || task.title || '').trim();
  const appMap = [
    [/记事本/, 'notepad'],
    [/计算器/, 'calculator'],
    [/文件资源管理器|资源管理器/, 'explorer'],
    [/画图/, 'paint'],
    [/Windows\s*Terminal|终端/i, 'terminal'],
  ];
  if (task.type === 'system' && /打开|启动/.test(text)) {
    const app = appMap.find(([pattern]) => pattern.test(text));
    if (app) {
      return {
        source: 'heuristic', summary: `启动${app[1]}白名单应用。`, confidence: 0.98,
        needsClarification: false,
        steps: [{ id: 'step_1', capabilityId: 'app.launch', args: { appId: app[1] }, successCriteria: 'Windows 接受应用启动请求' }],
      };
    }
    const url = text.match(/https?:\/\/[^\s，,。]+/i)?.[0];
    if (url) {
      return {
        source: 'heuristic', summary: `通过默认浏览器打开 ${url}`, confidence: 0.98,
        needsClarification: false,
        steps: [{ id: 'step_1', capabilityId: 'desktop.open_target', args: { target: url }, successCriteria: 'Windows 接受打开请求' }],
      };
    }
    const alias = /下载/.test(text) ? 'downloads' : /文档/.test(text) ? 'documents' : /桌面/.test(text) ? 'desktop' : '';
    if (alias) {
      return {
        source: 'heuristic', summary: `打开${alias}目录。`, confidence: 0.96,
        needsClarification: false,
        steps: [{ id: 'step_1', capabilityId: 'desktop.open_target', args: { target: alias }, successCriteria: 'Windows 接受打开请求' }],
      };
    }
  }

  if (task.type === 'file' && /恢复|还原/.test(text)) {
    return {
      source: 'heuristic', summary: '恢复最近一次移入 Amadeus 可恢复区的文件。', confidence: 0.94,
      needsClarification: false,
      steps: [{ id: 'step_1', capabilityId: 'file.restore', args: { trashId: 'latest' }, successCriteria: '文件回到原路径且恢复区副本消失' }],
    };
  }

  if (task.type === 'file' && /创建|新建|写一个/.test(text)) {
    const name = text.match(/(?:叫(?:做|作)?|名为)\s*[“"']?([^”"'，,\s]{1,100})/i)?.[1]
      || text.match(/(?:创建|新建|写)(?:一个)?\s*([^，,\s]+\.(?:txt|md|json|csv))/i)?.[1];
    const content = text.match(/内容(?:是|为)[:：]?\s*([\s\S]+)$/)?.[1] || '';
    const directory = /桌面/.test(text) ? 'desktop'
      : /下载/.test(text) ? 'downloads'
        : /文档/.test(text) ? 'documents'
          : String(userContext.operationalPreferences?.defaultDirectory || '');
    if (!name || !directory) {
      return {
        source: 'heuristic', summary: '创建文件所需的位置或文件名不完整。', confidence: 0.94,
        needsClarification: true, clarificationQuestion: '请告诉我要在哪个目录创建、文件名是什么？', steps: [],
      };
    }
    return {
      source: 'heuristic', summary: `在${directory}创建新文本文件 ${name}，不覆盖已有文件。`, confidence: 0.96,
      needsClarification: false,
      steps: [{ id: 'step_1', capabilityId: 'file.create_text', args: { directory, name, content }, successCriteria: '文件存在且内容哈希一致' }],
    };
  }
  if (task.type === 'reminder') {
    const dueAt = parseDueAt(text, now);
    if (!dueAt) {
      return {
        source: 'heuristic',
        summary: '提醒时间不明确',
        confidence: 0.98,
        needsClarification: true,
        clarificationQuestion: '你希望我在什么时候提醒？',
        steps: [],
      };
    }
    return {
      source: 'heuristic',
      summary: '创建本地持久提醒，并重新读取确认保存成功。',
      confidence: 0.98,
      needsClarification: false,
      steps: [{
        id: 'step_1',
        capabilityId: 'reminder.create',
        args: { content: reminderContent(text), dueAt },
        successCriteria: '提醒已持久化且到期时间一致',
      }],
    };
  }

  if (task.type === 'system' && /(?:检查|看看|查看|获取).*(?:电脑|系统|运行|状态|内存)|(?:电脑|系统).*(?:状态|情况)/.test(text)) {
    return {
      source: 'heuristic',
      summary: '读取并验证本机运行状态。',
      confidence: 0.97,
      needsClarification: false,
      steps: [{ id: 'step_1', capabilityId: 'system.inspect', args: {}, successCriteria: '返回当前系统状态' }],
    };
  }

  if (task.type === 'file' && /找|查|搜索/.test(text)) {
    if (/昨天.*(?:下载|保存)|(?:下载|保存).*昨天/.test(text)) {
      const today = new Date(now);
      today.setHours(0, 0, 0, 0);
      const modifiedBefore = today.getTime();
      const modifiedAfter = modifiedBefore - 86400000;
      return {
        source: 'heuristic',
        summary: '在下载目录中按修改时间查找昨天保存的文件。',
        confidence: 0.94,
        needsClarification: false,
        steps: [{
          id: 'step_1', capabilityId: 'file.search',
          args: { query: '', modifiedAfter, modifiedBefore, maxResults: 50, sort: 'modified_desc' },
          successCriteria: '返回昨天时间范围内仍然存在的下载文件',
        }],
      };
    }
    const query = fileQuery(text);
    if (!query) {
      return {
        source: 'heuristic', summary: '缺少文件搜索关键词', confidence: 0.95,
        needsClarification: true, clarificationQuestion: '你要找的文件名或关键词是什么？', steps: [],
      };
    }
    return {
      source: 'heuristic',
      summary: `在允许的本地目录中搜索“${query}”。`,
      confidence: 0.9,
      needsClarification: false,
      steps: [{
        id: 'step_1', capabilityId: 'file.search', args: { query, maxResults: 30 },
        successCriteria: '返回仍然存在的匹配路径',
      }],
    };
  }
  return null;
}

function validatePlan(rawPlan, capabilities) {
  const ids = new Set((capabilities || []).filter((item) => item.available && item.executable).map((item) => item.id));
  const source = rawPlan && typeof rawPlan === 'object' ? rawPlan : {};
  const canExecute = source.canExecute !== false;
  const needsClarification = source.needsClarification === true;
  const steps = Array.isArray(source.steps) ? source.steps.slice(0, 8).map((step, index) => ({
    id: String(step?.id || `step_${index + 1}`).slice(0, 80),
    capabilityId: String(step?.capabilityId || '').trim(),
    args: step?.args && typeof step.args === 'object' && !Array.isArray(step.args) ? step.args : {},
    successCriteria: String(step?.successCriteria || '').trim().slice(0, 500),
  })) : [];
  if (canExecute && !needsClarification && !steps.length) throw new Error('planner returned no executable steps');
  for (const step of steps) {
    if (!ids.has(step.capabilityId)) throw new Error(`planner selected unavailable capability: ${step.capabilityId}`);
  }
  return {
    source: source.source === 'heuristic' ? 'heuristic' : 'work_brain',
    summary: String(source.summary || '').trim().slice(0, 1200),
    confidence: Math.max(0, Math.min(1, Number(source.confidence) || 0)),
    canExecute,
    blockedReason: String(source.blockedReason || '').trim().slice(0, 800),
    needsClarification,
    clarificationQuestion: String(source.clarificationQuestion || '').trim().slice(0, 500),
    steps,
  };
}

class ButlerPlanner {
  constructor(options = {}) {
    this.reasoner = typeof options.reasoner === 'function' ? options.reasoner : null;
  }

  async plan(task, capabilities, options = {}) {
    const local = heuristicPlan(task, options.now || Date.now(), options.userContext || {});
    if (local) return validatePlan(local, capabilities);
    if (options.allowStrong === false) return null;
    if (!this.reasoner) throw new Error('strong work brain is not configured for this task');
    const raw = await this.reasoner({
      task: {
        id: task.id,
        title: task.title,
        description: task.description,
        type: task.type,
        risk: task.risk,
      },
      capabilities: capabilities.filter((item) => item.available && item.executable),
      userContext: options.userContext || {},
      // 恢复循环：把上次失败的证据交给执行脑，要求换一条能避开失败原因的路径
      failureContext: Array.isArray(options.failureContext) ? options.failureContext.slice(0, 5) : [],
    });
    const parsed = typeof raw === 'object' ? raw : parseJsonObject(raw);
    if (!parsed) throw new Error('work brain returned invalid plan JSON');
    return validatePlan(parsed, capabilities);
  }
}

module.exports = {
  ButlerPlanner,
  parseJsonObject,
  parseDueAt,
  heuristicPlan,
  validatePlan,
};
