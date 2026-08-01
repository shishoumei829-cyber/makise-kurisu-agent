'use strict';

const { parseDueAt } = require('../butler/planner');
const { extractDeferredSpeak, scheduleContentFromRequest } = require('../../cognitive/deferredSpeak');

const FORBIDDEN_RE = /(?:帮我|替我|给我)?(?:去)?(?:买|拿|带|送|取|接|跑腿|上门).{0,12}(?:咖啡|茶|外卖|东西|快递)|(?:伸手)?(?:推醒|摇醒)|(?:去|到)(?:你家|他那儿|现场)/;

const DIRECTED_AT_HER_RE = /(?:叫|喊|提醒|告诉|通知|记得|别忘|盯|回来找我|问我|说一声|开口)|(?:能|可以|帮|替|给).{0,8}(?:我|一下)|你(?:到时候|到点)|(?:设|定)(?:个|一个)?闹钟|(?:打开|启动|关闭|查找|搜索|检查).{0,20}(?:记事本|计算器|资源管理器|文件|系统|电脑)|记住/;

const APP_MAP = [
  [/记事本/, 'notepad', '打开记事本'],
  [/计算器/, 'calculator', '打开计算器'],
  [/文件资源管理器|资源管理器/, 'explorer', '打开资源管理器'],
  [/画图/, 'paint', '打开画图'],
  [/Windows\s*Terminal|终端/i, 'terminal', '打开终端'],
];

/**
 * 能力判决：forbidden / native / augmented / none / clarify
 * @returns {{
 *   verdict: string,
 *   goal?: string,
 *   trigger?: object,
 *   effectors?: string[],
 *   plan?: object[],
 *   alternative?: string,
 *   reason?: string,
 * }}
 */
function judgeRequest(text, options = {}) {
  const raw = String(text || '').trim();
  if (!raw) return { verdict: 'none', reason: 'empty' };
  const now = Number(options.now) || Date.now();

  if (FORBIDDEN_RE.test(raw)) {
    return {
      verdict: 'forbidden',
      goal: raw.slice(0, 120),
      reason: 'physical_world_manipulation',
      alternative: '我没有身体去跑腿或推醒你；能做的是记住约定、到点开口喊你，或用本机工具增强。',
      effectors: [],
    };
  }

  const app = matchAppLaunch(raw);
  if (app) {
    return {
      verdict: 'augmented',
      goal: app.goal,
      trigger: { kind: 'immediate' },
      effectors: ['app.launch'],
      plan: [{
        capabilityId: 'app.launch',
        args: { appId: app.appId },
        successCriteria: 'Windows 接受应用启动请求',
      }],
      category: 'system',
      reason: 'local_app_launch',
    };
  }

  if (/检查|查看|看看/.test(raw) && /电脑|系统|运行状态|内存|负载/.test(raw)) {
    return {
      verdict: 'augmented',
      goal: '检查本机运行状态',
      trigger: { kind: 'immediate' },
      effectors: ['system.inspect'],
      plan: [{
        capabilityId: 'system.inspect',
        args: {},
        successCriteria: '返回系统状态摘要',
      }],
      category: 'system',
      reason: 'system_inspect',
    };
  }

  if (/(?:找|搜索|查找).{0,20}文件/.test(raw) || /文件.{0,12}(?:找|搜)/.test(raw)) {
    const q = raw
      .replace(/^(?:请|麻烦|帮我|替我|给我)?\s*/, '')
      .replace(/(?:找一下|找找|查找|搜索|找).{0,4}/, '')
      .replace(/文件|文档/g, '')
      .trim()
      .slice(0, 80) || '文件';
    return {
      verdict: 'augmented',
      goal: `查找文件：${q}`,
      trigger: { kind: 'immediate' },
      effectors: ['file.search'],
      plan: [{
        capabilityId: 'file.search',
        args: { query: q },
        successCriteria: '返回匹配文件列表或明确未找到',
      }],
      category: 'file',
      reason: 'file_search',
    };
  }

  const deferred = extractDeferredSpeak(raw, now);
  if (deferred) {
    return {
      verdict: 'native',
      goal: deferred.content,
      trigger: { kind: 'at_time', dueAt: deferred.dueAt, windowMs: 15 * 60 * 1000 },
      effectors: ['speech.deferred', 'time.schedule', 'commitment.track'],
      speakHint: deferred.content,
      category: 'deferred_speak',
      reason: 'native_deferred_speak',
    };
  }

  // 有时间但不够「对她作为」——避免把「明天有课」建成意图
  const dueAt = parseDueAt(raw, now);
  if (dueAt && dueAt > now && DIRECTED_AT_HER_RE.test(raw)) {
    return {
      verdict: 'native',
      goal: scheduleContentFromRequest(raw),
      trigger: { kind: 'at_time', dueAt, windowMs: 15 * 60 * 1000 },
      effectors: ['speech.deferred', 'time.schedule', 'commitment.track'],
      speakHint: scheduleContentFromRequest(raw),
      category: 'deferred_speak',
      reason: 'native_timed_directed',
    };
  }

  if (/^(?:记住|帮我记|记下|别忘了记)/.test(raw) || /请记住/.test(raw)) {
    return {
      verdict: 'native',
      goal: raw.replace(/^(?:请)?(?:记住|帮我记|记下|别忘了记)\s*/, '').slice(0, 200) || '记住这件事',
      trigger: { kind: 'immediate' },
      effectors: ['memory.retain', 'commitment.track'],
      category: 'memory',
      reason: 'native_remember',
    };
  }

  if (dueAt && dueAt > now && !DIRECTED_AT_HER_RE.test(raw)) {
    return { verdict: 'none', reason: 'schedule_fact_only' };
  }

  return { verdict: 'none', reason: 'conversation' };
}

function matchAppLaunch(text) {
  if (!/(?:打开|启动|开一下|帮我开)/.test(text)) return null;
  for (const [re, appId, goal] of APP_MAP) {
    if (re.test(text)) return { appId, goal };
  }
  return null;
}

module.exports = {
  judgeRequest,
  matchAppLaunch,
  FORBIDDEN_RE,
  DIRECTED_AT_HER_RE,
};
