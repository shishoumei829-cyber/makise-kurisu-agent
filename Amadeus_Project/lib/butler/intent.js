'use strict';

const ACTION_PREFIX = /^(?:请|麻烦|帮我|替我|给我|记得|提醒我|别忘了|打开|关闭|启动|停止|查找|找一下|搜索|查一下|整理|创建|新建|写|修改|修复|检查|运行|执行|下载|安装|删除|恢复|还原|移动|发送|发布|提交|预约|安排)/;
const ACTION_ANYWHERE = /(?:帮我|替我|提醒我|给我查|给我找|需要你|交给你|你来处理|你去处理|你来做|你去做)/;

function extractUserText(body = {}) {
  if (typeof body === 'string') return body.trim();
  const messages = Array.isArray(body.messages) ? body.messages : [];
  const lastUser = [...messages].reverse().find((item) => item && item.role === 'user');
  return String(lastUser?.content || body.userMsg || body.message || body.prompt || '').trim();
}

function classifyButlerIntent(text) {
  const value = String(text || '').trim();
  if (!value) return { actionable: false, category: 'empty', confidence: 1, risk: 'low' };
  const actionable = ACTION_PREFIX.test(value) || ACTION_ANYWHERE.test(value);
  if (!actionable) return { actionable: false, category: 'conversation', confidence: 0.82, risk: 'low' };

  let category = 'general';
  if (/提醒|别忘|闹钟|到点|日程|安排/.test(value)) category = 'reminder';
  else if (/搜索|查一下|调查|研究|资料|比较/.test(value)) category = 'research';
  else if (/文件|文件夹|目录|下载|文档|表格|图片|视频|\.(?:txt|md|json|csv)\b/i.test(value)) category = 'file';
  else if (/打开|关闭|启动|停止|软件|应用|音量|电脑|系统/.test(value)) category = 'system';
  else if (/写|修改|修复|代码|项目|测试|运行/.test(value)) category = 'project';

  const destructive = /删除|清空|覆盖|卸载|格式化|永久|销毁/.test(value);
  const externalImpact = /发送|发布|提交|购买|付款|预约|投递|上传/.test(value);
  const sensitive = /密码|密钥|账号|隐私|身份证|银行卡/.test(value);
  const risk = destructive || externalImpact || sensitive ? 'high' : /移动|重命名|关闭|停止/.test(value) ? 'medium' : 'low';

  return {
    actionable: true,
    category,
    confidence: ACTION_PREFIX.test(value) ? 0.94 : 0.86,
    risk,
    requiresConfirmation: risk === 'high',
    title: value.slice(0, 180),
  };
}

module.exports = { extractUserText, classifyButlerIntent };
