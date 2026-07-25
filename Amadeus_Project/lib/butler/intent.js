'use strict';

/** 她决定动手时写入的结构化标记；没有这一行就不进管家队列。 */
const TASK_MARKER_RE = /⟦\s*AMADEUS_TASK\s*([\s\S]*?)⟧/i;

function extractUserText(body = {}) {
  if (typeof body === 'string') return body.trim();
  const messages = Array.isArray(body.messages) ? body.messages : [];
  const lastUser = [...messages].reverse().find((item) => item && item.role === 'user');
  return String(lastUser?.content || body.userMsg || body.message || body.prompt || body.text || '').trim();
}

/** 元数据推断：只描述「已决定要办的事」，不当作进队开关。 */
function inferTaskMeta(description) {
  const value = String(description || '').trim();
  let category = 'general';
  if (/提醒|别忘|闹钟|到点|日程|安排/.test(value)) category = 'reminder';
  else if (/搜索|查一下|调查|研究|资料|比较/.test(value)) category = 'research';
  else if (/文件|文件夹|目录|下载|文档|表格|图片|视频|\.(?:txt|md|json|csv)\b/i.test(value)) category = 'file';
  else if (/打开|关闭|启动|停止|软件|应用|音量|电脑|系统|记事本|计算器/.test(value)) category = 'system';
  else if (/写|修改|修复|代码|项目|测试|运行/.test(value)) category = 'project';

  const destructive = /删除|清空|覆盖|卸载|格式化|永久|销毁/.test(value);
  const externalImpact = /发送|发布|提交|购买|付款|预约|投递|上传/.test(value);
  const sensitive = /密码|密钥|账号|隐私|身份证|银行卡/.test(value);
  const risk = destructive || externalImpact || sensitive
    ? 'high'
    : /移动|重命名|关闭|停止/.test(value) ? 'medium' : 'low';

  return {
    category,
    risk,
    requiresConfirmation: risk === 'high',
  };
}

function _parseProposalBody(body) {
  const raw = String(body || '').trim();
  if (!raw) return null;
  if (raw.startsWith('{')) {
    try {
      const data = JSON.parse(raw);
      if (!data || typeof data !== 'object') return null;
      return data;
    } catch {
      return null;
    }
  }
  const data = {};
  for (const line of raw.split(/\r?\n/)) {
    const m = line.match(/^\s*([A-Za-z_\u4e00-\u9fff]+)\s*[:：]\s*(.+?)\s*$/);
    if (!m) continue;
    const key = m[1].toLowerCase();
    const mapped = key === '标题' ? 'title'
      : key === '描述' || key === '内容' ? 'description'
        : key === '类型' || key === '分类' ? 'category'
          : key;
    data[mapped] = m[2].trim();
  }
  return Object.keys(data).length ? data : null;
}

/**
 * 从红莉栖回复里提取动手提案，并剥掉标记得到可说出台词。
 * 进队与否只看有没有合法标记，不看用户原句里有没有「帮我/给我」。
 */
function parseAgencyTaskProposal(rawText) {
  const text = String(rawText || '');
  const match = text.match(TASK_MARKER_RE);
  if (!match) {
    return { proposal: null, spoken: text.trim() };
  }
  const data = _parseProposalBody(match[1]);
  const spoken = text.replace(TASK_MARKER_RE, '').replace(/\n{3,}/g, '\n\n').trim();
  const description = String(data?.description || data?.title || data?.task || '').trim();
  if (!description) return { proposal: null, spoken };

  const meta = inferTaskMeta(description);
  const category = String(data.category || data.type || meta.category || 'general').trim() || 'general';
  const risk = ['low', 'medium', 'high'].includes(String(data.risk || '').toLowerCase())
    ? String(data.risk).toLowerCase()
    : meta.risk;

  return {
    proposal: {
      title: String(data.title || description).trim().slice(0, 180),
      description: description.slice(0, 2000),
      category,
      risk,
      requiresConfirmation: risk === 'high' || data.requiresConfirmation === true,
    },
    spoken,
  };
}

function stripAgencyTaskMarkers(text) {
  return String(text || '')
    .replace(TASK_MARKER_RE, '')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

/** @deprecated 词表门已废弃；保留空壳避免外部误用再悄悄进队。 */
function classifyButlerIntent() {
  return { actionable: false, category: 'conversation', confidence: 1, risk: 'low', gatedBy: 'agency_only' };
}

module.exports = {
  extractUserText,
  inferTaskMeta,
  parseAgencyTaskProposal,
  stripAgencyTaskMarkers,
  classifyButlerIntent,
  TASK_MARKER_RE,
};
