'use strict';

const NATIVE_CAPABILITIES = [
  {
    id: 'speech.reply',
    name: '当下对话回应',
    effector: 'speech',
    tier: 'native',
    available: true,
    risk: 'low',
    examPass: true,
    describeForSelf: '说话、拒绝、追问',
  },
  {
    id: 'speech.initiative',
    name: '有动机主动开口',
    effector: 'speech',
    tier: 'native',
    available: true,
    risk: 'low',
    examPass: true,
    describeForSelf: '主动开口',
  },
  {
    id: 'speech.deferred',
    name: '到期用自己的嘴开口',
    effector: 'speech',
    tier: 'native',
    available: true,
    risk: 'low',
    examPass: true,
    describeForSelf: '到点开口',
  },
  {
    id: 'commitment.track',
    name: '登记与兑现承诺',
    effector: 'commitment',
    tier: 'native',
    available: true,
    risk: 'low',
    examPass: true,
    describeForSelf: '记住并兑现约定',
  },
  {
    id: 'memory.retain',
    name: '记住事实与约定',
    effector: 'memory',
    tier: 'native',
    available: true,
    risk: 'low',
    examPass: true,
    describeForSelf: '记忆',
  },
  {
    id: 'time.schedule',
    name: '进程内等到某时刻',
    effector: 'time_awareness',
    tier: 'native',
    available: true,
    risk: 'low',
    examPass: true,
    describeForSelf: '时间感与到点调度',
  },
];

/**
 * @param {{ toolCaps?: Array<object> }} options
 */
function buildCapabilitySnapshot(options = {}) {
  const toolCaps = Array.isArray(options.toolCaps) ? options.toolCaps : [];
  const native = NATIVE_CAPABILITIES.map((c) => ({ ...c }));
  const augmented = toolCaps.map((c) => ({
    id: c.id,
    name: c.name || c.id,
    effector: guessEffector(c.id),
    tier: c.available === false && /web\./.test(c.id) ? 'experimental' : 'augmented',
    available: c.available === true,
    risk: c.risk || 'low',
    examPass: c.examPass !== false && c.available === true,
    describeForSelf: c.description || c.name || c.id,
    executable: c.executable === true,
  }));
  return { native, augmented, forbiddenNote: 'user_body / 上门跑腿 / 物理操纵' };
}

function guessEffector(id) {
  const s = String(id || '');
  if (s.startsWith('file.')) return 'local_files';
  if (s.startsWith('app.') || s.startsWith('desktop.')) return 'local_apps';
  if (s.startsWith('system.')) return 'local_system_info';
  if (s.startsWith('web.')) return 'web';
  if (s.startsWith('reminder.')) return 'local_runtime';
  return 'local_runtime';
}

function formatSnapshotForPrompt(snapshot) {
  const native = (snapshot?.native || [])
    .filter((c) => c.available)
    .map((c) => c.describeForSelf || c.id)
    .slice(0, 8);
  const available = (snapshot?.augmented || [])
    .filter((c) => c.available)
    .map((c) => c.id)
    .slice(0, 12);
  const unavailable = (snapshot?.augmented || [])
    .filter((c) => !c.available)
    .map((c) => c.id)
    .slice(0, 8);
  const lines = [
    '【此刻能力】',
    `原生：${native.join(' / ') || '说话 / 记约定 / 到点开口'}`,
  ];
  if (available.length) lines.push(`本机可用：${available.join(', ')}`);
  if (unavailable.length) lines.push(`本机不可用/实验：${unavailable.join(', ')}`);
  lines.push(`禁止：${snapshot?.forbiddenNote || '身体接触、上门跑腿'}`);
  return lines.join('\n');
}

module.exports = {
  NATIVE_CAPABILITIES,
  buildCapabilitySnapshot,
  formatSnapshotForPrompt,
  guessEffector,
};
