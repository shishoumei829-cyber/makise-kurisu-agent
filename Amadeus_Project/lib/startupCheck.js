'use strict';

const DEFAULT_CHAT_MODEL = 'kurisu:latest';
const DEFAULT_EMBED_MODEL = 'nomic-embed-text';

function summarizeHealth(checks) {
  const blockers = checks.filter((c) => c.required && !c.ok);
  const warnings = checks.filter((c) => !c.required && !c.ok);
  return { ready: blockers.length === 0, blockers, warnings, checks };
}

async function probeOllama({ ollamaBase, fetchFn = fetch, timeoutMs = 5000 }) {
  const base = String(ollamaBase || '').replace(/\/$/, '');
  try {
    const r = await fetchFn(`${base}/api/tags`, { signal: AbortSignal.timeout(timeoutMs) });
    if (!r.ok) return { ok: false, error: `HTTP ${r.status}`, models: [] };
    const j = await r.json();
    const models = (j.models || []).map((m) => m.name || m.model).filter(Boolean);
    return { ok: true, models };
  } catch (e) {
    return { ok: false, error: String(e.message || e), models: [] };
  }
}

function hasModel(models, name) {
  const target = String(name || '').trim();
  if (!target) return false;
  const base = target.split(':')[0];
  return models.some((m) => {
    const listed = String(m || '');
    return listed === target || listed.startsWith(`${target}:`) || listed.split(':')[0] === base;
  });
}

function buildUserHints(summary) {
  const hints = [];
  for (const c of summary.blockers) {
    if (c.id === 'ollama') hints.push('请先启动 Ollama（运行 ollama serve 或打开 Ollama 应用）');
    else if (c.id === 'chat_model') hints.push(`缺少对话模型，请执行 ollama pull 并确保 ${DEFAULT_CHAT_MODEL} 可用`);
    else hints.push(c.message);
  }
  for (const c of summary.warnings) {
    if (c.id === 'rag') hints.push('长期记忆索引未建立：在项目目录运行 npm run ingest（不影响基础聊天）');
    else if (c.id === 'embed_model') hints.push(`RAG 需要嵌入模型 ${DEFAULT_EMBED_MODEL}（可选）`);
    else if (c.id === 'tts') hints.push('语音服务未连接（可选，文字对话不受影响）');
  }
  return hints;
}

async function runStartupChecks(opts = {}) {
  const ollamaBase = opts.ollamaBase || 'http://127.0.0.1:11434';
  const chatModel = opts.chatModel || DEFAULT_CHAT_MODEL;
  const embedModel = opts.embedModel || DEFAULT_EMBED_MODEL;
  const ragIndexed = !!opts.ragIndexed;
  const sovitsUrl = String(opts.sovitsUrl || 'http://localhost:9880').replace(/\/$/, '');
  const fetchFn = opts.fetchFn || fetch;

  const checks = [];

  const ollama = await probeOllama({ ollamaBase, fetchFn });
  checks.push({
    id: 'ollama',
    label: 'Ollama',
    required: true,
    ok: ollama.ok,
    message: ollama.ok ? 'Ollama 已连接' : `Ollama 未就绪：${ollama.error}`,
  });

  const chatOk = ollama.ok && hasModel(ollama.models, chatModel);
  checks.push({
    id: 'chat_model',
    label: '对话模型',
    required: true,
    ok: chatOk,
    message: chatOk ? `对话模型 ${chatModel} 可用` : `缺少对话模型 ${chatModel}`,
  });

  const embedOk = ollama.ok && hasModel(ollama.models, embedModel);
  checks.push({
    id: 'embed_model',
    label: '嵌入模型',
    required: false,
    ok: embedOk,
    message: embedOk ? `嵌入模型 ${embedModel} 可用` : `嵌入模型 ${embedModel} 未安装（RAG 可选）`,
  });

  checks.push({
    id: 'rag',
    label: 'RAG 索引',
    required: false,
    ok: ragIndexed,
    message: ragIndexed ? '记忆索引已就绪' : '尚未建立 RAG 索引（可选）',
  });

  let ttsOk = false;
  try {
    const r = await fetchFn(`${sovitsUrl}/`, { signal: AbortSignal.timeout(3000) });
    ttsOk = r.status >= 200 && r.status < 500;
  } catch (_) { /* optional */ }

  checks.push({
    id: 'tts',
    label: 'TTS',
    required: false,
    ok: ttsOk,
    message: ttsOk ? '语音服务已连接' : '语音服务未连接（可选）',
  });

  const summary = summarizeHealth(checks);
  return {
    ok: summary.ready,
    ready: summary.ready,
    checks: summary.checks,
    blockers: summary.blockers.map((c) => ({ id: c.id, message: c.message })),
    warnings: summary.warnings.map((c) => ({ id: c.id, message: c.message })),
    hints: buildUserHints(summary),
    chatModel,
    ollamaBase,
  };
}

module.exports = {
  DEFAULT_CHAT_MODEL,
  DEFAULT_EMBED_MODEL,
  probeOllama,
  hasModel,
  summarizeHealth,
  buildUserHints,
  runStartupChecks,
};
