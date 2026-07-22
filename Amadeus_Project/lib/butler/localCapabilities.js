'use strict';

const fs = require('fs');
const path = require('path');

function insideAllowedRoot(target, roots) {
  const normalized = path.resolve(target).toLowerCase();
  return roots.some((root) => {
    const base = path.resolve(root).toLowerCase();
    return normalized === base || normalized.startsWith(`${base}${path.sep}`);
  });
}

function searchFiles(args, roots) {
  const query = String(args.query || '').trim().toLowerCase();
  const modifiedAfter = Number(args.modifiedAfter) || 0;
  const modifiedBefore = Number(args.modifiedBefore) || 0;
  if (!query && !modifiedAfter && !modifiedBefore) throw new Error('file search query or time range required');
  const defaultRoot = roots.find((root) => /downloads$/i.test(root)) || roots[0];
  const root = path.resolve(String(args.root || defaultRoot));
  if (!insideAllowedRoot(root, roots)) throw new Error('search root is outside allowed roots');
  if (!fs.existsSync(root)) throw new Error('search root does not exist');
  const maxResults = Math.min(100, Math.max(1, Number(args.maxResults) || 30));
  const maxDepth = Math.min(8, Math.max(1, Number(args.maxDepth) || 5));
  const results = [];
  const queue = [{ dir: root, depth: 0 }];
  let scanned = 0;
  while (queue.length && results.length < maxResults && scanned < 20000) {
    const { dir, depth } = queue.shift();
    let entries = [];
    try { entries = fs.readdirSync(dir, { withFileTypes: true }); } catch { continue; }
    for (const entry of entries) {
      if (results.length >= maxResults || scanned >= 20000) break;
      scanned += 1;
      if (entry.name.startsWith('.') || ['node_modules', '$Recycle.Bin', 'System Volume Information'].includes(entry.name)) continue;
      const fullPath = path.join(dir, entry.name);
      const nameMatch = !query || entry.name.toLowerCase().includes(query);
      let stat = null;
      if (nameMatch && (modifiedAfter || modifiedBefore || args.sort === 'modified_desc')) {
        try { stat = fs.statSync(fullPath); } catch { stat = null; }
      }
      const modifiedAt = stat?.mtimeMs || 0;
      const timeMatch = (!modifiedAfter || modifiedAt >= modifiedAfter)
        && (!modifiedBefore || modifiedAt < modifiedBefore);
      if (nameMatch && timeMatch) {
        results.push({
          path: fullPath,
          name: entry.name,
          kind: entry.isDirectory() ? 'directory' : 'file',
          modifiedAt: modifiedAt || null,
        });
      }
      if (entry.isDirectory() && !entry.isSymbolicLink() && depth < maxDepth) queue.push({ dir: fullPath, depth: depth + 1 });
    }
  }
  if (args.sort === 'modified_desc') results.sort((a, b) => (b.modifiedAt || 0) - (a.modifiedAt || 0));
  return { root, query, results, scanned, truncated: queue.length > 0 || scanned >= 20000 };
}

function registerLocalCapabilities(kernel, options = {}) {
  const roots = [...new Set((options.allowedRoots || [])
    .filter(Boolean)
    .map((item) => path.resolve(item))
    .filter((item) => fs.existsSync(item)))];
  kernel.registerCapability({
    id: 'file.search',
    name: '本地文件搜索',
    available: roots.length > 0,
    risk: 'low',
    description: '只读搜索已授权目录，不读取文件正文。',
  }, {
    execute: async (args) => {
      const output = searchFiles(args, roots);
      return {
        ok: true,
        summary: `在 ${output.root} 扫描 ${output.scanned} 项，找到 ${output.results.length} 个匹配结果。`,
        artifact: output.results.slice(0, 20).map((item) => item.path).join('\n'),
        data: output,
      };
    },
    verify: async (result) => {
      const matches = result.data?.results || [];
      const existing = matches.filter((item) => fs.existsSync(item.path));
      return {
        passed: existing.length === matches.length,
        summary: `重新检查 ${matches.length} 个结果，当前存在 ${existing.length} 个。`,
      };
    },
  });

  kernel.registerCapability({
    id: 'reminder.create',
    name: '创建本地提醒',
    available: Boolean(kernel.reminders),
    risk: 'low',
    description: '把提醒持久化到本机，重启后仍然存在。',
  }, {
    execute: async (args, context) => {
      const reminder = kernel.reminders.create({ ...args, taskId: context.task.id });
      return {
        ok: true,
        summary: `提醒已保存：${reminder.content}，时间=${new Date(reminder.dueAt).toLocaleString('zh-CN')}`,
        artifact: reminder.id,
        data: { reminder },
      };
    },
    verify: async (result) => {
      const id = result.data?.reminder?.id;
      const saved = id ? kernel.reminders.get(id) : null;
      return {
        passed: Boolean(saved && saved.status === 'scheduled' && saved.dueAt === result.data.reminder.dueAt),
        summary: saved ? '已从持久化提醒库重新读取并核对时间。' : '提醒未能从持久化存储重新读取。',
      };
    },
  });
}

module.exports = { registerLocalCapabilities, searchFiles, insideAllowedRoot };
