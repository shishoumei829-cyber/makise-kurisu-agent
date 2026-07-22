'use strict';

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { insideAllowedRoot } = require('./localCapabilities');

function resolveKnownDirectory(value, roots) {
  const alias = String(value || '').trim().toLowerCase();
  const home = roots.find((root) => path.basename(root).toLowerCase() === path.basename(require('os').homedir()).toLowerCase())
    || require('os').homedir();
  const aliases = {
    desktop: path.join(home, 'Desktop'),
    documents: path.join(home, 'Documents'),
    downloads: path.join(home, 'Downloads'),
  };
  const resolved = aliases[alias] || path.resolve(String(value || ''));
  if (!resolved || !insideAllowedRoot(resolved, roots)) throw new Error('target directory is outside allowed roots');
  if (!fs.existsSync(resolved) || !fs.statSync(resolved).isDirectory()) throw new Error('target directory does not exist');
  return resolved;
}

function safeFileName(value) {
  const name = String(value || '').trim();
  if (!name || name === '.' || name === '..' || /[\\/:*?"<>|]/.test(name)) throw new Error('invalid file name');
  return name.slice(0, 180);
}

function hashText(value) {
  return crypto.createHash('sha256').update(String(value), 'utf8').digest('hex');
}

function registerFileMutationCapabilities(kernel, options = {}) {
  const roots = (options.allowedRoots || []).filter(Boolean).map((item) => path.resolve(item)).filter((item) => fs.existsSync(item));

  kernel.registerCapability({
    id: 'file.create_text',
    name: '创建文本文件',
    available: roots.length > 0,
    risk: 'low',
    description: '在授权目录创建新的 UTF-8 文本文件；拒绝覆盖已有文件。',
    inputSchema: { directory: 'desktop|documents|downloads|absolute path', name: 'file name', content: 'UTF-8 text' },
  }, {
    execute: async (args) => {
      const directory = resolveKnownDirectory(args.directory, roots);
      const target = path.join(directory, safeFileName(args.name));
      if (!insideAllowedRoot(target, roots)) throw new Error('target file is outside allowed roots');
      if (fs.existsSync(target)) throw new Error('target already exists; refusing to overwrite');
      const content = String(args.content ?? '');
      const handle = fs.openSync(target, 'wx');
      try { fs.writeFileSync(handle, content, 'utf8'); } finally { fs.closeSync(handle); }
      return {
        ok: true,
        summary: `已创建文本文件 ${target}，${Buffer.byteLength(content, 'utf8')} 字节。`,
        artifact: target,
        data: { path: target, sha256: hashText(content), undo: { capabilityId: 'file.trash', args: { path: target } } },
      };
    },
    verify: async (result) => {
      const target = result.data?.path;
      if (!target || !fs.existsSync(target)) return { passed: false, summary: '创建后的文件不存在。' };
      const content = fs.readFileSync(target, 'utf8');
      const passed = hashText(content) === result.data.sha256;
      return { passed, summary: passed ? '已重新读取文件并核对内容哈希。' : '文件内容哈希不一致。' };
    },
  });

  kernel.registerCapability({
    id: 'file.trash',
    name: '移入 Amadeus 可恢复区',
    available: roots.length > 0,
    risk: 'high',
    description: '仅处理授权目录中的单个文件，移动到 Amadeus 可恢复区，不永久删除。',
    inputSchema: { path: 'absolute file path' },
  }, {
    execute: async (args, context) => {
      const target = path.resolve(String(args.path || ''));
      if (!insideAllowedRoot(target, roots)) throw new Error('source file is outside allowed roots');
      const entry = kernel.fileUndo.moveToTrash(target, context.task.id);
      return {
        ok: true,
        summary: `文件已移入可恢复区：${entry.originalPath}`,
        artifact: entry.id,
        data: { entry, undo: { capabilityId: 'file.restore', args: { trashId: entry.id } } },
      };
    },
    verify: async (result) => {
      const entry = result.data?.entry;
      const passed = Boolean(entry && !fs.existsSync(entry.originalPath) && fs.existsSync(entry.trashPath));
      return { passed, summary: passed ? '原路径已不存在，恢复副本存在。' : '未能确认文件安全移入恢复区。' };
    },
  });

  kernel.registerCapability({
    id: 'file.restore',
    name: '恢复文件',
    available: true,
    risk: 'low',
    description: '把 Amadeus 可恢复区中的文件恢复到原路径；拒绝覆盖。',
    inputSchema: { trashId: 'trash id or latest' },
  }, {
    execute: async (args, context) => {
      const entry = kernel.fileUndo.restore(String(args.trashId || 'latest'), context.task.id);
      return {
        ok: true,
        summary: `文件已恢复到 ${entry.originalPath}`,
        artifact: entry.originalPath,
        data: { entry },
      };
    },
    verify: async (result) => {
      const entry = result.data?.entry;
      const passed = Boolean(entry && fs.existsSync(entry.originalPath) && !fs.existsSync(entry.trashPath));
      return { passed, summary: passed ? '已确认文件回到原路径。' : '恢复结果未通过检查。' };
    },
  });
}

module.exports = { registerFileMutationCapabilities, resolveKnownDirectory, safeFileName };
