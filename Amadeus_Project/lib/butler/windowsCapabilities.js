'use strict';

const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');
const { insideAllowedRoot } = require('./localCapabilities');

const APPS = Object.freeze({
  notepad: { exe: 'notepad.exe', label: '记事本' },
  calculator: { exe: 'calc.exe', label: '计算器' },
  explorer: { exe: 'explorer.exe', label: '文件资源管理器' },
  paint: { exe: 'mspaint.exe', label: '画图' },
  terminal: { exe: 'wt.exe', label: 'Windows Terminal' },
});

function launchDetached(exe, args = []) {
  const child = spawn(exe, args, { detached: true, stdio: 'ignore', windowsHide: false, shell: false });
  const pid = child.pid;
  child.unref();
  if (!Number.isInteger(pid) || pid <= 0) throw new Error('Windows did not accept the launch request');
  return pid;
}

function registerWindowsCapabilities(kernel, options = {}) {
  const roots = (options.allowedRoots || []).filter(Boolean).map((item) => path.resolve(item)).filter((item) => fs.existsSync(item));
  const launcher = options.launcher || launchDetached;

  kernel.registerCapability({
    id: 'app.launch',
    name: '启动受信任应用',
    available: process.platform === 'win32' || options.allowTestPlatform === true,
    risk: 'low',
    description: `只允许启动白名单应用：${Object.keys(APPS).join(', ')}`,
    inputSchema: { appId: Object.keys(APPS) },
  }, {
    execute: async (args) => {
      const app = APPS[String(args.appId || '').trim().toLowerCase()];
      if (!app) throw new Error('application is not in the allowlist');
      const pid = launcher(app.exe, []);
      return {
        ok: true,
        summary: `Windows 已接受启动 ${app.label} 的请求。`,
        artifact: `process://${pid}`,
        data: { pid, appId: args.appId, executable: app.exe },
      };
    },
    verify: async (result) => ({
      passed: Number.isInteger(result.data?.pid) && result.data.pid > 0,
      summary: '已获得 Windows 创建进程的 PID；这证明启动请求被接受，不代表应用内部操作已完成。',
    }),
  });

  kernel.registerCapability({
    id: 'desktop.open_target',
    name: '打开网页或本地路径',
    available: process.platform === 'win32' || options.allowTestPlatform === true,
    risk: 'low',
    description: '通过 Windows 打开 http/https 网页，或打开授权目录中的现有本地路径。',
    inputSchema: { target: 'http(s) URL | downloads | documents | desktop | allowed absolute path' },
  }, {
    execute: async (args) => {
      const raw = String(args.target || '').trim();
      const home = require('os').homedir();
      const aliases = {
        downloads: path.join(home, 'Downloads'),
        documents: path.join(home, 'Documents'),
        desktop: path.join(home, 'Desktop'),
      };
      let target = aliases[raw.toLowerCase()] || raw;
      if (!/^https?:\/\//i.test(target)) {
        target = path.resolve(target);
        if (!insideAllowedRoot(target, roots)) throw new Error('target path is outside allowed roots');
        if (!fs.existsSync(target)) throw new Error('target path does not exist');
      }
      const pid = launcher('explorer.exe', [target]);
      return {
        ok: true,
        summary: `Windows 已接受打开请求：${target}`,
        artifact: target,
        data: { pid, target },
      };
    },
    verify: async (result) => ({
      passed: Number.isInteger(result.data?.pid) && result.data.pid > 0,
      summary: '已确认 Windows 接受打开请求；网页实际内容仍需后续浏览器感知能力验证。',
    }),
  });
}

module.exports = { APPS, launchDetached, registerWindowsCapabilities };
