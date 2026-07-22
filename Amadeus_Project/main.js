const { app, BrowserWindow, Menu, Tray, globalShortcut, powerMonitor, dialog, session } = require('electron');
const path = require('path');
const fs = require('fs');
const net = require('net');
const { spawn } = require('child_process');

function loadDotEnv() {
  const envPath = path.join(__dirname, '.env');
  if (!fs.existsSync(envPath)) return;
  for (const line of fs.readFileSync(envPath, 'utf8').split(/\r?\n/)) {
    const text = line.trim();
    if (!text || text.startsWith('#')) continue;
    const eq = text.indexOf('=');
    if (eq < 1) continue;
    const key = text.slice(0, eq).trim();
    let value = text.slice(eq + 1).trim();
    if (
      (value.startsWith('"') && value.endsWith('"'))
      || (value.startsWith("'") && value.endsWith("'"))
    ) value = value.slice(1, -1);
    if (process.env[key] == null || process.env[key] === '') process.env[key] = value;
  }
}
loadDotEnv();

let tray = null;
let mainWindow = null;
let backendProcess = null;
const BACKEND_PORT = Number(process.env.AMADEUS_BACKEND_PORT) || 3000;
const BACKEND_URL = `http://localhost:${BACKEND_PORT}`;

function resolveIconPath() {
  const candidates = [
    path.join(__dirname, 'assets', 'icon.png'),
    path.join(__dirname, 'assets', 'Live2d', 'kurisu', '0.png'),
  ];
  for (const p of candidates) {
    if (fs.existsSync(p)) return p;
  }
  return candidates[0];
}

function resolveBackendPaths() {
  const root = __dirname;
  return { root, serverPath: path.join(root, 'server.js') };
}

function isPortOpen(port, host = '127.0.0.1', timeoutMs = 800) {
  return new Promise((resolve) => {
    const socket = new net.Socket();
    let done = false;
    const finish = (ok) => {
      if (done) return;
      done = true;
      socket.destroy();
      resolve(ok);
    };
    socket.setTimeout(timeoutMs);
    socket.once('connect', () => finish(true));
    socket.once('timeout', () => finish(false));
    socket.once('error', () => finish(false));
    socket.connect(port, host);
  });
}

async function waitForBackend(port, timeoutMs = 15000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    let ok = false;
    try {
      const res = await fetch(`http://127.0.0.1:${port}/health`, { signal: AbortSignal.timeout(1200) });
      const data = await res.json();
      ok = res.ok && data && data.ok === true;
    } catch (_) { /* backend is still starting */ }
    if (ok) return true;
    // eslint-disable-next-line no-await-in-loop
    await new Promise((r) => setTimeout(r, 300));
  }
  return false;
}

async function ensureBackendRunning() {
  const portOccupied = await isPortOpen(BACKEND_PORT);
  if (portOccupied) {
    try {
      const res = await fetch(`${BACKEND_URL}/health`, { signal: AbortSignal.timeout(1500) });
      const data = await res.json();
      if (res.ok && data && data.ok === true) {
        console.log(`[main] Backend already running on ${BACKEND_URL}`);
        return true;
      }
    } catch (_) { /* another process owns the port */ }
    console.error(`[main] Port ${BACKEND_PORT} is occupied but is not a healthy Amadeus backend.`);
    return false;
  }

  const { root, serverPath } = resolveBackendPaths();
  if (!fs.existsSync(serverPath)) {
    console.error('[main] server.js not found:', serverPath);
    return false;
  }

  backendProcess = spawn(process.execPath, [serverPath], {
    cwd: root,
    env: {
      ...process.env,
      ELECTRON_RUN_AS_NODE: '1',
      AMADEUS_BACKEND_PORT: String(BACKEND_PORT),
    },
    stdio: 'ignore',
    windowsHide: true,
    detached: false,
  });

  backendProcess.once('error', (err) => {
    console.error('[main] Failed to start backend:', err.message);
  });
  backendProcess.once('exit', (code, signal) => {
    console.log(`[main] Backend exited (code=${code}, signal=${signal || 'none'})`);
    backendProcess = null;
  });

  const ready = await waitForBackend(BACKEND_PORT, 18000);
  if (!ready) {
    console.warn('[main] Backend did not become ready in time; UI will still open.');
  }
  return ready;
}

function stopBackendProcess() {
  if (!backendProcess || backendProcess.killed) return;
  try {
    backendProcess.kill();
  } catch (e) {
    console.warn('[main] Failed to stop backend process:', e.message);
  }
}

  // 摄像头/麦克风/通知只放行给本机后端页面，其余权限仍拒绝
function configureMediaPermissions() {
  const allowedPermissions = new Set([
    'media',
    'microphone',
    'camera',
    'notifications',
    'speaker-selection',
    'display-capture',
  ]);
  const isTrustedOrigin = (url) => {
    const u = String(url || '');
    return u.startsWith(BACKEND_URL) || u.startsWith('http://127.0.0.1:') || u.startsWith('http://localhost:') || u.startsWith('file://');
  };
  session.defaultSession.setPermissionRequestHandler((webContents, permission, callback) => {
    callback(allowedPermissions.has(permission) && isTrustedOrigin(webContents.getURL()));
  });
  session.defaultSession.setPermissionCheckHandler((webContents, permission, requestingOrigin) => (
    allowedPermissions.has(permission) && isTrustedOrigin(requestingOrigin || (webContents && webContents.getURL()))
  ));
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 800,
    minWidth: 1080,
    minHeight: 680,
    frame: false,
    transparent: false,
    backgroundColor: '#000000',
    alwaysOnTop: false,
    resizable: true,
    hasShadow: true,
    skipTaskbar: false,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      nodeIntegration: false,
      contextIsolation: true,
    },
    icon: resolveIconPath(),
  });

  mainWindow.loadURL(BACKEND_URL).catch(() => {
    mainWindow.loadFile('amadeus_work.html');
  });
  
  // Power Monitor Events
  powerMonitor.on('resume', () => {
      if (mainWindow) {
          mainWindow.webContents.send('system-wakeup', 'resume');
      }
  });
  
  powerMonitor.on('unlock-screen', () => {
      if (mainWindow) {
          mainWindow.webContents.send('system-wakeup', 'unlock');
      }
  });
  
  // Context Menu
  mainWindow.webContents.on('context-menu', (e, params) => {
    const menu = Menu.buildFromTemplate([
      { label: '隐藏 Amadeus', click: () => mainWindow.hide() },
      { label: '退出 Amadeus', click: () => app.quit() },
      { type: 'separator' },
      { label: '调试模式', click: () => mainWindow.webContents.openDevTools({ mode: 'detach' }) }
    ]);
    menu.popup();
  });

  // Open DevTools optionally
  // mainWindow.webContents.openDevTools({ mode: 'detach' });

  // Handle window close to hide instead of quit
  mainWindow.on('close', (event) => {
      if (!app.isQuitting) {
          event.preventDefault();
          mainWindow.hide();
      }
      return false;
  });
}

function createTray() {
    tray = new Tray(resolveIconPath());
    const contextMenu = Menu.buildFromTemplate([
        { label: '显示/隐藏 Amadeus', click: toggleWindow },
        { label: '退出程序', click: () => {
            app.isQuitting = true;
            app.quit();
        }}
    ]);
    tray.setToolTip('Amadeus Project');
    tray.setContextMenu(contextMenu);
    
    tray.on('click', toggleWindow);
}

function toggleWindow() {
    if (mainWindow.isVisible()) {
        mainWindow.hide();
    } else {
        mainWindow.show();
    }
}

app.whenReady().then(async () => {
  configureMediaPermissions();
  const ready = await ensureBackendRunning();
  if (!ready) {
    await dialog.showMessageBox({
      type: 'warning',
      title: 'Amadeus',
      message: '后端未能启动',
      detail: `请确认 Ollama 已启动，且端口 ${BACKEND_PORT} 未被占用。\n源码用户可运行 run_backend.bat；安装包用户请重启应用或查看 INSTALL.md。`,
    });
  }
  createWindow();
  createTray();

  if (process.env.AMADEUS_AUTO_LAUNCH === '1') {
    app.setLoginItemSettings({
      openAtLogin: true,
      path: process.execPath,
      args: [],
    });
  }

  // Global Shortcut
  globalShortcut.register('CommandOrControl+H', () => {
      toggleWindow();
  });

  app.on('activate', function () {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

const gotSingleInstanceLock = app.requestSingleInstanceLock();
if (!gotSingleInstanceLock) {
  app.quit();
} else {
  app.on('second-instance', () => {
    if (!mainWindow) return;
    if (mainWindow.isMinimized()) mainWindow.restore();
    mainWindow.show();
    mainWindow.focus();
  });
}

app.on('will-quit', () => {
  // Unregister all shortcuts.
  globalShortcut.unregisterAll();
  stopBackendProcess();
});

app.on('window-all-closed', function () {
  if (process.platform !== 'darwin') app.quit();
});
