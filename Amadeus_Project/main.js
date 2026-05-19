const { app, BrowserWindow, Menu, Tray, globalShortcut, powerMonitor, dialog } = require('electron');
const path = require('path');
const fs = require('fs');
const net = require('net');
const { spawn } = require('child_process');

// Fix for transparency issues on some systems
app.disableHardwareAcceleration();

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
    // eslint-disable-next-line no-await-in-loop
    const ok = await isPortOpen(port);
    if (ok) return true;
    // eslint-disable-next-line no-await-in-loop
    await new Promise((r) => setTimeout(r, 300));
  }
  return false;
}

async function ensureBackendRunning() {
  const alreadyRunning = await isPortOpen(BACKEND_PORT);
  if (alreadyRunning) {
    console.log(`[main] Backend already running on ${BACKEND_URL}`);
    return true;
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

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 600,
    height: 400,
    frame: false,
    transparent: true,
    alwaysOnTop: true,
    resizable: false,
    hasShadow: false,
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

app.on('will-quit', () => {
  // Unregister all shortcuts.
  globalShortcut.unregisterAll();
  stopBackendProcess();
});

app.on('window-all-closed', function () {
  if (process.platform !== 'darwin') app.quit();
});
