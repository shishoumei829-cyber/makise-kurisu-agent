const { app, BrowserWindow, Menu, Tray, globalShortcut, powerMonitor } = require('electron');
const path = require('path');

// Fix for transparency issues on some systems
app.disableHardwareAcceleration();

let tray = null;
let mainWindow = null;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 600,
    height: 400,
    frame: false, // Frameless window
    transparent: true, // Transparent background
    alwaysOnTop: true, // Keep on top
    resizable: false, // Fixed size
    hasShadow: false, // Remove shadow for cleaner overlay
    skipTaskbar: false, // Keep in taskbar for access
    webPreferences: {
      nodeIntegration: true, // Allow node integration
      contextIsolation: false // Required for some node integration features in renderer
    },
    icon: path.join(__dirname, 'assets/icon.png') // Optional: add icon if available
  });

  mainWindow.loadFile('index.html');
  
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
    const iconPath = path.join(__dirname, 'assets/icon.png'); // Ensure this icon exists
    tray = new Tray(iconPath);
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

app.whenReady().then(() => {
  createWindow();
  createTray();

  // Auto-launch configuration (Force enable)
  app.setLoginItemSettings({
    openAtLogin: true,
    path: process.execPath,
    args: []
  });

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
});

app.on('window-all-closed', function () {
  if (process.platform !== 'darwin') app.quit();
});
