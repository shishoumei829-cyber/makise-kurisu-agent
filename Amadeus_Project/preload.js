'use strict';

/**
 * Electron preload 脚本
 *
 * 通过 contextBridge 向渲染进程暴露受控 API，
 * 避免直接开放 Node.js / ipcRenderer 到网页上下文。
 */

const { contextBridge, ipcRenderer } = require('electron');

// ★ 禁止暴露为 window.amadeus：页面主脚本也用 const amadeus = new AmadeusConsciousness()，
// Electron contextBridge 会把同名 API 注册成渲染进程词法绑定，直接 SyntaxError，整页卡在初始化。
contextBridge.exposeInMainWorld('amadeusDesktop', {
  /**
   * 订阅系统唤醒事件（系统从休眠/锁屏恢复时触发）
   * @param {(type: 'resume' | 'unlock') => void} callback
   */
  onSystemWakeup(callback) {
    ipcRenderer.on('system-wakeup', (_event, type) => callback(type));
  },

  /**
   * 移除系统唤醒监听
   */
  offSystemWakeup() {
    ipcRenderer.removeAllListeners('system-wakeup');
  },

  minimize() {
    return ipcRenderer.invoke('window-minimize');
  },
  maximize() {
    return ipcRenderer.invoke('window-maximize');
  },
  close() {
    return ipcRenderer.invoke('window-close');
  },
  isMaximized() {
    return ipcRenderer.invoke('window-is-maximized');
  },

  /**
   * 获取运行平台
   */
  platform: process.platform,
});
