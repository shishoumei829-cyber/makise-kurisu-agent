'use strict';

/**
 * Electron preload 脚本
 *
 * 通过 contextBridge 向渲染进程暴露受控 API，
 * 避免直接开放 Node.js / ipcRenderer 到网页上下文。
 */

const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('amadeus', {
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

  /**
   * 获取运行平台
   */
  platform: process.platform,
});
