'use strict';

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

class FileUndoStore {
  constructor(dataDir, journal) {
    this.dir = path.join(dataDir, 'butler', 'trash');
    this.path = path.join(dataDir, 'butler', 'trash_manifest.json');
    this.journal = journal;
    fs.mkdirSync(this.dir, { recursive: true });
    this.entries = this._load();
  }

  _load() {
    try {
      const parsed = JSON.parse(fs.readFileSync(this.path, 'utf8'));
      return Array.isArray(parsed.entries) ? parsed.entries : [];
    } catch { return []; }
  }

  _save() {
    const temp = `${this.path}.tmp`;
    fs.writeFileSync(temp, JSON.stringify({ version: 1, savedAt: Date.now(), entries: this.entries }, null, 2), 'utf8');
    fs.renameSync(temp, this.path);
  }

  moveToTrash(sourcePath, taskId = '') {
    const source = path.resolve(sourcePath);
    if (!fs.existsSync(source)) throw new Error('source file does not exist');
    const stat = fs.lstatSync(source);
    if (!stat.isFile()) throw new Error('only files can be moved to Amadeus trash');
    const id = `trash_${crypto.randomUUID()}`;
    const trashPath = path.join(this.dir, `${id}_${path.basename(source)}`);
    try {
      fs.renameSync(source, trashPath);
    } catch (error) {
      if (error.code !== 'EXDEV') throw error;
      fs.copyFileSync(source, trashPath, fs.constants.COPYFILE_EXCL);
      fs.unlinkSync(source);
    }
    const entry = {
      id,
      originalPath: source,
      trashPath,
      status: 'trashed',
      size: stat.size,
      taskId: String(taskId || ''),
      createdAt: Date.now(),
      restoredAt: null,
    };
    this.entries.push(entry);
    this._save();
    this.journal?.append('file.trashed', { entry }, { actor: 'executor', correlationId: taskId || id });
    return entry;
  }

  get(id) {
    if (id === 'latest') return [...this.entries].reverse().find((entry) => entry.status === 'trashed') || null;
    return this.entries.find((entry) => entry.id === id) || null;
  }

  list(status = '') {
    return this.entries.filter((entry) => !status || entry.status === status).slice().sort((a, b) => b.createdAt - a.createdAt);
  }

  restore(id, taskId = '') {
    const entry = this.get(id);
    if (!entry) throw new Error('trash entry not found');
    if (entry.status !== 'trashed' || !fs.existsSync(entry.trashPath)) throw new Error('trash entry is not restorable');
    if (fs.existsSync(entry.originalPath)) throw new Error('original path is occupied; refusing to overwrite');
    fs.mkdirSync(path.dirname(entry.originalPath), { recursive: true });
    try {
      fs.renameSync(entry.trashPath, entry.originalPath);
    } catch (error) {
      if (error.code !== 'EXDEV') throw error;
      fs.copyFileSync(entry.trashPath, entry.originalPath, fs.constants.COPYFILE_EXCL);
      fs.unlinkSync(entry.trashPath);
    }
    entry.status = 'restored';
    entry.restoredAt = Date.now();
    this._save();
    this.journal?.append('file.restored', { entry }, { actor: 'executor', correlationId: taskId || entry.id });
    return entry;
  }
}

module.exports = { FileUndoStore };
