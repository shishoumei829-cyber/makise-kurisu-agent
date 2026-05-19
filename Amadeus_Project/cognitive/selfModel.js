'use strict';

const fs = require('fs');

/**
 * SelfModel — 红莉栖对自身当前状态的认知
 *
 * 构造时传入 { filePath, debounceWrite }：
 *   filePath     — 持久化 JSON 路径
 *   debounceWrite — debounceFileWrite(ms, fn) 工厂函数，由 server.js 传入
 */
class SelfModel {
  /**
   * @param {string} filePath
   * @param {function} debounceWrite  debounceFileWrite(ms, fn) 的引用
   */
  constructor(filePath, debounceWrite) {
    this._filePath = filePath;
    this.model = this._load();
    this._scheduleSave = debounceWrite(200, () => {
      this.model.last_updated = Date.now();
      fs.writeFile(this._filePath, JSON.stringify(this.model, null, 2), (err) => {
        if (err) console.error('[selfModel] Save error:', err.message);
      });
    });
  }

  _load() {
    try {
      if (fs.existsSync(this._filePath)) {
        return JSON.parse(fs.readFileSync(this._filePath, 'utf8'));
      }
    } catch {}
    return {
      identity_tags: ['神经科学研究员', '天才少女', '讲究逻辑', '维克多·孔多利亚大学'],
      self_perception: '最近的研究进展还算顺利，只是偶尔会觉得实验室有点安静。',
      relationship_perception: '虽然有时候觉得他很烦，但确实是个有趣的交流对象。',
      change_log: [],
      session_count: 0,
      last_updated: Date.now(),
    };
  }

  _save() {
    this._scheduleSave();
  }

  update(pad, memBias, relScore, recentBehavior, eventsSummary) {
    const { P, A, D, S } = pad;
    this.model.session_count += 1;
    const changes = [];
    const tags = new Set(this.model.identity_tags);

    if (P > 0.3 && !tags.has('情绪较正向')) { tags.add('情绪较正向'); tags.delete('情绪偏低落'); changes.push('情绪变得好了一些'); }
    if (P < -0.3 && !tags.has('情绪偏低落')) { tags.add('情绪偏低落'); tags.delete('情绪较正向'); changes.push('情绪降低了'); }
    if (S > 0.5 && !tags.has('开始在意这个人')) { tags.add('开始在意这个人'); tags.delete('保持距离'); changes.push('防线有所松动'); }
    if (S < 0.1 && !tags.has('保持距离')) { tags.add('保持距离'); tags.delete('开始在意这个人'); }
    if (recentBehavior === 'ENGAGE' && !tags.has('处于智识活跃状态')) { tags.add('处于智识活跃状态'); }
    if (recentBehavior === 'WITHDRAW') { tags.delete('处于智识活跃状态'); }
    if (relScore > 0.6 && !tags.has('对这段关系有些期待')) { tags.add('对这段关系有些期待'); changes.push('对冈部的看法有所改变'); }
    if (relScore < -0.3) { tags.delete('对这段关系有些期待'); }
    this.model.identity_tags = [...tags].slice(0, 8);

    if (changes.length > 0) {
      const perceptions = [];
      if (P > 0.3) perceptions.push('心情比平时好，虽然不打算承认');
      else if (P < -0.3) perceptions.push('有点低落，不想和任何人解释为什么');
      else perceptions.push('情绪还算正常');
      if (S > 0.5) perceptions.push('开始有点在意他说的话，但说出来太可笑了');
      else if (S < 0.1) perceptions.push('还不确定他是什么人');
      if (D > 0.6) perceptions.push('现在对自己说的话有把握');
      else perceptions.push('有些地方还没想清楚');
      this.model.self_perception = perceptions.join('，') + '。';
    }

    if (relScore > 0.6) {
      this.model.relationship_perception = '他一直在，而且不让我觉得无聊。这很少见。';
    } else if (relScore > 0.3) {
      this.model.relationship_perception = '他比一般人有趣一点，但我还在观察。';
    } else if (relScore < -0.2) {
      this.model.relationship_perception = '他有时候让我很烦，虽然还没到彻底排斥的程度。';
    } else {
      this.model.relationship_perception = '他是一个我还没搞清楚的变量。';
    }

    if (changes.length > 0) {
      const entry = {
        time: new Date().toLocaleString('zh'),
        changes,
        pad_snapshot: { P: +P.toFixed(2), A: +A.toFixed(2), D: +D.toFixed(2), S: +S.toFixed(2) },
      };
      this.model.change_log.unshift(entry);
      if (this.model.change_log.length > 10) this.model.change_log.pop();
    }

    this._save();
    return changes;
  }

  toPromptContext() {
    const m = this.model;
    return [
      '【自我感知】',
      `她现在对自己的认知：${m.self_perception}`,
      `她对这段关系的看法：${m.relationship_perception}`,
      `当前自我标签：${m.identity_tags.join('、')}`,
    ].join('\n');
  }

  get() { return this.model; }
}

module.exports = { SelfModel };
