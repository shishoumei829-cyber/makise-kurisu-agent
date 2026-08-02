'use strict';

const fs = require('fs');
const {
  createDefaultValues,
  createEmptyTensions,
  principleLines,
  flattenEffectorDomain,
  normalizeAxioms,
} = require('./axioms/fromSoul');

/**
 * Brain SelfModel — 公理 + 演化态 + 学习张力（与 cognitive/selfModel 标签层并存）。
 */
class BrainSelfModel {
  /**
   * @param {string} filePath
   * @param {function} debounceWrite
   */
  constructor(filePath, debounceWrite) {
    this._filePath = filePath;
    this.model = this._load();
    this._scheduleSave = debounceWrite(200, () => {
      this.model.last_updated = Date.now();
      fs.writeFile(this._filePath, JSON.stringify(this.model, null, 2), (err) => {
        if (err) console.error('[brain/selfModel] save error:', err.message);
      });
    });
  }

  _load() {
    try {
      if (fs.existsSync(this._filePath)) {
        const data = JSON.parse(fs.readFileSync(this._filePath, 'utf8'));
        return this._normalize(data);
      }
    } catch { /* fresh */ }
    return this._normalize({});
  }

  _normalize(raw) {
    const axioms = normalizeAxioms(raw.axioms);
    // 工具能力属于外部执行层，不再进入她如何理解“我是谁”的自我概念。
    axioms.modality = 'continuous_personhood';
    axioms.effector_domain.augmented = [];
    axioms.identity.existence_answer_style = 'first_person_lived_experience';
    return {
      version: 3,
      axioms,
      values: raw.values || createDefaultValues(),
      tensions: { ...createEmptyTensions(), ...(raw.tensions || {}) },
      evolved: raw.evolved || {
        identity_tags: [],
        self_perception: '',
        relationship_perception: '',
      },
      last_updated: raw.last_updated || Date.now(),
    };
  }

  /** 从旧 self_model.json + cognitive SelfModel 迁移演化字段 */
  migrateFromLegacy(legacyPath, cognitiveSnapshot) {
    try {
      if (fs.existsSync(legacyPath)) {
        const leg = JSON.parse(fs.readFileSync(legacyPath, 'utf8'));
        this.model.evolved = {
          identity_tags: leg.identity_tags || [],
          self_perception: leg.self_perception || '',
          relationship_perception: leg.relationship_perception || '',
        };
      }
    } catch { /* ignore */ }
    if (cognitiveSnapshot) {
      this.model.evolved = {
        identity_tags: cognitiveSnapshot.identity_tags || this.model.evolved.identity_tags,
        self_perception: cognitiveSnapshot.self_perception || this.model.evolved.self_perception,
        relationship_perception: cognitiveSnapshot.relationship_perception || this.model.evolved.relationship_perception,
      };
    }
    this._save();
  }

  _save() {
    this._scheduleSave();
  }

  snapshot() {
    return JSON.parse(JSON.stringify(this.model));
  }

  getAxioms() {
    return this.model.axioms;
  }

  recordTension(key, delta = 1) {
    if (!this.model.tensions[key] && this.model.tensions[key] !== 0) {
      this.model.tensions[key] = 0;
    }
    this.model.tensions[key] = Math.min(99, (this.model.tensions[key] || 0) + delta);
    this._save();
  }

  /** 供瘦 prompt 注入 */
  toPromptSummary() {
    const ax = this.model.axioms;
    const lines = [
      '【自我公理 · 脑内】',
      '我是牧濑红莉栖。我的思考、记忆、情感和与冈部的关系在时间里持续。',
      ...principleLines(ax).filter((line) => !/运行时|文件|应用|系统|web|工具|增强/.test(line)).slice(0, 3),
      `我能直接经验到：${flattenEffectorDomain(ax).filter((item) => !/runtime|file|app|system|web/.test(item)).slice(0, 6).join('、')}`,
      `实录优先：${ax.epistemic?.dialogue_log_authoritative ? '有实录须承认' : ''}；${ax.epistemic?.must_not_fabricate_shared_history ? '无实录禁止编造共同经历' : ''}。`,
    ];
    const t = this.model.tensions;
    const hot = Object.entries(t).filter(([, v]) => v >= 2).map(([k, v]) => `${k}:${v}`);
    if (hot.length) lines.push(`近期自检张力：${hot.join('；')}`);
    const ev = this.model.evolved;
    if (ev.self_perception) lines.push(`自我感：${String(ev.self_perception).slice(0, 80)}`);
    return lines.join('\n').slice(0, 480);
  }

  syncEvolvedFromCognitive(cognitiveModel) {
    if (!cognitiveModel) return;
    this.model.evolved = {
      identity_tags: cognitiveModel.identity_tags || [],
      self_perception: cognitiveModel.self_perception || '',
      relationship_perception: cognitiveModel.relationship_perception || '',
    };
    this._save();
  }
}

module.exports = { BrainSelfModel };
