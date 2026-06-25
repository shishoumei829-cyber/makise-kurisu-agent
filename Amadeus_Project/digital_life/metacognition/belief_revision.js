'use strict';

const { DEFAULT_BELIEFS } = require('./constants');

/**
 * 信念修正：证据链、领域一致性、与事件耦合的更新。
 */
class BeliefRevision {
  constructor() {
    this.beliefs = new Map(DEFAULT_BELIEFS);
    this.evidence = [];
    this.conflicts = [];
  }

  addEvidence(source, detail, weight = 0.5) {
    this.evidence.push({
      source,
      detail: String(detail || '').slice(0, 140),
      weight: Math.max(0.1, Math.min(1, weight)),
      at: Date.now(),
    });
    if (this.evidence.length > 250) this.evidence = this.evidence.slice(-250);
  }

  reviseBelief(beliefKey, evidenceDetail, delta = 0.05) {
    const b = this.beliefs.get(beliefKey);
    if (!b) return null;
    this.addEvidence(beliefKey, evidenceDetail, Math.abs(delta) * 10);
    b.confidence = Math.max(0.1, Math.min(0.98, b.confidence + delta));
    if (evidenceDetail && delta > 0) {
      b.text = String(evidenceDetail).slice(0, 90);
    }
    this.maintainConsistency();
    return b;
  }

  updateWorldview(event) {
    const type = event?.type || 'neutral';
    const content = event?.content || '';

    if (type === 'positive' || type === 'intimate') {
      this.reviseBelief('user_unknown', '他愿意靠近，未知在减少', 0.04);
      this.reviseBelief('rel_guarded', '亲近有进展，但边界仍在', -0.02);
    }
    if (type === 'negative' || type === 'conflict') {
      this.reviseBelief('user_unknown', '关系有摩擦，需要重新评估', -0.03);
      this.reviseBelief('rel_guarded', '边界感要拉回来一点', 0.05);
    }
    if (type === 'scientific') {
      this.reviseBelief('self_continuity', `科学话题激活：${content.slice(0, 30)}`, 0.02);
    }
    if (/孤独|寂寞|一个人/.test(content)) {
      this.reviseBelief('world_local', '独处时世界感会收窄', 0.03);
    }
    if (/相信|记得|延续/.test(content)) {
      this.reviseBelief('self_continuity', '对话里出现连续性线索', 0.03);
    }

    return this.maintainConsistency();
  }

  ingestMetacognitionInsight(insight) {
    if (!insight?.content) return;
    this.addEvidence('insight', insight.content, 0.6);
    if (/太冷|太疏远|回避/.test(insight.content)) {
      this.reviseBelief('rel_guarded', '可能过度防御', -0.04);
    }
    if (/套话|敷衍/.test(insight.content)) {
      this.reviseBelief('self_continuity', '需要更真实的表达', 0.03);
    }
  }

  maintainConsistency() {
    this.conflicts = [];
    const user = this.beliefs.get('user_unknown');
    const self = this.beliefs.get('self_continuity');
    const rel = this.beliefs.get('rel_guarded');

    if (user && self && user.confidence > 0.9 && self.confidence < 0.4) {
      this.conflicts.push('对用户了解过多但自我连续性偏低');
      self.confidence = Math.min(0.6, self.confidence + 0.05);
    }
    if (user && rel && user.confidence > 0.85 && rel.confidence > 0.9) {
      this.conflicts.push('亲近信念与防御信念拉扯');
      rel.confidence = Math.max(0.55, rel.confidence - 0.03);
    }
    return this.conflicts;
  }

  toPromptLines(max = 3) {
    return [...this.beliefs.values()]
      .sort((a, b) => b.confidence - a.confidence)
      .slice(0, max)
      .map((b) => `信念(${b.confidence.toFixed(2)})：${b.text}`);
  }

  lowConfidenceBeliefs(threshold = 0.55) {
    return [...this.beliefs.entries()]
      .filter(([, b]) => b.confidence < threshold)
      .map(([k, b]) => ({ key: k, ...b }));
  }

  load(data) {
    if (!data) return;
    if (data.beliefs) this.beliefs = new Map(Object.entries(data.beliefs));
    if (data.evidence) this.evidence = data.evidence;
    if (data.conflicts) this.conflicts = data.conflicts;
  }

  snapshot() {
    return {
      beliefs: Object.fromEntries(this.beliefs),
      evidence: this.evidence.slice(-12),
      conflicts: this.conflicts.slice(-5),
      lowConfidence: this.lowConfidenceBeliefs(),
    };
  }
}

module.exports = { BeliefRevision };
