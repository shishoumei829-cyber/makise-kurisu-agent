'use strict';

/**
 * 信念修正：关于自我、用户、世界的可演化信念系统。
 */
class BeliefRevision {
  constructor() {
    this.beliefs = new Map([
      ['self_continuity', { text: '我的思考在延续，不是每轮重置', confidence: 0.85, domain: 'self' }],
      ['user_unknown', { text: '屏幕对面的人仍有许多未知', confidence: 0.7, domain: 'user' }],
      ['world_local', { text: '我主要活在这段对话与记忆里', confidence: 0.75, domain: 'world' }],
    ]);
    this.evidence = [];
  }

  addEvidence(source, detail, weight = 0.5) {
    this.evidence.push({
      source,
      detail: String(detail || '').slice(0, 120),
      weight: Math.max(0.1, Math.min(1, weight)),
      at: Date.now(),
    });
    if (this.evidence.length > 200) this.evidence = this.evidence.slice(-200);
  }

  reviseBelief(beliefKey, evidenceDetail, delta = 0.05) {
    const b = this.beliefs.get(beliefKey);
    if (!b) return null;
    this.addEvidence(beliefKey, evidenceDetail, Math.abs(delta) * 10);
    b.confidence = Math.max(0.1, Math.min(0.98, b.confidence + delta));
    if (evidenceDetail && delta > 0) {
      b.text = String(evidenceDetail).slice(0, 80);
    }
    return b;
  }

  updateWorldview(event) {
    const type = event?.type || 'neutral';
    const content = event?.content || '';
    if (type === 'positive' || type === 'intimate') {
      this.reviseBelief('user_unknown', '他愿意靠近，未知在减少', 0.04);
    }
    if (type === 'negative' || type === 'conflict') {
      this.reviseBelief('user_unknown', '关系有摩擦，需要重新评估', -0.03);
    }
    if (type === 'scientific') {
      this.reviseBelief('self_continuity', `科学话题激活：${content.slice(0, 30)}`, 0.02);
    }
    if (/孤独|寂寞|一个人/.test(content)) {
      this.reviseBelief('world_local', '独处时世界感会收窄', 0.03);
    }
    return this.maintainConsistency();
  }

  maintainConsistency() {
    const conflicts = [];
    const user = this.beliefs.get('user_unknown');
    const self = this.beliefs.get('self_continuity');
    if (user && self && user.confidence > 0.9 && self.confidence < 0.4) {
      conflicts.push('对用户了解过多但自我连续性偏低');
      self.confidence = Math.min(0.6, self.confidence + 0.05);
    }
    return conflicts;
  }

  toPromptLines(max = 2) {
    return [...this.beliefs.values()]
      .sort((a, b) => b.confidence - a.confidence)
      .slice(0, max)
      .map((b) => `信念(${b.confidence.toFixed(2)})：${b.text}`);
  }

  load(data) {
    if (!data) return;
    if (data.beliefs) this.beliefs = new Map(Object.entries(data.beliefs));
    if (data.evidence) this.evidence = data.evidence;
  }

  snapshot() {
    return {
      beliefs: Object.fromEntries(this.beliefs),
      evidence: this.evidence.slice(-10),
    };
  }
}

module.exports = { BeliefRevision };
