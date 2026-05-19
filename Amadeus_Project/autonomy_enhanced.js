'use strict';

// ══════════════════════════════════════════════════════════════════
//  自主性增强模块（精简）：好奇心引擎 + 内驱力常量
// ══════════════════════════════════════════════════════════════════

const DRIVE_TYPES = {
  CURIOSITY: '好奇心',
  CREATIVITY: '创造性',
  EXPLORATION: '探索欲',
  MASTERY: '掌控欲',
  CONNECTION: '连接欲',
  AUTONOMY: '自主欲',
  MEANING: '意义欲',
  PLAYFULNESS: '玩心',
  PROTECTION: '保护欲',
  CREATION: '创造欲',
};

class CuriosityEngine {
  constructor() {
    this.knowledgeGaps = new Map();
    this.informationGain = new Map();
    this.curiousQuestions = [];
    this._lastUpdate = Date.now();
  }

  calculateInformationGain(topic, memory) {
    const now = Date.now();
    const daysSinceLastExplore = (now - (this.informationGain.get(topic)?.lastExplored || 0)) / (1000 * 60 * 60 * 24);
    const novelty = Math.min(1, daysSinceLastExplore / 7);
    const relatedEvents = memory.events.filter(e =>
      e.content && e.content.includes(topic)
    ).length;
    const relevance = Math.min(1, relatedEvents / 5);
    const topicValue = this._getTopicValue(topic);
    const gain = (novelty * 0.4 + relevance * 0.3 + topicValue * 0.3);
    this.informationGain.set(topic, {
      gain,
      lastExplored: now,
      explorationCount: (this.informationGain.get(topic)?.explorationCount || 0) + 1,
    });
    return gain;
  }

  _getTopicValue(topic) {
    if (/科学|量子|神经|时间|物理|数学|实验|理论/.test(topic)) return 0.9;
    if (/你|我|他|她|我们|他们/.test(topic)) return 0.7;
    return 0.5;
  }

  discoverKnowledgeGaps(memory, selfModel) {
    const gaps = [];
    const exploredTopics = new Set();
    for (const event of memory.events) {
      if (event.content) {
        const keywords = event.content.match(/[\u4e00-\u9fa5]+/g) || [];
        keywords.forEach(kw => exploredTopics.add(kw));
      }
    }
    const self = selfModel.get();
    if (self.identity_tags) {
      self.identity_tags.forEach(tag => {
        if (!exploredTopics.has(tag)) {
          gaps.push({ topic: tag, importance: 0.7, reason: '自我认知中的未知领域' });
        }
      });
    }
    if (self.relationship_perception) {
      const relationshipKeywords = self.relationship_perception.match(/[\u4e00-\u9fa5]+/g) || [];
      relationshipKeywords.forEach(kw => {
        if (!exploredTopics.has(kw)) {
          gaps.push({ topic: kw, importance: 0.6, reason: '关系认知中的未知领域' });
        }
      });
    }
    return gaps;
  }

  generateCuriousQuestions(context, memory) {
    const questions = [];
    const { pad, selfModel, relScore } = context;
    const gaps = this.discoverKnowledgeGaps(memory, selfModel);
    for (const gap of gaps.slice(0, 2)) {
      questions.push({
        question: `我想知道更多关于${gap.topic}的事情`,
        topic: gap.topic,
        priority: gap.importance,
        type: 'knowledge_gap',
      });
    }
    if (pad.S > 0.3 && pad.S < 0.7) {
      questions.push({
        question: '他到底是什么样的人？',
        topic: 'user_personality',
        priority: 0.6,
        type: 'understanding',
      });
    }
    if (pad.A > 0.5) {
      questions.push({
        question: '有什么有趣的事情可以聊？',
        topic: 'interesting_topics',
        priority: 0.5,
        type: 'exploration',
      });
    }
    return questions;
  }
}

module.exports = {
  DRIVE_TYPES,
  CuriosityEngine,
};
