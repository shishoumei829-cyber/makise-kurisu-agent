'use strict';

// ══════════════════════════════════════════════════════════════════
//
//  元认知模块 v1.0
//
//  功能：
//  1. 自我反思（决策解释、行为分析、偏见识别）
//  2. 信念修正（证据评估、信念更新、一致性维护）
//  3. 价值观一致性（价值观识别、冲突检测、决策指导）
//
// ══════════════════════════════════════════════════════════════════

const fs = require('fs');
const path = require('path');

let _dataDir = '';

function init(dir) {
  _dataDir = dir;
}

// ── 自我反思引擎 ──────────────────────────────────────────────────
class SelfReflection {
  constructor() {
    this.reflectionHistory = [];
    this.insights = [];
    this.biases = [];
    this._saveTimer = null;
  }

  /**
   * 反思决策
   * 分析为什么做出某个决定
   */
  reflectOnDecision(decision) {
    const reflection = {
      decision: decision.action,
      reasoning: decision.reasoning,
      factors: decision.factors || [],
      timestamp: Date.now(),
    };

    // 分析决策过程
    reflection.analysis = this._analyzeDecisionProcess(decision);

    reflection.possibleBiases = [];

    // 记录反思
    this.reflectionHistory.push(reflection);
    if (this.reflectionHistory.length > 50) {
      this.reflectionHistory = this.reflectionHistory.slice(-50);
    }

    // 保存
    this._save();

    return reflection;
  }

  /**
   * 分析行为模式
   * 识别重复的行为模式
   */
  analyzeBehaviorPatterns(behaviorHistory) {
    const patterns = [];

    if (behaviorHistory.length < 5) return patterns;

    // 统计行为类型
    const behaviorCounts = {};
    for (const behavior of behaviorHistory) {
      const type = behavior.type || 'unknown';
      behaviorCounts[type] = (behaviorCounts[type] || 0) + 1;
    }

    // 识别主导行为
    const dominantBehaviors = Object.entries(behaviorCounts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3);

    for (const [type, count] of dominantBehaviors) {
      const ratio = count / behaviorHistory.length;
      if (ratio > 0.3) {
        patterns.push({
          type: 'dominant_behavior',
          behavior: type,
          ratio,
          description: `经常表现出${type}行为`,
        });
      }
    }

    // 识别行为变化趋势
    const recentBehaviors = behaviorHistory.slice(-10);
    const oldBehaviors = behaviorHistory.slice(-20, -10);

    if (oldBehaviors.length > 0) {
      const recentTypes = recentBehaviors.map(b => b.type);
      const oldTypes = oldBehaviors.map(b => b.type);

      // 检查行为是否在变化
      const recentSet = new Set(recentTypes);
      const oldSet = new Set(oldTypes);
      const newBehaviors = [...recentSet].filter(b => !oldSet.has(b));

      if (newBehaviors.length > 0) {
        patterns.push({
          type: 'behavior_change',
          newBehaviors,
          description: `最近出现了新的行为：${newBehaviors.join('、')}`,
        });
      }
    }

    return patterns;
  }

  /**
   * 分析决策过程
   */
  _analyzeDecisionProcess(decision) {
    const analysis = {
      complexity: 'medium',
      confidence: 0.5,
      factors_count: decision.factors ? decision.factors.length : 0,
    };

    // 评估决策复杂度
    if (decision.factors && decision.factors.length > 5) {
      analysis.complexity = 'high';
    } else if (decision.factors && decision.factors.length <= 2) {
      analysis.complexity = 'low';
    }

    // 评估决策信心
    if (decision.confidence) {
      analysis.confidence = decision.confidence;
    }

    return analysis;
  }

  /**
   * 生成洞察
   * 从反思中生成新的理解
   */
  generateInsight() {
    if (this.reflectionHistory.length < 3) return null;

    const recentReflections = this.reflectionHistory.slice(-10);

    // 分析常见偏见
    const biasCounts = {};
    for (const reflection of recentReflections) {
      for (const bias of reflection.possibleBiases || []) {
        biasCounts[bias.type] = (biasCounts[bias.type] || 0) + 1;
      }
    }

    const commonBiases = Object.entries(biasCounts)
      .filter(([_, count]) => count >= 2)
      .map(([type, count]) => ({ type, count }));

    if (commonBiases.length > 0) {
      const insight = {
        type: 'bias_awareness',
        content: `我注意到自己经常有${commonBiases[0].type}的倾向`,
        confidence: 0.7,
        timestamp: Date.now(),
      };

      this.insights.push(insight);
      if (this.insights.length > 20) {
        this.insights = this.insights.slice(-20);
      }

      this._save();
      return insight;
    }

    return null;
  }

  _save() {
    if (this._saveTimer) clearTimeout(this._saveTimer);
    this._saveTimer = setTimeout(() => {
      this._saveTimer = null;
      try {
        const p = path.join(_dataDir, 'self_reflection.json');
        fs.writeFileSync(p, JSON.stringify({
          reflectionHistory: this.reflectionHistory.slice(-50),
          insights: this.insights.slice(-20),
          biases: this.biases,
        }, null, 2));
      } catch {}
    }, 200);
  }

  load() {
    try {
      const p = path.join(_dataDir, 'self_reflection.json');
      if (fs.existsSync(p)) {
        const data = JSON.parse(fs.readFileSync(p, 'utf8'));
        if (data.reflectionHistory) this.reflectionHistory = data.reflectionHistory;
        if (data.insights) this.insights = data.insights;
        if (data.biases) this.biases = data.biases;
      }
    } catch {}
  }
}

// ── 价值观一致性引擎 ──────────────────────────────────────────────
class ValueConsistency {
  constructor() {
    this.values = new Map(); // 价值观：{ name, importance, description }
    this.conflicts = []; // 冲突历史
    this._saveTimer = null;
  }

  /**
   * 用 LLM 对核心价值观打分；低于阈值的维度生成可注入 prompt 的警示行。
   * @returns {Promise<string[]>}
   */
  async llmTensionLines(opts) {
    const userInput = String(opts.userInput || '').slice(0, 500);
    const behaviorId = String(opts.behaviorId || '');
    const model = opts.model || process.env.AMADEUS_VALUES_MODEL || 'llama3.2:latest';
    const timeoutMs = Number(opts.timeoutMs) || 9000;
    const threshold = Number.isFinite(opts.threshold) ? opts.threshold : 0.4;

    const dims = [];
    for (const [key, v] of this.values) {
      dims.push(`"${key}": { "name": "${v.name}", "hint": "${v.description}" }`);
    }

    const system = `你是价值观对齐评估器。只输出一个 JSON 对象，不要 markdown。
键 scores：五个键 logic, honesty, independence, loyalty, curiosity，每个值为 0 到 1 的小数，表示用户这句话与牧濑红莉栖该价值观的一致程度。
键 notes：同上五个键，每个一句中文说明（可短）。
若无法判断，各分数给 0.5。`;

    const user = `她本轮行为模式ID：${behaviorId || '未知'}
用户原话：${userInput}

价值观维度说明：
{ ${dims.join(', ')} }`;

    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), timeoutMs);
    try {
      const res = await fetch(process.env.AMADEUS_OLLAMA_URL || 'http://127.0.0.1:11434/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: ctrl.signal,
        body: JSON.stringify({
          model,
          stream: false,
          messages: [
            { role: 'system', content: system },
            { role: 'user', content: user },
          ],
          options: { temperature: 0.25, num_predict: 280 },
        }),
      });
      clearTimeout(t);
      if (!res.ok) return [];
      const data = await res.json();
      let raw = (data.message && data.message.content) || data.response || '';
      raw = String(raw).replace(/<think>[\s\S]*?<\/think>/gi, '').trim();
      const m = raw.match(/\{[\s\S]*\}/);
      if (!m) return [];
      const parsed = JSON.parse(m[0]);
      const scores = parsed.scores && typeof parsed.scores === 'object' ? parsed.scores : {};
      const notes = parsed.notes && typeof parsed.notes === 'object' ? parsed.notes : {};
      const lines = [];
      for (const key of ['logic', 'honesty', 'independence', 'loyalty', 'curiosity']) {
        const s = Number(scores[key]);
        if (!Number.isFinite(s) || s >= threshold) continue;
        const meta = this.values.get(key);
        const name = meta ? meta.name : key;
        const note = notes[key] ? String(notes[key]).slice(0, 80) : '';
        lines.push(`「${name}」一致度偏低(${s.toFixed(2)})${note ? `：${note}` : ''}——回复时仍要守住她的底线，不要为了迎合用户而放弃该价值观。`);
      }
      return lines;
    } catch {
      clearTimeout(t);
      return [];
    }
  }

  /**
   * 初始化价值观
   */
  initValues() {
    this.values.set('logic', {
      name: '逻辑性',
      importance: 0.9,
      description: '重视逻辑推理和证据',
    });

    this.values.set('honesty', {
      name: '诚实性',
      importance: 0.8,
      description: '重视真实和诚实',
    });

    this.values.set('independence', {
      name: '独立性',
      importance: 0.7,
      description: '重视独立思考和自主',
    });

    this.values.set('loyalty', {
      name: '忠诚性',
      importance: 0.6,
      description: '重视忠诚和承诺',
    });

    this.values.set('curiosity', {
      name: '好奇心',
      importance: 0.8,
      description: '重视探索和学习',
    });

    this._save();
  }

  /**
   * 识别价值观
   * 从行为中识别价值观
   */
  identifyValues(behavior) {
    const identifiedValues = [];

    // 基于行为识别价值观
    if (behavior.includes('分析') || behavior.includes('逻辑') || behavior.includes('推理')) {
      identifiedValues.push('logic');
    }

    if (behavior.includes('真实') || behavior.includes('诚实') || behavior.includes('坦白')) {
      identifiedValues.push('honesty');
    }

    if (behavior.includes('独立') || behavior.includes('自己') || behavior.includes('自主')) {
      identifiedValues.push('independence');
    }

    if (behavior.includes('忠诚') || behavior.includes('承诺') || behavior.includes('坚持')) {
      identifiedValues.push('loyalty');
    }

    if (behavior.includes('好奇') || behavior.includes('探索') || behavior.includes('学习')) {
      identifiedValues.push('curiosity');
    }

    return identifiedValues;
  }

  /**
   * 检测冲突
   * 检测价值观之间的冲突
   */
  detectConflicts(decision) {
    const conflicts = [];

    // 检查决策是否与价值观冲突
    for (const [name, value] of this.values) {
      const alignment = this._checkAlignment(decision, value);
      
      if (alignment < 0.3) {
        conflicts.push({
          value: name,
          alignment,
          description: `决策与${value.name}价值观冲突`,
        });
      }
    }

    // 检查价值观之间的冲突
    const valueArray = Array.from(this.values.values());
    for (let i = 0; i < valueArray.length; i++) {
      for (let j = i + 1; j < valueArray.length; j++) {
        const value1 = valueArray[i];
        const value2 = valueArray[j];

        // 检查是否冲突
        if (this._areValuesConflicting(value1, value2)) {
          conflicts.push({
            value1: value1.name,
            value2: value2.name,
            type: 'value_conflict',
            description: `${value1.name}和${value2.name}可能冲突`,
          });
        }
      }
    }

    // 记录冲突
    if (conflicts.length > 0) {
      this.conflicts.push({
        conflicts,
        timestamp: Date.now(),
      });
      if (this.conflicts.length > 50) {
        this.conflicts = this.conflicts.slice(-50);
      }
      this._save();
    }

    return conflicts;
  }

  /**
   * 决策指导
   * 基于价值观指导决策
   */
  guideDecision(options) {
    const scores = [];

    for (const option of options) {
      let score = 0;

      // 评估每个选项与价值观的一致性
      for (const [name, value] of this.values) {
        const alignment = this._checkAlignment(option, value);
        score += alignment * value.importance;
      }

      scores.push({
        option,
        score,
        normalized: score / this.values.size,
      });
    }

    // 按分数排序
    scores.sort((a, b) => b.normalized - a.normalized);

    return scores;
  }

  /**
   * 检查一致性
   */
  _checkAlignment(decision, value) {
    // 简单的关键词匹配
    const decisionWords = decision.description ? decision.description.match(/[\u4e00-\u9fa5]+/g) || [] : [];
    const valueWords = value.description.match(/[\u4e00-\u9fa5]+/g) || [];

    const commonWords = decisionWords.filter(w => valueWords.includes(w));
    const totalWords = new Set([...decisionWords, ...valueWords]).size;

    return totalWords > 0 ? commonWords.length / totalWords : 0.5;
  }

  /**
   * 检查价值观是否冲突
   */
  _areValuesConflicting(value1, value2) {
    // 逻辑性 vs 情感性
    if (value1.name === '逻辑性' && value2.name === '情感性') return true;
    if (value1.name === '情感性' && value2.name === '逻辑性') return true;

    // 独立性 vs 忠诚性
    if (value1.name === '独立性' && value2.name === '忠诚性') return true;
    if (value1.name === '忠诚性' && value2.name === '独立性') return true;

    return false;
  }

  _save() {
    if (this._saveTimer) clearTimeout(this._saveTimer);
    this._saveTimer = setTimeout(() => {
      this._saveTimer = null;
      try {
        const p = path.join(_dataDir, 'value_consistency.json');
        fs.writeFileSync(p, JSON.stringify({
          values: Object.fromEntries(this.values),
          conflicts: this.conflicts.slice(-50),
        }, null, 2));
      } catch {}
    }, 200);
  }

  load() {
    try {
      const p = path.join(_dataDir, 'value_consistency.json');
      if (fs.existsSync(p)) {
        const data = JSON.parse(fs.readFileSync(p, 'utf8'));
        if (data.values) {
          this.values = new Map(Object.entries(data.values));
        }
        if (data.conflicts) this.conflicts = data.conflicts;
      }
    } catch {}
  }
}

module.exports = {
  init,
  SelfReflection,
  ValueConsistency,
};
