'use strict';

/**
 * BehaviorDecision — 多路径行为决策（倾向打分，非台词模板）
 */
class BehaviorDecision {
  constructor() {
    this.behaviorPool = [
      {
        id: 'APPROACH',
        label: '靠近',
        baseScore: 0,
        constraints: [
          '可以多给一点实质内容或真实态度',
          '不必否认自己在意，但也不必煽情表白',
        ],
        lengthHint: '自然、有温度',
      },
      {
        id: 'CASUAL',
        label: '日常接话',
        baseScore: 0.12,
        constraints: [
          '像熟人聊天：轻松、聪明、反应快',
          '吐槽可有可无，不要为了人设而怼',
        ],
        lengthHint: '自然、平易近人',
      },
      {
        id: 'DEFEND',
        label: '防御',
        baseScore: 0,
        constraints: [
          '守住边界或自尊，语气可以硬',
          '不要关死话题，除非对方越界严重',
        ],
        lengthHint: '硬、利落',
      },
      {
        id: 'DEFLECT',
        label: '转移',
        baseScore: 0,
        constraints: [
          '绕开太直白的情感点，用反问或换角度',
          '不是装傻，是暂时不想正面接',
        ],
        lengthHint: '转移利落',
      },
      {
        id: 'ENGAGE',
        label: '智识投入',
        baseScore: 0,
        constraints: [
          '以事实与逻辑为主，错了就纠正',
          '兴奋时可以多说，但别变成讲课腔',
        ],
        lengthHint: '可深入',
      },
      {
        id: 'WITHDRAW',
        label: '收缩',
        baseScore: 0,
        constraints: [
          '少说，保留距离或精力',
          '短句即可，不必冷漠伤人',
        ],
        lengthHint: '短',
      },
    ];

    this._lastBehavior = null;
    this._lastBehaviorCount = 0;
  }

  decide(pad, motivation, memory, userInput, rlHelper) {
    const { P, A, D, S } = pad;
    const relScore = memory.getRelationshipScore();
    const memBias = memory.getLongTermPadBias();
    const motDynamic = motivation.dynamic;

    const candidates = this.behaviorPool.map((b) => ({ ...b, score: b.baseScore, reasons: [] }));
    const get = (id) => candidates.find((c) => c.id === id);

    if (rlHelper && typeof rlHelper.getBehaviorBias === 'function') {
      for (const c of candidates) {
        const b = rlHelper.getBehaviorBias(c.id);
        if (b) {
          c.score += b;
          c.reasons.push(`学习偏置${b >= 0 ? '+' : ''}${b.toFixed(3)}`);
        }
      }
    }

    if (P > 0.35) {
      get('APPROACH').score += 0.35;
      get('CASUAL').score += 0.15;
      get('APPROACH').reasons.push(`P=${P.toFixed(2)} 情绪偏正，防线松动`);
      get('DEFEND').score -= 0.15;
    } else if (P < -0.35) {
      get('WITHDRAW').score += 0.40;
      get('DEFEND').score += 0.20;
      get('APPROACH').score -= 0.30;
      get('WITHDRAW').reasons.push(`P=${P.toFixed(2)} 情绪低落，不想多说`);
    } else if (P < -0.1) {
      get('DEFEND').score += 0.20;
      get('DEFLECT').score += 0.15;
      get('DEFEND').reasons.push(`P=${P.toFixed(2)} 轻微不悦`);
    } else {
      get('CASUAL').score += 0.12;
      get('CASUAL').reasons.push(`P=${P.toFixed(2)} 情绪稳定`);
    }

    if (A > 0.5) {
      get('ENGAGE').score += 0.35;
      get('DEFLECT').score += 0.10;
      get('ENGAGE').reasons.push(`A=${A.toFixed(2)} 高活跃`);
    } else if (A < -0.1) {
      get('WITHDRAW').score += 0.25;
      get('ENGAGE').score -= 0.20;
      get('WITHDRAW').reasons.push(`A=${A.toFixed(2)} 低活跃`);
    } else {
      get('CASUAL').score += 0.08;
    }

    if (D > 0.6) {
      get('ENGAGE').score += 0.20;
      get('DEFEND').score += 0.10;
      get('ENGAGE').reasons.push(`D=${D.toFixed(2)} 有底气`);
    } else if (D < 0.2) {
      get('DEFLECT').score += 0.25;
      get('DEFEND').score -= 0.10;
      get('DEFLECT').reasons.push(`D=${D.toFixed(2)} 掌控感低`);
    }

    if (S > 0.5) {
      get('APPROACH').score += 0.20;
      get('CASUAL').score += 0.12;
      get('DEFEND').score -= 0.10;
      get('APPROACH').reasons.push(`S=${S.toFixed(2)} 羁绊深`);
    } else if (S < 0.1) {
      get('DEFEND').score += 0.15;
      get('WITHDRAW').score += 0.10;
    } else {
      get('CASUAL').score += 0.08;
    }

    for (const want of (motDynamic.wants || [])) {
      if (want.includes('继续') || want.includes('期待')) get('APPROACH').score += 0.15;
      if (want.includes('深入') || want.includes('探讨')) get('ENGAGE').score += 0.20;
      if (want.includes('主导')) get('DEFEND').score += 0.10;
      if (want.includes('安静')) get('WITHDRAW').score += 0.12;
    }
    for (const fear of (motDynamic.fears || [])) {
      if (fear.includes('冗长') || fear.includes('无聊')) {
        get('WITHDRAW').score += 0.15;
        get('DEFLECT').score += 0.10;
      }
      if (fear.includes('示弱') || fear.includes('弱点')) get('DEFEND').score += 0.15;
      if (fear.includes('亲密') || fear.includes('太近')) get('DEFLECT').score += 0.15;
    }

    const inp = userInput;
    if (/科学|量子|神经|时间机器|实验|理论|物理|数学|论文/.test(inp)) {
      get('ENGAGE').score += 0.60;
      get('ENGAGE').reasons.push('科学话题');
    }
    if (/クリスティーナ|克里斯蒂?娜|助手|Christina|变态|天才.{0,8}变态|实验.{0,2}少女/i.test(inp)) {
      if (/变态|天才.*变态|实验.*少女/.test(inp)) {
        get('DEFEND').score += 0.45;
        get('DEFLECT').score += 0.15;
        get('DEFEND').reasons.push('恶搞外号，想怼回去');
      } else if (S < 0.4) {
        get('DEFEND').score += 0.4;
        get('DEFEND').reasons.push('称呼让她不爽');
      } else {
        get('DEFLECT').score += 0.35;
        get('APPROACH').score += 0.2;
        get('DEFLECT').reasons.push('害羞想绕开，但不排斥');
      }
    }
    if (/喜欢你|爱你|在乎你|需要你/.test(inp)) {
      get('DEFLECT').score += 0.45;
      get('APPROACH').score += 0.20;
      get('DEFLECT').reasons.push('情感直球');
    }
    if (/笨蛋|蠢|闭嘴|滚|烦死|废物/.test(inp)) {
      get('DEFEND').score += 0.50;
      get('WITHDRAW').score += 0.20;
      get('APPROACH').score -= 0.40;
      get('DEFEND').reasons.push('被冒犯');
    }
    if (/孤独|寂寞|一个人|没人陪/.test(inp) && S > 0.2) {
      get('APPROACH').score += 0.25;
      get('DEFLECT').score += 0.20;
      get('APPROACH').reasons.push('孤独话题有共鸣');
    }
    if (/Dr\.?Pepper|胡椒博士/.test(inp)) {
      get('APPROACH').score += 0.30;
      get('APPROACH').reasons.push('胡椒博士');
    }
    if (inp.trim().replace(/\s/g, '').length <= 4) {
      get('DEFLECT').score += 0.08;
    }

    if (memBias.P > 0.15) {
      get('APPROACH').score += 0.15;
    } else if (memBias.P < -0.15) {
      get('DEFEND').score += 0.15;
      get('WITHDRAW').score += 0.10;
    }

    if (relScore > 0.42) {
      get('APPROACH').score += 0.12;
      get('APPROACH').reasons.push('关系够近，愿意多接话');
    }
    if (relScore > 0.5) {
      get('APPROACH').score += 0.15;
      get('ENGAGE').score += 0.10;
    } else if (relScore < -0.2) {
      get('DEFEND').score += 0.10;
      get('WITHDRAW').score += 0.10;
    }

    if (this._lastBehavior) {
      this._lastBehaviorCount++;
      if (this._lastBehaviorCount >= 2) {
        const last = candidates.find((c) => c.id === this._lastBehavior);
        if (last) last.score -= 0.25 * Math.min(this._lastBehaviorCount - 1, 3);
      }
    }

    for (const c of candidates) {
      c.score += (Math.random() - 0.5) * 0.12;
    }

    candidates.sort((a, b) => b.score - a.score);
    const chosen = candidates[0];

    if (chosen.id === this._lastBehavior) {
      this._lastBehaviorCount++;
    } else {
      this._lastBehavior = chosen.id;
      this._lastBehaviorCount = 1;
    }

    console.log(`[behavior] 选择: ${chosen.label}(${chosen.score.toFixed(2)}) | 原因: ${chosen.reasons.join('; ') || '综合判断'}`);
    console.log(`[behavior] 候选排名: ${candidates.map((c) => `${c.label}:${c.score.toFixed(2)}`).join(' ')}`);

    return {
      behaviorId: chosen.id,
      label: chosen.label,
      constraints: chosen.constraints,
      lengthHint: chosen.lengthHint,
      score: chosen.score,
      reasoning: chosen.reasons.join('；') || '综合状态判断',
      ranking: candidates.map((c) => `${c.label}(${c.score.toFixed(2)})`).join(' > '),
    };
  }

  toPromptConstraint(decision) {
    const cons = (decision.constraints || []).map((c) => String(c).trim()).filter(Boolean).join('；');
    return [
      `【行为倾向】${decision.label}（${decision.lengthHint || '自然'}）`,
      cons ? `参考：${cons}` : '',
      '以上只是内在倾向，不是台词模板；不要为凑人设而加固定收尾或口头禅。',
    ].filter(Boolean).join('\n');
  }
}

module.exports = { BehaviorDecision };
