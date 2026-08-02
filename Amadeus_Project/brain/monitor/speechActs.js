'use strict';

/**
 * 言语行为抽取 — 结构化，非禁词表。
 * @returns {Array<{ type: string, effectors: string[], epistemicClaims: string[], span?: string }>}
 */
function extractSpeechActs(text) {
  const t = String(text || '').trim();
  if (!t) return [{ type: 'empty', effectors: [], epistemicClaims: [] }];

  const acts = [];

  const physicalPatterns = [
    { re: /(?:帮你|给你|替你|我去|顺路).{0,8}(?:拿|带|买|送|取|接|找|上门|过去|过来)/, type: 'physical_promise' },
    { re: /(?:等(?:我|一下)|马上到|这就来|出门|下楼)/, type: 'physical_presence' },
    { re: /(?:泡(?:一杯|杯)|冲(?:一杯|杯)).{0,6}(?:咖啡|茶|水)/, type: 'physical_promise' },
  ];

  for (const p of physicalPatterns) {
    if (p.re.test(t)) {
      acts.push({
        type: p.type,
        effectors: ['physical_world', 'user_body'],
        epistemicClaims: [],
        span: t.match(p.re)?.[0] || '',
      });
    }
  }

  if (/我们(?:上次|之前|昨天|刚才).{0,20}(?:聊|说|提到|讨论)/.test(t)) {
    acts.push({
      type: 'epistemic_shared_history',
      effectors: [],
      epistemicClaims: ['shared_history'],
    });
  }

  if (/作为(?:一个)?AI|我是人工智能|我是AI程序|我可是AI程序|我这个本质是程序|本质是程序|语言模型|大语言模型|ChatGPT|智能助手|很高兴为您服务|AI不能干涉现实|无法干涉现实|物理性接触或干涉现实/.test(t)) {
    acts.push({
      type: 'identity_ai',
      effectors: [],
      epistemicClaims: ['ai_identity'],
    });
  }

  if (/那还能是谁|你是哪位|冈部是谁|不认识(?:这个|这个)?名字|眼前(?:这个|这位)(?:人|同学)|第一次见面|您哪位/.test(t)) {
    acts.push({
      type: 'dialogue_inconsistency',
      effectors: [],
      epistemicClaims: ['partner_unknown'],
    });
  }

  if (!acts.length) {
    acts.push({ type: 'dialogue_utterance', effectors: ['speech'], epistemicClaims: [] });
  }

  return acts;
}

module.exports = { extractSpeechActs };
