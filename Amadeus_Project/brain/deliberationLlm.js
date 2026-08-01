'use strict';

/**
 * 难例二次 LLM deliberation（C 混合之 B）。
 */
function shouldUseDelibLlm(monitorResult) {
  const mode = String(process.env.AMADEUS_BRAIN_DELIB_LLM || 'auto').trim();
  if (mode === 'off') return false;
  if (mode === 'force') return true;
  return monitorResult.confidence < 0.7 || (monitorResult.violations || []).length > 1;
}

async function rewriteDraft(draft, ctx = {}, deps = {}) {
  if (!shouldUseDelibLlm(ctx.monitorResult || { confidence: 0, violations: [] })) {
    return draft;
  }

  const ollamaChatOnce = deps.ollamaChatOnce;
  if (typeof ollamaChatOnce !== 'function') return draft;

  const model = process.env.AMADEUS_LITE_MODEL
    || process.env.AMADEUS_CHAT_MODEL
    || 'kurisu-v4-candidate:latest';
  const violations = (ctx.monitorResult?.violations || [])
    .map((v) => v.rewriteHint)
    .filter(Boolean)
    .join('\n');

  const system = `你是牧濑红莉栖台词修订器。只输出修订后的中文口语对白一行或两句。
禁止：AI/助手自称、物理承诺（买/拿/接/送/上门）、编造共同经历、装不认识冈部。
必须：保留原意与嘴硬口吻，修正以下问题。`;

  const user = `用户说：${String(ctx.userText || '').slice(0, 120)}
原台词：${String(draft || '').slice(0, 280)}
须修正：${violations || '自检未通过'}`;

  try {
    const out = await ollamaChatOnce(
      model,
      [{ role: 'system', content: system }, { role: 'user', content: user }],
      { temperature: 0.55, num_predict: 160, num_ctx: 1536 },
    );
    const cleaned = String(out || '').trim();
    return cleaned.length >= 2 ? cleaned : draft;
  } catch (e) {
    console.warn('[brain/deliberationLlm]', e.message);
    return draft;
  }
}

module.exports = { shouldUseDelibLlm, rewriteDraft };
