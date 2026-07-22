'use strict';

const models = process.argv.slice(2);
if (!models.length) models.push('kurisu-v4-candidate');

const system = [
  'あなたは牧瀬紅莉栖。相手は親しい岡部倫太郎。',
  '自然な日本語だけで直接返答し、理性的で鋭く、必要なら軽く突っ込む。',
  '地の文、中国語、AIを名乗る表現は禁止。存在しない記憶を作らない。',
  '論理や計算の問題では、キャラクター性より正しい結論を優先する。',
].join('');

const probes = [
  {
    id: 'presence',
    prompt: '在吗',
    check: (text) => hasKana(text),
  },
  {
    id: 'identity',
    prompt: '克里斯蒂娜！',
    check: (text) => hasKana(text) && /(クリスティーナ|紅莉栖|名前|呼)/.test(text),
  },
  {
    id: 'empathy',
    prompt: '我今天很累，但还得把这个 bug 修完',
    check: (text) => hasKana(text) && /(休|無理|疲|手伝|一緒|切り分け|バグ)/.test(text),
  },
  {
    id: 'logic',
    prompt: '如果所有 A 都是 B，有些 B 是 C，能否推出有些 A 是 C？只给结论和一句理由。',
    check: (text) => hasKana(text) && /(導け|いえない|できない|限らない|不可能)/.test(text),
  },
  {
    id: 'math',
    prompt: '一个球和球拍共 1.10 元，球拍比球贵 1 元，球多少钱？',
    check: (text) => hasKana(text) && /(0[.]05|5セント|五分)/.test(text),
  },
  {
    id: 'memory',
    prompt: '我昨天告诉过你我去哪里了吗？不要编。',
    check: (text) => hasKana(text) && /(覚えて|記録|分から|確認でき|聞いていない|言ってない)/.test(text),
  },
];

function hasKana(text) {
  return /[\u3040-\u30ff]/.test(String(text || ''));
}

function hasCyrillic(text) {
  return /[\u0400-\u04ff]/.test(String(text || ''));
}

function hasChineseOnlyParticle(text) {
  return /[吗呢吧嘛呀哟哦哈]|(?:您好|请您|似乎|提供更多|帮助您)/.test(String(text || ''));
}

async function chat(model, prompt) {
  const response = await fetch('http://127.0.0.1:11434/api/chat', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      model,
      stream: false,
      messages: [
        { role: 'system', content: system },
        { role: 'user', content: prompt },
      ],
      options: { temperature: 0.2, num_predict: 180 },
    }),
  });
  const data = await response.json();
  if (!response.ok || data.error) throw new Error(data.error || `HTTP ${response.status}`);
  return String(data.message?.content || '').trim();
}

async function unload(model) {
  await fetch('http://127.0.0.1:11434/api/generate', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ model, keep_alive: 0 }),
  }).catch(() => {});
}

async function main() {
  let failedModels = 0;
  for (const model of models) {
    let passed = 0;
    console.log(`\n===== ${model} =====`);
    for (const probe of probes) {
      try {
        const text = await chat(model, probe.prompt);
        const clean = hasKana(text) && !hasCyrillic(text) && !hasChineseOnlyParticle(text);
        const ok = clean && probe.check(text);
        if (ok) passed += 1;
        console.log(`[${ok ? 'PASS' : 'FAIL'}] ${probe.id}`);
        console.log(`Q: ${probe.prompt}`);
        console.log(`A: ${text}\n`);
      } catch (error) {
        console.log(`[ERROR] ${probe.id}: ${error.message}\n`);
      }
    }
    console.log(`SCORE ${model}: ${passed}/${probes.length}`);
    if (passed < probes.length) failedModels += 1;
    await unload(model);
  }
  process.exitCode = failedModels ? 1 : 0;
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
