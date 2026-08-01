'use strict';

/**
 * 验收：用户发话 → 必须收到非空中文回复；pending 时 proactive 不得开口。
 * node scripts/dialogue-acceptance.js [--base http://127.0.0.1:3002]
 */

const TURNS = [
  '你好',
  '我是冈部伦太郎',
  '准备去健身房了',
  '怎么样',
];

function arg(name) {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 ? process.argv[i + 1] : '';
}

async function postJson(base, path, body, timeoutMs = 120000) {
  const res = await fetch(`${base}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(timeoutMs),
  });
  const data = await res.json().catch(() => ({}));
  return { ok: res.ok, status: res.status, data };
}

async function main() {
  const base = (arg('base') || 'http://127.0.0.1:3002').replace(/\/$/, '');
  let health = null;
  for (let i = 0; i < 30; i += 1) {
    try {
      health = await fetch(`${base}/health`, { signal: AbortSignal.timeout(8000) }).then((r) => r.json());
      if (health?.ready) break;
    } catch (_) { /* retry */ }
    await new Promise((r) => setTimeout(r, 2000));
  }
  if (!health?.ready) {
    console.log(JSON.stringify({ ok: false, error: 'backend_not_ready' }));
    process.exit(2);
  }
  const conversationIdBase = `accept_${Date.now()}`;
  const results = [];
  let failed = 0;

  for (let i = 0; i < TURNS.length; i += 1) {
    const userText = TURNS[i];
    const conversationId = `${conversationIdBase}_${i + 1}`;
    const turnId = `${conversationId}_${Date.now()}`;

    const decide = await postJson(base, '/initiative/decide', {
      phase: 'copresence',
      alreadyTalking: true,
      dialogueStarted: true,
      lastUserText: userText,
      conversationId,
      isThinking: true,
    }, 15000);
    if (decide.data?.shouldSpeak) {
      failed += 1;
      results.push({ userText, fail: 'proactive_while_thinking', decide: decide.data });
      continue;
    }

    const chat = await postJson(base, '/chat', {
      messages: [{ role: 'user', content: userText }],
      stream: false,
      conversationId,
      turnId,
    }, 180000);

    const reply = String(
      chat.data?.response
      || chat.data?.choices?.[0]?.message?.content
      || '',
    ).trim();

    const bad = !reply
      || /根据你提供的信息|作为AI|对话系统|【情感带宽】/.test(reply)
      || /^嗯[。．.!！?？]?$/.test(reply)
      || /[（(](?:微笑|点头|温和|轻声|注视|看着|叹气|皱眉)[^）)]*[）)]/.test(reply);

    if (bad) {
      failed += 1;
      results.push({ userText, fail: reply ? 'bad_reply' : 'empty_reply', reply: reply.slice(0, 120) });
    } else {
      results.push({ userText, ok: true, reply: reply.slice(0, 120) });
    }
  }

  const report = {
    ok: failed === 0,
    model: health.chatModel,
    failed,
    total: TURNS.length,
    results,
  };
  console.log(JSON.stringify(report, null, 2));
  process.exit(failed === 0 ? 0 : 1);
}

main().catch((e) => {
  console.error(e);
  process.exit(2);
});
