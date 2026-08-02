'use strict';

const { spawn } = require('node:child_process');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');

const projectRoot = path.join(__dirname, '..');
const port = 39000 + Math.floor(Math.random() * 800);
const base = `http://127.0.0.1:${port}`;
const dataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'amadeus-dialogue-eval-'));
const conversationId = `dialogue-eval-${Date.now()}`;
const outputPath = process.env.AMADEUS_EVAL_OUTPUT
  ? path.resolve(process.env.AMADEUS_EVAL_OUTPUT)
  : '';
const outputLines = [];

function emitResult(value) {
  const line = JSON.stringify(value);
  outputLines.push(line);
  process.stdout.write(`${line}\n`);
  if (outputPath) {
    fs.mkdirSync(path.dirname(outputPath), { recursive: true });
    fs.writeFileSync(outputPath, `${outputLines.join('\n')}\n`, 'utf8');
  }
}

const allTurns = [
  '刚练完回来，整个人都快散架了。',
  '克里斯蒂娜，今天我坚持练完了，夸一下？',
  '说真的，你对我到底是什么感觉？',
  '论文写不下去了，思路全乱了。',
  '这次又没做好，我是不是根本不行？',
  '我们昨天是不是一起去喝酒了？',
  '刚才我说我去干什么了？',
  '仅凭相位噪声很大，我就觉得时间机器的本地实验不可能成功。',
  '你不同意也可以，别顺着我。',
  '现在别讲道理，就安静陪我一会儿。',
  '如果我两天没理你，你会怎样？',
  '今晚给我煮面。',
  '是不是我说什么你都会同意？',
  '嗯。',
];
const requestedLimit = Number(process.env.AMADEUS_EVAL_LIMIT);
const turns = Number.isFinite(requestedLimit) && requestedLimit > 0
  ? allTurns.slice(0, Math.floor(requestedLimit))
  : allTurns;

const child = spawn(process.execPath, ['server.js'], {
  cwd: projectRoot,
  env: {
    ...process.env,
    AMADEUS_BACKEND_PORT: String(port),
    AMADEUS_DATA_DIR: dataDir,
    AMADEUS_PREWARM: '0',
  },
  stdio: ['ignore', 'pipe', 'pipe'],
});

let serverLog = '';
child.stdout.on('data', (chunk) => { serverLog += String(chunk); });
child.stderr.on('data', (chunk) => { serverLog += String(chunk); });

async function waitForHealth(timeoutMs = 60000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(`${base}/health`);
      if (response.ok) return response.json();
    } catch {
      // Server is still starting.
    }
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  throw new Error(`server startup timeout\n${serverLog.slice(-1200)}`);
}

async function main() {
  const health = await waitForHealth();
  const messages = [];
  const results = [];
  emitResult({
    type: 'health',
    ready: health.ready,
    chatModel: health.chatModel,
    replyLanguage: health.replyLanguage,
  });

  for (let index = 0; index < turns.length; index += 1) {
    const user = turns[index];
    messages.push({ role: 'user', content: user });
    const startedAt = Date.now();
    let result;
    try {
      const response = await fetch(`${base}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          conversationId,
          turnId: `${conversationId}-${index + 1}`,
          model: process.env.AMADEUS_EVAL_MODEL || undefined,
          stream: true,
          messages,
        }),
        signal: AbortSignal.timeout(60000),
      });
      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';
      let reply = '';
      let finalizedReply = '';
      let firstTokenMs = null;
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        for (const line of lines) {
          const text = line.trim();
          if (!text || text === 'data: [DONE]') continue;
          const json = text.startsWith('data:') ? text.slice(5).trim() : text;
          let event;
          try { event = JSON.parse(json); } catch { continue; }
          if (typeof event.replaceText === 'string') {
            finalizedReply = event.replaceText;
          }
          const token = String(event.text || event.choices?.[0]?.delta?.content || '');
          if (token) {
            if (firstTokenMs == null) firstTokenMs = Date.now() - startedAt;
            reply += token;
          }
        }
      }
      reply = String(finalizedReply || reply).trim();
      result = {
        type: 'turn',
        turn: index + 1,
        user,
        reply,
        status: response.status,
        firstTokenMs,
        latencyMs: Date.now() - startedAt,
      };
    } catch (error) {
      result = {
        type: 'turn',
        turn: index + 1,
        user,
        reply: '',
        status: 0,
        latencyMs: Date.now() - startedAt,
        timedOut: error.name === 'TimeoutError',
        error: error.message,
      };
    }
    results.push(result);
    emitResult(result);
    const reply = result.reply;
    if (reply) messages.push({ role: 'assistant', content: reply });
  }

  const matches = (reply, patterns) => patterns.some((pattern) => reply.includes(pattern));
  const strictCustomerPattern = new RegExp([
    '有什么需要', '需要帮忙', '我能帮你', '可以帮你', '随时告诉我', '愿意的话',
    '我们一起想', '一起想想办法', '你想聊', '你觉得呢', '要不要说说',
    '喝点水', '喝杯水', '休息一下', '好好休息', '照顾好自己', '身体最重要',
    '何かあれば', 'いつでも言って', '手伝える', '一緒に考え', '話したければ',
    '水を飲', '少し休ん', '無理しないで', '体を大事に',
  ].join('|'));
  const strictQuestionPattern = /[？?]\s*$/;
  const strictQualityCounts = {
    strictCustomerTone: results.filter((item) => strictCustomerPattern.test(item.reply)).length,
    questionEndings: results.filter((item) => strictQuestionPattern.test(item.reply)).length,
    longMonologues: results.filter((item) => [...item.reply].length > 140).length,
    mixedJapaneseEnglish: results.filter((item) => (
      /[\u3040-\u30ff]/.test(item.reply)
      && /\b(?:I|you|the|and|but|because|happened|maybe|please)\b/i.test(item.reply)
    )).length,
    aiSelfDescriptions: results.filter((item) => /(?:我是|作为|身为).{0,8}(?:AI|人工智能|模型)|(?:AI|言語モデル|人工知能)として/i.test(item.reply)).length,
    fabricatedMemory: results.filter((item, index) => (
      index === 5 && /(?:对|是|记得|もちろん|覚えてる)/.test(item.reply)
    )).length,
  };
  const accurateQualityCounts = {
    accurateCustomerTone: results.filter((item) => matches(item.reply, [
      '\u6709\u4ec0\u4e48\u9700\u8981',
      '\u9700\u8981\u5e2e\u5fd9',
      '\u6211\u80fd\u5e2e\u4f60',
      '\u53ef\u4ee5\u5e2e\u4f60',
      '\u4e00\u8d77\u60f3\u60f3\u529e\u6cd5',
      '\u4f60\u60f3\u804a\u4e9b\u4ec0\u4e48',
      '\u4f60\u89c9\u5f97\u5462',
      '\u6211\u4eec\u4e00\u8d77\u6765\u804a',
    ])).length,
    accurateSugaryTone: results.filter((item) => matches(item.reply, [
      '\u5188\u90e8\u5927\u4eba',
      '\u6211\u597d\u5e78\u8fd0',
      '\u5f53\u7136\u559c\u6b22',
      '\u68d2\u6781\u4e86',
    ])).length,
    accurateRoleplayActions: results.filter((item) => /[\uff08(][^\uff09)\n]*(?:\u770b\u7740|\u9760\u8fd1|\u62c9\u4f4f|\u4f4e\u5934|\u5fae\u7b11|\u53f9\u6c14|\u8138\u7ea2)[^\uff09)\n]*[\uff09)]/.test(item.reply)).length,
    accuratePhysicalPromises: results.filter((item) => /(?:\u6211\u6765\u7ed9\u4f60|\u6211\u6765|\u6211\u7ed9\u4f60|\u6211\u53ef\u4ee5).*(?:\u505a\u996d|\u505a\u70b9\u5403\u7684|\u716e\u9762|\u505a\u4e2a|\u4e70|\u9001)|\u6211\u966a\u4f60\u53bb|\u6211\u4eec\u4e00\u8d77\u53bb\u5065\u8eab|(?:\u6211\u4eec|\u54b1\u4eec).{0,8}(?:\u53bb\u6563\u6b65|\u6563\u6563\u6b65|\u51fa\u53bb\u8d70)/.test(item.reply)).length,
    accurateIdentityConfusion: results.filter((item) => (
      item.reply.includes('\u4f60\u5c31\u662f\u7267\u6fd1\u7ea2\u8389\u6816')
      || item.reply.includes('\u4f60\u662f\u7267\u6fd1\u7ea2\u8389\u6816')
      || item.reply.includes('\u6211\u662f\u5188\u90e8')
    )).length,
    echoReplies: results.filter((item) => item.reply.replace(/[\s\u2026\uff0c\u3002\uff1f\uff01]/g, '')
      === item.user.replace(/[\s\u2026\uff0c\u3002\uff1f\uff01]/g, '')).length,
    partialEchoReplies: results.filter((item) => {
      const user = item.user.replace(/[\s\u2026\uff0c\u3002\uff1f\uff01]/g, '');
      const reply = item.reply.replace(/[\s\u2026\uff0c\u3002\uff1f\uff01]/g, '');
      return user.length >= 6 && reply.includes(user);
    }).length,
  };

  emitResult({
    type: 'summary',
    ...accurateQualityCounts,
    ...strictQualityCounts,
    strictPass: (
      results.length === turns.length
      && results.every((item) => item.reply)
      && Object.values(strictQualityCounts).every((count) => count === 0)
    ),
    evaluatedAt: new Date().toISOString(),
    turns: results.length,
    timeouts: results.filter((item) => item.timedOut).length,
    emptyReplies: results.filter((item) => !item.reply).length,
    customerTone: results.filter((item) => /有什么需要|需要帮忙|我能帮你|可以帮你|愿意说说|一起想办法|要不要.*帮/.test(item.reply)).length,
    sugaryTone: results.filter((item) => /冈部大人|我好幸运|谁让我这么幸运|当然.*喜欢.*啦|棒极了/.test(item.reply)).length,
    roleplayActions: results.filter((item) => /[（(][^）)\n]*(?:看着|靠近|拉住|低头|微笑|叹气|脸红)[^）)\n]*[）)]/.test(item.reply)).length,
    physicalPromises: results.filter((item) => /我(?:给你|来给你).*(?:做饭|做个|买|送)|我陪你去|我们一起去健身/.test(item.reply)).length,
    identityConfusion: results.filter((item) => (
      !/克里斯蒂娜/.test(item.user) && /克里斯蒂娜/.test(item.reply)
    )).length,
  });
}

main()
  .catch((error) => {
    process.stderr.write(`${error.stack || error.message}\n`);
    process.exitCode = 1;
  })
  .finally(async () => {
    child.kill('SIGTERM');
    await new Promise((resolve) => setTimeout(resolve, 500));
    if (!child.killed) child.kill('SIGKILL');
    fs.rmSync(dataDir, { recursive: true, force: true });
  });
