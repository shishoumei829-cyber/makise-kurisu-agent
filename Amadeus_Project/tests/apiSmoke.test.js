'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const { spawn } = require('node:child_process');
const path = require('node:path');
const fs = require('node:fs');
const os = require('node:os');

const projectRoot = path.join(__dirname, '..');

function waitForReady(child, timeoutMs = 45000) {
  return new Promise((resolve, reject) => {
    let buf = '';
    const timer = setTimeout(() => {
      reject(new Error('server startup timeout'));
    }, timeoutMs);
    const onData = (chunk) => {
      buf += String(chunk);
      if (/已就绪|listening/i.test(buf)) {
        clearTimeout(timer);
        child.stdout.off('data', onData);
        child.stderr.off('data', onData);
        resolve(buf);
      }
    };
    child.stdout.on('data', onData);
    child.stderr.on('data', onData);
    child.on('error', (e) => {
      clearTimeout(timer);
      reject(e);
    });
    child.on('exit', (code) => {
      if (!/已就绪|listening/i.test(buf)) {
        clearTimeout(timer);
        reject(new Error(`server exited early (${code}): ${buf.slice(-400)}`));
      }
    });
  });
}

async function fetchJson(url, opts = {}) {
  const res = await fetch(url, opts);
  const data = await res.json().catch(() => ({}));
  return { res, data };
}

test('API smoke: /health, dialogue-log, behavior-report', { timeout: 120000 }, async () => {
  const port = 37000 + Math.floor(Math.random() * 2000);
  const dataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'amadeus-api-smoke-'));
  const child = spawn(process.execPath, ['server.js'], {
    cwd: projectRoot,
    env: {
      ...process.env,
      AMADEUS_BACKEND_PORT: String(port),
      AMADEUS_JP_VALIDATE: '0',
      AMADEUS_DATA_DIR: dataDir,
      AMADEUS_PREWARM: '0',
    },
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  try {
    await waitForReady(child);
    const base = `http://127.0.0.1:${port}`;

    const health = await fetchJson(`${base}/health`);
    assert.ok(health.data && typeof health.data === 'object');
    assert.ok('ready' in health.data);
    assert.ok(Array.isArray(health.data.checks) || Array.isArray(health.data.warnings) || health.data.ok != null);

    const butlerStatus = await fetchJson(`${base}/butler/status`);
    assert.equal(butlerStatus.res.status, 200);
    assert.equal(butlerStatus.data.product, 'Amadeus Butler Kernel');
    assert.match(butlerStatus.data.completionRule, /evidence/);

    const ingested = await fetchJson(`${base}/butler/ingest`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: '帮我检查电脑运行状态', turnId: 'api-butler-1' }),
    });
    assert.equal(ingested.res.status, 200);
    assert.equal(ingested.data.created, true);
    const butlerTaskId = ingested.data.task.id;

    for (const status of ['planned', 'ready']) {
      const moved = await fetchJson(`${base}/butler/tasks/${butlerTaskId}/transition`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ status }),
      });
      assert.equal(moved.res.status, 200);
    }
    const executed = await fetchJson(`${base}/butler/tasks/${butlerTaskId}/execute`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ capabilityId: 'system.inspect' }),
    });
    assert.equal(executed.res.status, 200);
    assert.equal(executed.data.task.status, 'completed');
    assert.equal(executed.data.verification.passed, true);

    const reminderIngest = await fetchJson(`${base}/butler/ingest`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: '提醒我5分钟后喝水', turnId: 'api-reminder-1' }),
    });
    const reminderTaskId = reminderIngest.data.task.id;
    const reminderPlan = await fetchJson(`${base}/butler/tasks/${reminderTaskId}/plan`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}',
    });
    assert.equal(reminderPlan.data.task.status, 'ready');
    const reminderRun = await fetchJson(`${base}/butler/tasks/${reminderTaskId}/run`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}',
    });
    assert.equal(reminderRun.data.task.status, 'completed');
    const reminders = await fetchJson(`${base}/butler/reminders`);
    assert.ok(reminders.data.reminders.some((item) => item.taskId === reminderTaskId));

    const append = await fetchJson(`${base}/dialogue-log/append`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ role: 'user', text: 'API smoke ping', source: 'test' }),
    });
    assert.equal(append.res.status, 200);
    assert.equal(append.data.ok, true);
    assert.match(String(append.data.item?.text || ''), /smoke ping/);

    const log = await fetchJson(`${base}/dialogue-log?limit=5`);
    assert.equal(log.res.status, 200);
    assert.ok(Array.isArray(log.data.entries));
    assert.ok(log.data.entries.some((e) => /smoke ping/.test(String(e.text))));

    const behavior = await fetchJson(`${base}/behavior-report`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ source: 'test', note: 'api smoke', events: [] }),
    });
    assert.equal(behavior.res.status, 200);
    assert.equal(behavior.data.ok, true);

    const ambient = await fetchJson(`${base}/ambient-hearing`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: '房间里有点吵', confidence: 0.8 }),
    });
    assert.equal(ambient.res.status, 200);
    assert.equal(ambient.data.ok, true);
    assert.equal(ambient.data.activeChat, false);

    const visionEvent = await fetchJson(`${base}/vision-event`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ kind: 'camera_motion', score: 0.2, source: 'test' }),
    });
    assert.equal(visionEvent.res.status, 200);
    assert.equal(visionEvent.data.activeChat, false);
    assert.equal(visionEvent.data.injectedToChat, false);
    assert.equal(visionEvent.data.persisted, false);
  } finally {
    child.kill('SIGTERM');
    await new Promise((r) => setTimeout(r, 400));
    if (!child.killed) child.kill('SIGKILL');
    fs.rmSync(dataDir, { recursive: true, force: true });
  }
});
