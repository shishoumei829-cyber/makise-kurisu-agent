'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const { spawn } = require('node:child_process');
const path = require('node:path');

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
  const child = spawn(process.execPath, ['server.js'], {
    cwd: projectRoot,
    env: {
      ...process.env,
      AMADEUS_BACKEND_PORT: String(port),
      AMADEUS_JP_VALIDATE: '0',
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
  } finally {
    child.kill('SIGTERM');
    await new Promise((r) => setTimeout(r, 400));
    if (!child.killed) child.kill('SIGKILL');
  }
});
