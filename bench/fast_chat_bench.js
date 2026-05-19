import http from 'node:http';

const payload = JSON.stringify({ model: 'kurisu:latest', userMsg: '你好' });
const options = {
  hostname: process.env.AMADEUS_BENCH_HOST || 'localhost',
  port: Number(process.env.AMADEUS_BENCH_PORT || 3000),
  path: '/chat',
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Content-Length': Buffer.byteLength(payload),
  },
};

function parseChatBody(body) {
  try {
    const j = JSON.parse(body);
    const text = j.response || j.choices?.[0]?.message?.content || '';
    return { textLen: String(text).length, preview: String(text).slice(0, 120) };
  } catch {
    return { textLen: 0, preview: '' };
  }
}

function callOne() {
  return new Promise((resolve, reject) => {
    const start = Date.now();
    const req = http.request(options, (res) => {
      res.setEncoding('utf8');
      let body = '';
      res.on('data', (chunk) => {
        body += chunk;
      });
      res.on('end', () => {
        const { textLen, preview } = parseChatBody(body);
        resolve({
          total_ms: Date.now() - start,
          status: res.statusCode,
          textLen,
          preview,
          body: body.length > 2000 ? `${body.slice(0, 2000)}…` : body,
        });
      });
    });
    req.on('error', (e) => reject(e));
    req.write(payload);
    req.end();
  });
}

async function main(n) {
  n = n || 5;
  const results = [];
  for (let i = 0; i < n; i++) {
    try {
      results.push(await callOne());
    } catch (err) {
      results.push({ error: String(err && err.message ? err.message : err) });
    }
  }
  const lens = results.map((r) => r.textLen).filter((x) => typeof x === 'number');
  const lat = results.map((r) => r.total_ms).filter((x) => typeof x === 'number');
  const summary =
    lens.length > 0
      ? {
          n: lens.length,
          reply_len_min: Math.min(...lens),
          reply_len_max: Math.max(...lens),
          reply_len_avg: lens.reduce((a, b) => a + b, 0) / lens.length,
          latency_ms_avg: lat.length ? lat.reduce((a, b) => a + b, 0) / lat.length : null,
        }
      : {};
  console.log(JSON.stringify({ summary, runs: results }, null, 2));
}

main(process.argv[2] ? Number(process.argv[2]) : 5).catch((e) => {
  console.error(e);
  process.exit(1);
});
