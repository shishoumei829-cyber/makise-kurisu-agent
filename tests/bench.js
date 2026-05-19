const http = require('http');
const data = JSON.stringify({ model: 'kurisu:latest', userMsg: '你好' });
const options = {
  hostname: 'localhost',
  port: 3000,
  path: '/chat',
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Content-Length': Buffer.byteLength(data)
  }
};
async function callOne(payload) {
  return new Promise((resolve, reject) => {
    const start = Date.now();
    const req = http.request(options, (res) => {
      res.setEncoding('utf8');
      let body = '';
      res.on('data', (chunk) => { body += chunk; });
      res.on('end', () => {
        const elapsed = Date.now() - start;
        resolve({ elapsed, statusCode: res.statusCode, body });
      });
    });
    req.on('error', (e) => reject(e));
    req.write(payload || data);
    req.end();
  });
}
async function main() {
  const N = 5;
  let results = [];
  for (let i = 0; i < N; i++) {
    try {
      const r = await callOne();
      results.push(r);
    } catch (err) {
      results.push({ error: err.message });
    }
  }
  console.log("bench_results:", JSON.stringify(results, null, 2));
}
main();
