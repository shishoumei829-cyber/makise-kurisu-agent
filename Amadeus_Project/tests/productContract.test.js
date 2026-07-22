'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const root = path.join(__dirname, '..');
const read = (name) => fs.readFileSync(path.join(root, name), 'utf8');

test('desktop launch has one Electron shell and no browser-only fork', () => {
  const bat = read('一键启动.bat');
  const auto = read(path.join('scripts', 'autostart-amadeus.ps1'));
  const main = read('main.js');
  assert.match(bat, /node_modules\\electron\\dist\\electron\.exe/);
  assert.match(bat, /start "Amadeus"/);
  assert.doesNotMatch(bat, /open_browser|start "" "http:\/\/localhost/);
  assert.match(auto, /electron\.exe/);
  assert.doesNotMatch(auto, /Start-Process \$ui/);
  assert.match(main, /requestSingleInstanceLock/);
  assert.match(main, /width: 1280/);
  assert.match(main, /backgroundColor: '#000000'/);
  assert.doesNotMatch(main, /disableHardwareAcceleration/);
});

test('voice surface exposes one active-call entry and rejects unverified ambient memory', () => {
  const html = read('amadeus_work.html');
  const server = read('server.js');
  const main = read('main.js');
  assert.equal((html.match(/id="callbtn"/g) || []).length, 1);
  assert.equal((html.match(/id="micbtn"/g) || []).length, 0);
  assert.equal((html.match(/id="hearbtn"/g) || []).length, 0);
  assert.match(html, />🎙 通话<\/button>/);
  assert.match(html, /async toggleLocalCall\(/);
  assert.match(html, /_ensureMicrophoneAccess\(/);
  assert.match(html, /_submitCallUtterance\(/);
  assert.match(html, /\/asr/);
  assert.match(server, /app\.post\('\/asr'/);
  assert.match(server, /windows-asr-wav\.ps1/);
  assert.match(main, /'microphone'/);
  const ambient = server.slice(server.indexOf("app.post('/ambient-hearing'"), server.indexOf("app.get('/pad-state'"));
  assert.ok(ambient.indexOf('if (!verified)') < ambient.indexOf('memorySystem.addEvent'));
  assert.match(ambient, /reason: 'speaker_unverified'/);
});

test('vision is event-driven and VLM is only on explicit visual demand', () => {
  const html = read('amadeus_work.html');
  const idle = html.slice(html.indexOf('async _autonomyTick('), html.indexOf('forceObserve()', html.indexOf('async _autonomyTick(')));
  const send = html.slice(html.indexOf('async _userSendCore('), html.indexOf('async _captureFrame(', html.indexOf('async _userSendCore(')));
  assert.doesNotMatch(idle, /await this\._captureVision/);
  assert.match(send, /const wantVisual = this\._needsVisualAnswer\(text\)/);
  assert.match(send, /wantVisual\s*\? this\._captureVision/);
  assert.match(html, /mediapipe-face-detector/);
  assert.match(html, /local-frame-diff/);
});

test('TTS has one backend proxy route', () => {
  const html = read('amadeus_work.html');
  assert.match(html, /this\.config\.sovitsBase = apiBase/);
  assert.match(html, /fetch\(`\$\{this\.config\.sovitsBase\}\/tts`/);
});

test('local reminders are polled, displayed and acknowledged as delivered', () => {
  const html = read('amadeus_work.html');
  assert.match(html, /_startReminderLoop\(\)/);
  assert.match(html, /\/butler\/reminders\?due=1/);
  assert.match(html, /_deliverReminder\(reminder\)/);
  assert.match(html, /\/butler\/reminders\/\$\{encodeURIComponent\(reminder\.id\)\}\/delivered/);
});

test('background butler work returns to the same conversation for confirmation and completion', () => {
  const html = read('amadeus_work.html');
  const server = read('server.js');
  assert.match(server, /app\.get\('\/butler\/updates'/);
  assert.match(html, /_startButlerFollowupLoop\(\)/);
  assert.match(html, /_pollButlerUpdates\(\)/);
  assert.match(html, /source: 'butler_followup'/);
  assert.match(html, /_startBackgroundJapaneseTTS\(text\)/);
});
