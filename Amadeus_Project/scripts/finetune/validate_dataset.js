#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '../..');
const TRAIN = path.join(ROOT, 'data', 'finetune', 'kurisu_sft.jsonl');

function main() {
  if (!fs.existsSync(TRAIN)) {
    console.error('[validate] 请先运行: node scripts/finetune/build_sft_dataset.js');
    process.exit(1);
  }
  const lines = fs.readFileSync(TRAIN, 'utf8').split(/\r?\n/).filter(Boolean);
  let bad = 0;
  let jaAssist = 0;
  let cnUser = 0;
  let risky = 0;
  for (const line of lines) {
    let row;
    try {
      row = JSON.parse(line);
    } catch {
      bad++;
      continue;
    }
    const msgs = row.messages || [];
    const sys = msgs.find((m) => m.role === 'system');
    const user = msgs.find((m) => m.role === 'user');
    const asst = msgs.find((m) => m.role === 'assistant');
    if (!sys || !user || !asst) bad++;
    if (asst && /[\u3040-\u30ff]/.test(asst.content)) jaAssist++;
    if (user && /[\u4e00-\u9fff]/.test(user.content)) cnUser++;
    if (
      !sys
      || String(sys.content || '').length > 800
      || !asst
      || !/[\u3040-\u30ff]/.test(String(asst.content || ''))
      || /[\u0400-\u04ff]/.test(String(asst.content || ''))
      || String(user?.content || '').trim() === String(asst?.content || '').trim()
    ) {
      risky++;
    }
  }
  console.log(`[validate] 样本数 ${lines.length}, 解析失败 ${bad}`);
  console.log(`[validate] assistant 含假名/日语 ${jaAssist}/${lines.length}`);
  console.log(`[validate] user 含中文 ${cnUser}/${lines.length}`);
  console.log(`[validate] 高风险样本 ${risky}/${lines.length}`);
  if (bad > 0 || risky > 0 || lines.length < 30) process.exit(1);
}

main();
