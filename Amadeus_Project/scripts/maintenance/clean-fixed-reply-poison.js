'use strict';

/*
 * Removes one known set of old training/debug turns that taught the runtime to
 * respond to criticism with a canned self-defence.  This is intentionally
 * narrow: normal dialogue is preserved.  Use --apply to write changes; a
 * timestamped copy is placed in amadeus_data/quarantine for recovery.
 */

const fs = require('fs');
const path = require('path');

const dataRoot = process.env.AMADEUS_DATA_DIR || 'C:/Users/SHIKIMORI/amadeus_data';
const targetFiles = [
  'unified_dialogue_log.json',
  'memory_palace.json',
  'memory_admission.json',
  'subject_state.json',
  'autonomy_subsystem.json',
  'understanding_subsystem.json',
  'user_model.json',
].map((name) => path.join(dataRoot, name));
const jsonlFiles = [path.join(dataRoot, 'butler', 'events.jsonl')];

// 只匹配已经确认的坏助手输出，不按泛词删除用户的正常对话。
// 这些句子会被召回后反复强化“客服式说明身份”或“泛问候”。
const poison = /(?:今天有点累，但还不想听客服式安慰|我今天累得要命，别用客服话术|我今天有点累，不想听客服式的安慰|我不拿那些客服式的套话敷衍你|不拿那些客服式|别用客服话术|不想听客服式(?:的)?安慰|哎呀，怎么了？是不是有什么麻烦的事情吗？|(?:……)?(?:什么|怎么了)？又叫我克里斯蒂娜？[\s\S]{0,120}(?:神经科学研究者|我是红莉栖，你的恋人)|我是牧濑红莉栖，你的恋人。我们在一起已经有一段时间了)/;
const apply = process.argv.includes('--apply');

function hasPoison(value) {
  return poison.test(typeof value === 'string' ? value : JSON.stringify(value));
}

function cleanse(value, stats) {
  if (typeof value === 'string') return value;
  if (Array.isArray(value)) {
    const result = [];
    for (const entry of value) {
      if (hasPoison(entry)) {
        stats.removedRecords += 1;
        continue;
      }
      result.push(cleanse(entry, stats));
    }
    return result;
  }
  if (!value || typeof value !== 'object') return value;

  const result = {};
  for (const [key, entry] of Object.entries(value)) {
    if (hasPoison(key) || (typeof entry === 'string' && hasPoison(entry))) {
      stats.removedFields += 1;
      continue;
    }
    result[key] = cleanse(entry, stats);
  }
  return result;
}

for (const file of targetFiles) {
  if (!fs.existsSync(file)) continue;
  const original = fs.readFileSync(file, 'utf8');
  const data = JSON.parse(original);
  const stats = { removedRecords: 0, removedFields: 0 };
  const cleaned = cleanse(data, stats);
  const changed = stats.removedRecords > 0 || stats.removedFields > 0;
  console.log(`${path.basename(file)} records=${stats.removedRecords} fields=${stats.removedFields} changed=${changed}`);
  if (!apply || !changed) continue;

  const quarantine = path.join(dataRoot, 'quarantine');
  fs.mkdirSync(quarantine, { recursive: true });
  const backup = path.join(quarantine, `${path.basename(file)}.fixed-reply-poison-${Date.now()}.bak`);
  fs.writeFileSync(backup, original, 'utf8');
  fs.writeFileSync(file, `${JSON.stringify(cleaned, null, 2)}\n`, 'utf8');
}

for (const file of jsonlFiles) {
  if (!fs.existsSync(file)) continue;
  const original = fs.readFileSync(file, 'utf8');
  const lines = original.split(/\r?\n/);
  const kept = lines.filter((line) => line && !hasPoison(line));
  const removed = lines.filter((line) => line && hasPoison(line)).length;
  console.log(`${path.basename(file)} records=${removed} changed=${removed > 0}`);
  if (!apply || !removed) continue;
  const quarantine = path.join(dataRoot, 'quarantine');
  fs.mkdirSync(quarantine, { recursive: true });
  const backup = path.join(quarantine, `${path.basename(file)}.fixed-reply-poison-${Date.now()}.bak`);
  fs.writeFileSync(backup, original, 'utf8');
  fs.writeFileSync(file, `${kept.join('\n')}\n`, 'utf8');
}
