'use strict';

const fs = require('fs');
const path = require('path');

const DEFAULT_CONFIG = {
  ollamaBase: 'http://127.0.0.1:11434',
  planningModel: 'qwen3-coder:480b-cloud',
  providerOrder: ['a1111'],
  a1111: {
    baseUrl: 'http://127.0.0.1:7860',
    steps: 28,
    cfg_scale: 6.5,
    sampler_name: 'DPM++ 2M Karras',
  },
};

function ensureDir(dir) {
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
}

function readJson(file, fallback) {
  try {
    if (fs.existsSync(file)) return { ...fallback, ...JSON.parse(fs.readFileSync(file, 'utf8')) };
  } catch {}
  return fallback;
}

function ratioToSize(ratio) {
  const table = {
    '1:1': [1024, 1024],
    '16:9': [1344, 768],
    '9:16': [768, 1344],
    '4:3': [1152, 864],
    '3:4': [864, 1152],
  };
  return table[ratio] || table['1:1'];
}

function extractJson(text) {
  const raw = String(text || '').replace(/```json|```/g, '').trim();
  const start = raw.indexOf('{');
  const end = raw.lastIndexOf('}');
  if (start < 0 || end <= start) return null;
  try { return JSON.parse(raw.slice(start, end + 1)); } catch { return null; }
}

async function ollamaGenerate(config, prompt, model) {
  const r = await fetch(`${config.ollamaBase}/api/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: model || config.planningModel,
      prompt,
      stream: false,
      options: { temperature: 0.35, num_predict: 900, num_ctx: 4096 },
    }),
    signal: AbortSignal.timeout(45000),
  });
  if (!r.ok) throw new Error(`Ollama ${r.status}`);
  return (await r.json()).response || '';
}

function fallbackBrief(request, options) {
  const ratio = options.ratio || '1:1';
  const style = options.style || 'cinematic editorial interface design';
  return {
    title: 'Design Draft',
    intent: request,
    visual_direction: `high-end ${style}, strong composition, readable subject, deliberate lighting`,
    prompt: `${request}, ${style}, cohesive visual system, strong focal point, layered composition, refined details, high quality, professional design, crisp edges, balanced contrast`,
    negative_prompt: 'low quality, blurry, noisy, malformed anatomy, bad hands, unreadable text, watermark, logo artifacts, jpeg artifacts, cluttered composition, oversaturated',
    ratio,
    evaluation: [
      '主题是否第一眼可识别',
      '构图是否有明确视觉中心',
      '是否避免 AI 常见伪影和错误文字',
    ],
  };
}

async function createBrief(config, request, options = {}) {
  const prompt = `你是一个严苛的视觉设计总监。把用户需求转成可执行的图像生成设计方案。
要求：
- 不要说空话，要给出视觉方向、主体、构图、色彩、材质、镜头/画面语言。
- 文字元素尽量建议后期排版，不要依赖图像模型生成长文字。
- 输出严格 JSON，不要 markdown。

用户需求：${request}
偏好风格：${options.style || '未指定'}
画幅：${options.ratio || '1:1'}

JSON schema:
{
  "title": "短标题",
  "intent": "设计目标",
  "visual_direction": "视觉方向",
  "prompt": "英文出图 prompt",
  "negative_prompt": "英文 negative prompt",
  "ratio": "1:1/16:9/9:16/4:3/3:4",
  "evaluation": ["评估标准1", "评估标准2", "评估标准3"]
}`;

  try {
    const text = await ollamaGenerate(config, prompt, options.model);
    const parsed = extractJson(text);
    if (parsed?.prompt) {
      return {
        ...fallbackBrief(request, options),
        ...parsed,
        ratio: parsed.ratio || options.ratio || '1:1',
      };
    }
  } catch (e) {
    return { ...fallbackBrief(request, options), planner_error: e.message };
  }
  return fallbackBrief(request, options);
}

async function renderA1111(config, brief, outputDir) {
  const [width, height] = ratioToSize(brief.ratio);
  const body = {
    prompt: brief.prompt,
    negative_prompt: brief.negative_prompt,
    width,
    height,
    steps: config.a1111.steps,
    cfg_scale: config.a1111.cfg_scale,
    sampler_name: config.a1111.sampler_name,
    batch_size: 1,
    n_iter: 1,
    save_images: false,
  };

  const r = await fetch(`${config.a1111.baseUrl}/sdapi/v1/txt2img`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(180000),
  });
  if (!r.ok) throw new Error(`A1111 ${r.status}: ${(await r.text()).slice(0, 120)}`);
  const data = await r.json();
  const image = data.images?.[0];
  if (!image) throw new Error('A1111 returned no image');

  ensureDir(outputDir);
  const fileName = `design_${Date.now()}.png`;
  const filePath = path.join(outputDir, fileName);
  fs.writeFileSync(filePath, Buffer.from(image.replace(/^data:image\/\w+;base64,/, ''), 'base64'));
  return { provider: 'a1111', fileName, filePath, url: `/design_outputs/${fileName}`, width, height };
}

function critiqueBrief(brief, render) {
  const warnings = [];
  if ((brief.prompt || '').length < 80) warnings.push('prompt 信息密度偏低，可能导致画面泛化。');
  if (/text|letter|typography|文字/i.test(brief.prompt || '')) warnings.push('图像模型生成文字不稳定，建议图片完成后再单独排版文字。');
  if (!render) warnings.push('共同设计模式：只输出设计方案，不调用图像生成。');
  return {
    score: Math.max(0.45, 0.86 - warnings.length * 0.08),
    warnings,
  };
}

async function createDesignTask({ rootPath, dataDir, request, options = {} }) {
  const configPath = path.join(rootPath, 'design_config.json');
  const config = readJson(configPath, DEFAULT_CONFIG);
  const outputDir = path.join(dataDir || rootPath, 'design_outputs');
  const brief = await createBrief(config, request, options);

  let render = null;
  let render_error = null;
  if (options.render !== false) {
    for (const provider of config.providerOrder || []) {
      try {
        if (provider === 'a1111') {
          render = await renderA1111(config, brief, outputDir);
          break;
        }
      } catch (e) {
        render_error = `${provider}: ${e.message}`;
      }
    }
  }

  const critique = critiqueBrief(brief, render);
  const record = {
    id: `design_${Date.now()}`,
    created_at: new Date().toISOString(),
    request,
    brief,
    render,
    render_error,
    critique,
  };

  ensureDir(outputDir);
  fs.writeFileSync(path.join(outputDir, `${record.id}.json`), JSON.stringify(record, null, 2));
  return record;
}

module.exports = { createDesignTask };
