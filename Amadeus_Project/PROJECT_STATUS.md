# Amadeus 数字生命体系统 - 项目状态文档

## 项目概述
牧濑红莉栖 AI 数字生命体伴侣系统，基于《命运石之门》角色。
本地 LLM（Ollama）+ 自建情感/认知状态机 + RAG 长期记忆 + Electron 桌面壳。

---

## 目录结构

```
Amadeus_Project/
├── server.js              — Express 后端入口（端口 3000）
├── main.js                — Electron 主进程
├── preload.js             — Electron 安全预加载脚本（contextBridge）
├── amadeus_work.html      — 前端单页应用（赛博风 UI）
├── ingest.js              — RAG 索引构建脚本
│
├── cognitive/             — 认知子系统（从 server.js 拆出）
│   ├── motivation.js      — MotivationSystem（固定+动态动机）
│   ├── goals.js           — InternalGoalSystem（内生目标）
│   ├── strategy.js        — StrategyLayer（跨轮次策略）
│   ├── selfModel.js       — SelfModel（自我认知模型）
│   ├── behavior.js        — BehaviorDecision（多路径行为决策）
│   ├── pad.js             — PAD 状态管理 + 事件推断
│   ├── prompts.js         — buildPrompt + 辅助工具函数
│   ├── replyAlign.js      — 焦点行、RAG 过滤、Markdown 剥离
│   └── chatTurns.js       — 解析多轮 messages，对齐 Ollama chat
│
├── lib/
│   └── memory.js          — MemorySystem（事件/时间线/观察/模式）
│
├── autonomy_enhanced.js   — CuriosityEngine（好奇心驱动）
├── bdi_engine.js          — BDI 推断引擎（信念/欲望/意图，LLM驱动）
├── learning_engine.js     — ReinforcementLearning + PersonalityEvolution
├── metacognition.js       — SelfReflection + ValueConsistency
├── user_model.js          — UserModel + ConversationAnalytics + HabitExtractor
├── design_engine.js       — 设计任务（Ollama brief + Stable Diffusion 出图）
│
├── brain_data/            — RAG 知识库文本（.txt）
├── vector_store/          — RAG 向量索引（运行时生成）
│
├── kurisu_soul.txt        — 灵魂文本（自我叙述，长）
├── kurisu_core_prompt.txt — 人格硬约束（短，高优先级）
└── kurisu_character_rules.txt — 行为规则（可选）
```

---

## 系统架构

### 后端模块
| 模块 | 文件 | 功能 |
|------|------|------|
| **多轮对话转发** | `server.js` + `cognitive/chatTurns.js` | 将 `messages` 中的 `user`/`assistant` 原样交给 Ollama（不再合并成单条 user），模型能看到自己上一轮 assistant 正文 |
| 记忆系统 | `lib/memory.js` | 事件权重、长期PAD影响、关系演化 |
| 动机系统 | `cognitive/motivation.js` | 固定核心动机 + 状态驱动动态动机 |
| 行为决策 | `cognitive/behavior.js` | 5种行为候选 + 6层打分 |
| 自我模型 | `cognitive/selfModel.js` | 身份标签、自我感知、关系感知 |
| 内生目标 | `cognitive/goals.js` | 好奇心/测试/暴露/话题探索等冲动 |
| 策略延续 | `cognitive/strategy.js` | 观察→信任→维持→退缩→深度投入 |
| PAD 状态 | `cognitive/pad.js` | 非线性情感更新 + 事件推断 |
| Prompt 构建 | `cognitive/prompts.js` | buildPrompt + 辅助工具函数 |
| 用户理解 | `user_model.js` | 情绪/话题/关系分析，**含 BDI** |
| BDI 推断 | `bdi_engine.js` | 每5轮异步推断用户信念/欲望/意图 |
| 学习引擎 | `learning_engine.js` | 强化学习偏置 + 人格演化 |
| 元认知 | `metacognition.js` | 自我反思、价值观一致性检查 |
| 自主性 | `autonomy_enhanced.js` | 好奇心引擎，知识缺口检测 |
| 设计出图 | `design_engine.js` | Ollama brief + A1111 生图 |
| RAG | `ingest.js` / server.js | LangChain + hnswlib 向量检索 |

### 前端
- `amadeus_work.html`：PAD面板、记忆宫殿、视觉系统、聊天界面
- 通过 `fetch` 调用 `localhost:3000` 后端 API

### Electron 桌面壳
- `main.js` + `preload.js`：无边框置顶窗口，托盘，全局快捷键 `Ctrl+H`
- 安全架构：`contextIsolation: true`，通过 `preload.js` 的 `contextBridge` 暴露 `window.amadeus`

---

## 运行时数据目录

所有持久化数据存放在 **`%USERPROFILE%\amadeus_data\`**（`os.homedir()/amadeus_data`），
不在项目目录内，防止文件监听器触发页面刷新。

| 文件 | 用途 |
|------|------|
| `pad_state.json` | PAD 情感状态 |
| `event_log.json` | 记忆事件日志 |
| `motivation.json` | 动机状态 |
| `self_model.json` | 自我认知 |
| `strategy.json` | 当前策略 |
| `user_profile.json` | 用户档案（旧） |
| `user_model.json` | 用户画像（含 BDI、情绪历史） |
| `whoami.json` | 用户身份档案 |
| `learning_state.json` | 强化学习偏置 |
| `personality_evolution.json` | 人格演化状态 |
| `metacognition_*.json` | 元认知状态 |

---

## API 端点（完整）

| 端点 | 方法 | 功能 |
|------|------|------|
| `/` | GET | 主页（amadeus_work.html） |
| `/chat` | POST | 主对话接口（流式/非流式） |
| `/pad-state` | GET | PAD 情感状态 |
| `/internal-state` | GET | 完整内部状态（含学习/元认知） |
| `/save-memory` | POST | 保存观察记录 |
| `/client-debug` | POST | 客户端调试日志 |
| `/vision` | POST | 摄像头视觉分析 |
| `/get-vision-status` | GET | 视觉分析状态 |
| `/vision-feedback` | POST | 视觉反馈 |
| `/tts-health` | GET | TTS 服务健康检查 |
| `/tts` | POST | GPT-SoVITS 语音合成代理 |
| `/whoami` | GET/POST | 用户身份档案读写 |
| `/design` | POST | 设计任务（Ollama + SD） |
| `/ollama/:api` | POST | Ollama API 反向代理 |

---

## 启动方式

```bash
# 启动后端服务器（推荐）
npm run dev          # node server.js

# 或批处理
一键启动.bat          # 后台启动 + 自动打开浏览器
run_backend.bat      # 仅后端（含 npm install --omit=dev）

# 构建 RAG 索引（首次或更新 brain_data/ 后执行）
npm run ingest

# Electron 桌面壳
npm start
```

## 依赖服务

- **Ollama**：`http://127.0.0.1:11434`（必须，对话/嵌入）
- **GPT-SoVITS**（可选）：`http://localhost:9880`（TTS 语音）
- **Stable Diffusion WebUI**（可选）：见 `design_config.json`（设计出图）

## 模型

| 模型 | 用途 |
|------|------|
| `kurisu:latest` | 主对话模型 |
| `nomic-embed-text` | RAG 嵌入 |
| `llama3.2-vision` | 视觉分析（可选） |
| `llama3.2:latest` 或其他 | BDI 推断（`AMADEUS_BDI_MODEL` 环境变量） |

## 环境变量（见 env.example）

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `AMADEUS_OLLAMA_BASE` | `http://127.0.0.1:11434` | Ollama 地址 |
| `AMADEUS_OLLAMA_NUM_CTX` | `2048` | 上下文长度（8GB 显存建议） |
| `AMADEUS_MAX_PROMPT_CHARS` | `6000` | 最大 Prompt 字符数 |
| `AMADEUS_CHAT_HISTORY_MSGS` | `24` | 发给 Ollama 的多轮条数上限（user+assistant 合计） |
| `AMADEUS_OLLAMA_KEEP_ALIVE` | `2m` | 模型保活时间 |
| `AMADEUS_OLLAMA_REPEAT_PENALTY` | `1.12` | 重复惩罚系数 |
| `AMADEUS_RAG_MS` | `900` | RAG 超时（毫秒） |
| `AMADEUS_CHAT_MINIMAL` | `1` | 关闭则启用元认知洞察注入 |
| `AMADEUS_BDI_MODEL` | `llama3.2:latest` | BDI 推断模型 |
| `AMADEUS_SOVITS_URL` | `http://localhost:9880` | SoVITS 服务地址 |
| `AMADEUS_SOVITS_REF` | — | 参考音频路径 |

## GPU / Ollama 加速

- 诊断脚本：`scripts/diag_ollama_gpu.ps1`
- 详细文档：[docs/GPU_OLLAMA_SOVITS.md](docs/GPU_OLLAMA_SOVITS.md)
- 建议在 Ollama 环境设置 `OLLAMA_NUM_GPU=999` 和 `OLLAMA_FLASH_ATTENTION=1`

## 注意事项

1. 运行时数据在 `%USERPROFILE%\amadeus_data\`，不在项目目录。
2. `Amadeus_Project/Amadeus_Project/` 子目录为历史旧版副本，勿修改。
3. 修改 `kurisu_soul.txt` 后需重启 server.js（加载时缓存）。
4. RAG 索引更新需重新运行 `npm run ingest`。
