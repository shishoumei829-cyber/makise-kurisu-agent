# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### 在场感与声画一致（2026-07-22）

- **主动对话**：从「空闲冷却门控」改为在场优先（打开/看见你 → 概率先开口；冷感回拉；允许 poke/tease 但口吻仍走 `kurisu_voice`）。
- **声画锁定**：有 TTS 日文时，界面中文必须为同句硬翻；禁止流式草稿/OOC 兜底把显示换成另一套；主动气泡按 `{cn,jp}` 一一对齐。
- **日→中纠偏**：硬翻提示 + `alignLiteralCnToJapanese`（如 `もう？` 不再译成「再三」）。
- **接话焦点**：用户叙述「坏了→又好了」类转折时，禁止只抓前半段焦虑（如「还在担心吗」）。
- **默认模型**：主对话回退 `kurisu:latest`（`kurisu-v4-candidate` 微调候选保留在 env.example / finetune 文档）。
- **本地 ASR 通话**：Electron 侧 Windows ASR + `/asr` 路径；意识标签泄漏清理。
- **测试**：conversationInitiative、replyLanguage、architecture 等用例补齐。

### Amadeus v6.0 / Continuity Build（2026-07-11）

- **整机能力编排**：启动自检统一暴露对话、记忆、视觉理解、摄像头预览和语音能力，不再把可选服务伪装成已就绪。
- **首启体验**：新增本地优先的实验室首启页，展示当前能力边界并明确进入对话的产品契约。
- **对话质量门**：保留本轮关系/情绪/当前话题锚点；检测“讲/怎么了”等无信息回复并按具体意图重生成或角色化兜底。
- **具身表达**：接入现有两套红莉栖立绘资源，在待机、思考、说话和表达性状态之间切换姿态与光晕。
- **可靠性**：移除强制原生 HNSW 安装依赖，修复低上下文 502 与首页入口问题；67 项测试全部通过。
- **硬故障修复**：修复主动轻量对话上下文溢出与空回复；视觉状态无真实结果时返回空状态；关系亲密度默认改为真实积累；Electron 启动改为校验健康后端；API 冒烟测试使用临时数据目录隔离真实记忆。
- **视觉架构修正**：摄像头常驻改为本地低分辨率帧差事件，事件只写入低权重环境记忆，不进入主对话；视觉语言模型只保留给明确视觉问题/显著事件的按需调用。
- **模型输出根因修复**：为 thinking 对话模型预留正文 token，并真正压缩轻量 Prompt；避免 `message.content` 因预算耗尽为空后再靠角色兜底掩盖。
- **本地实时语音架构**：保留牧濑红莉栖 SoVITS 声色；模型按句流式产出后立即进入本地日语翻译与 SoVITS 队列，新的用户输入会取消旧音频，不接入 OpenAI 语音替换现有 TTS。
- **轻量摄像头模型**：加入 MediaPipe Tasks Vision 的 BlazeFace short-range 模型；浏览器本地检测有人/无人，不向服务器上传帧，不把每帧转成文本。
- **对话隐私修复**：移除误走主对话的个人信息提取器，并在统一日志、Prompt 与界面三层拒绝 `NAME/TRAIT/PREFER/BASIC` 内部控制输出。
- **声画一致与低延迟**：固定台词兜底默认关闭；实时模式跳过结束后的二次 LLM 改写，按较长分句提前译日并送入红莉栖 SoVITS；译日复用已加载的红莉栖主模型，并在后端启动后后台预热，避免 8GB 显存下模型切换造成下一轮冷启动。
- **通话与视觉**：恢复本地 `CALL` 连续通话入口，支持抢话中断；移除网格与粒子底纹，主舞台改为纯黑背景。
- **主动对话连接修复**：修复 `/chat/lite` 主动轮只发送 system、遗漏临时 user 消息导致 200 空回复的问题；空回复不再消耗主动配额，并新增 `PROACTIVE ARMED/ACTIVE/RETRY/OFFLINE` 实时状态。

### 本机接力深化（2026-06-25）

- **前端清理**：删除已废弃的 `_buildSystemPrompt` 及仅其使用的辅助函数；主对话 Prompt 唯由后端 `buildPrompt` 构建。
- **实录统一**：用户发言与换题经 `_appendDialogueToServer` 写入 `dialogue-log/append`，与服务端 `/chat` 去重逻辑对齐。
- **JP-first 流式融合**：流式结束后 `await` 日语定稿/校验，经 SSE `replaceText` + `modelJp` 一次性回传；实录与界面定稿一致。
- **API 冒烟**：新增 `tests/apiSmoke.test.js`（子进程启动 server，探测 `/health`、`/dialogue-log`、`/behavior-report`）。
- **README**：根文档叙事与「数字生命五模块」对齐。

### 架构深化 Phase 2（2026-06-25）

- **clientContext 管道**：前端视觉/冲动/宫殿/情境经 `POST /chat` 的 `clientContext` 注入后端 `buildPrompt`，修复此前本地 `_buildSystemPrompt` 被忽略导致上下文丢失的问题。
- **实录统一**：设计/自主/换题等路径改用 `dialogue-log/append`；主对话仍由服务端 `/chat` 落盘。
- **JP-first**：`AMADEUS_JP_FIRST=1` 时回复后先日语定稿、校验、再译中文（可选，较慢）。
- **行为感知**：前端监听页面可见性/焦点/长时间静默，经 `behavior-report` 写入 `BehaviorIngest`。
- **测试**：`architecture.test.js` 增至 64 项。

### 数字生命五大模块深度实现（2026-06-25）

> 分支：`cursor/module-01-autonomy-3e20` · PR #3  
> 测试：`npm test` 50/50 通过

#### 一、发现的问题

**1. 叙事与实现脱节**

- 路线图 `digital_life_upgrade.md` 与 `PROJECT_STATUS.md` 描述的是「数字生命」五阶段能力，但仓库里长期并存两套叙事：数字生命升级 vs AI 伴侣 MVP（根 `README.md`、`MVP_STATUS.md`）。
- 主循环 `server.js` 虽已接入 `DigitalLifeOrchestrator`，但除模块一外，其余能力多为根目录下 50～100 行的浅层骨架，**有名字、无闭环**。

**2. 模块一之外的四个子系统过浅**

| 区域 | 原状 | 缺口 |
|------|------|------|
| 自我进化 | `memory_reorganization.js`、`dream_engine.js` 各约 80 行 | 无情节分段、无睡眠阶段、无人格轨迹、RL 与数字生命未桥接 |
| 用户理解 | `emotional_resonance.js` 关键词匹配 | 无共情调节策略、无心智模型、无言外之意 |
| 具身化 | `environment.js`、`time_perception.js` 规则片段 | 无 PAD→表情/Live2D 联动，视觉状态不可观测 |
| 元认知 | `belief_revision.js` 3 条信念 | 与 `metacognition.js` 双轨并行，洞察默认被 `AMADEUS_CHAT_MINIMAL=1` 挡在对话外 |

**3. 编排与集成缺陷**

- `orchestrator.js` 直接 `new` 浅层类，与 `autonomy/` 深度子系统标准不一致。
- 冲动 `intentKey` 曾用中文键，导致 `INTENT_BEHAVIOR_MAP` 查不到、`behaviorBoosts` 为空（模块一已修，本次保留兼容）。
- RL 奖励更新需要**上一轮** `behaviorId`，但 `onUserTurn` 在行为决策之前执行，时序错位。
- 元认知 `reflectOnDecision` 需要行为结果，却与用户轮次混在同一阶段调用。
- 梦境生成后无 carryover，独处残影无法影响主动开口。
- INTERNAL STATE 面板只展示自主性，进化/理解/具身/元认知状态不可见。

**4. 持久化与兼容**

- 旧版统一写在 `digital_life_state.json`，无子系统级隔离，难以单独观测与回滚。
- 测试只覆盖浅层 API 与编排器表面字段，缺少子系统行为闭环测试。

---

#### 二、设计原则

沿用模块一（`digital_life/autonomy/`）已验证的模式：

1. **子系统独立目录**：`constants` + 领域引擎 + `index.js` 统一入口（`XxxSubsystem`）。
2. **独立持久化**：`%USERPROFILE%/amadeus_data/{subsystem}_subsystem.json`，防抖写入。
3. **可观测**：`getPublicState()` / `snapshot()` 供 `/internal-state` 与前端 INTERNAL STATE 消费。
4. **Prompt 闭环**：`buildPromptBlock()` 产出可注入 `digitalLifeCtx` 的自然语言碎片，而非硬编码话术脚本。
5. **行为闭环**：每模块有对应 `tests/*.test.js`，覆盖真实状态迁移而非空构造。
6. **向后兼容**：根目录旧文件改为 `deprecated` re-export，旧测试与 `digital_life.test.js` 不断裂。

**编排器职责划分**

```
DigitalLifeOrchestrator
├── onUserTurn()           — 用户轮：自主性 + 理解 + 具身 + 进化(RL 记上一轮)
├── afterBehaviorDecision()— 行为后：元认知反思与洞察
├── runIdleCycle()         — 空闲轮：自主环 + 记忆整理 + 梦境
├── onVision()             — 视觉：环境感知
└── buildPromptContext()   — 汇聚五模块 Prompt 块
```

---

#### 三、实现说明

##### 模块二 · 自我进化 `digital_life/evolution/`

| 文件 | 职责 |
|------|------|
| `memory_consolidation.js` | 按 `EPISODE_GAP_MS` 情节分段；类型加权关联图谱；模式/图式提取；写入 `memorySystem.addObservation` |
| `dream_engine.js` | 浅睡/深睡/REM（`SLEEP_PHASES`）；碎片叙事重组；`carryover` + `proactiveEligible` 供主动轮 |
| `personality_trajectory.js` | 五特质漂移、里程碑（`PERSONALITY_MILESTONES`）；与 `learning_engine.PersonalityEvolution` 双向同步 |
| `rl_bridge.js` | 封装 Q-learning 更新与 `behaviorBias` EMA；`recordTurn` 使用**上一轮** behaviorId |
| `index.js` | `EvolutionSubsystem`；`runIdleCycle` 15 分钟节流整理 + 梦境触发 |

##### 模块三 · 用户理解 `digital_life/understanding/`

| 文件 | 职责 |
|------|------|
| `emotional_resonance.js` | 扩展情绪词表；连续情绪 streak；调节模式 `shield` / `validate_first` / `warm_match` 等 → PAD 增量 |
| `mental_model.js` | 从用户文本与 BDI 推断信念/欲望/意图；偏好图谱；`buildHypothesis` 心智假设 |
| `subtext.js` | `SUBTEXT_PATTERNS` 检测掩饰/试探/撤回/求关注；输出 `pendingNeed` |
| `index.js` | `UnderstandingSubsystem` 统一 `onConversationTurn` 输出共情行、言外之意、PAD 增量 |

##### 模块四 · 具身化 `digital_life/embodiment/`

| 文件 | 职责 |
|------|------|
| `environment.js` | `SCENE_LEX` 场景标签；用户可用性 busy/away/distracted；情境压力 `situationalPressure` |
| `time_perception.js` | 从 UserModel 学习活跃时段；特殊日期；离常规节律提示 |
| `expression_mapper.js` | PAD → `EXPRESSION_PRESETS` 最近邻；sprite 索引 0–5；CSS 滤镜；TTS 情绪 hint |
| `index.js` | `EmbodimentSubsystem`；`onVision` + 每轮表情更新 |

##### 模块五 · 元认知 `digital_life/metacognition/`

| 文件 | 职责 |
|------|------|
| `belief_revision.js` | 四域信念（self/user/world/relationship）；证据链；与洞察双向修正 |
| `reflection_loop.js` | 决策回顾；习惯/回避/套话偏见检测；`generateInsight` → `insightToGoalInjection` |
| `index.js` | `MetacognitionSubsystem`；在 `afterBehaviorDecision` 阶段调用，避免时序错误 |

##### 主循环集成（`server.js`）

- `onUserTurn` 传入 `rl`、`previousBehaviorId: lastChatBehaviorId`、`externalTraits`。
- 行为决策后调用 `digitalLife.afterBehaviorDecision()`，洞察注入 `goalSystem`（`METACOG_INSIGHT`）。
- `buildPromptContext` 增加 `mentalModelLine`、`subtextLine`、`pendingNeed`、`timeLine`、梦境 proactive hint。
- `/internal-state` 返回 `expression` 快照。

##### 前端（`amadeus_work.html`）

- INTERNAL STATE 新增：梦境残念、人格漂移、记忆联想、言外之意、心智模型、共情模式、表情/sprite、环境、时间感、元认知洞察与信念冲突。
- `_applyExpression()`：按 `spriteIndex` 切换 `assets/Live2d/kurisu/{n}.png`，应用 PAD 滤镜。

##### 自主行为环增强（`autonomy/behavior_loop.js`）

- 支持 `dreamCarryover`：REM/长空闲梦境残念可触发 `SPEAK` 与 `speakHint`。

##### 测试

新增 `tests/evolution.test.js`、`understanding.test.js`、`embodiment.test.js`、`metacognition.test.js`；扩展 `digital_life.test.js`；`package.json` test 脚本纳入全套。**50 项全绿**。

##### 迁移说明

- 首次启动若仅有旧 `digital_life_state.json`，编排器 `load()` 会将 `memoryReorg` / `dream` / `beliefs` / `resonance` / `time` / `environment` 迁入各子系统（子系统文件不存在时）。
- 根目录 `memory_reorganization.js`、`dream_engine.js` 等保留为兼容 re-export，新代码应引用子系统路径。

---

### Added

- Added `README.md` (CN) and `README_EN.md` for product/open-source onboarding
- Added `CONTRIBUTING.md`, `SECURITY.md`, and `RESUME_PITCH.md`
- Added minimal Node test suite (`tests/*.test.js`) for core pure modules
- Added GitHub Actions CI workflow at `.github/workflows/ci.yml`
- Added `release:win` npm script for Windows package build

### Changed

- Updated root `.gitignore` with runtime/cache/binary exclusions
- Updated Electron `main.js` to auto-ensure backend service on startup and stop spawned backend on quit
- Updated `启动实验终端.bat` to use relative path for portability

### Open Source & Distribution

- Added root `README.md`, `LICENSE` (ISC), and `NOTICE.md` (character IP boundary)
- Added `INSTALL.md` and `scripts/install-check.bat` for end users
- Added GitHub Issue/PR templates
- Fixed Electron packaging (`ELECTRON_RUN_AS_NODE`, `signAndEditExecutable: false`)
- Added `scripts/build-release.bat` for reproducible Windows installer builds
- Generated installer: `dist/Amadeus Setup 1.0.0.exe`
- Added frontend startup banner and block chat when system not ready
- Gated debug logging behind `AMADEUS_DEBUG=1`
- Removed hardcoded SoVITS paths from `run_backend.bat` and `env.example`
- Unified backend port via `AMADEUS_BACKEND_PORT`
- Added `MVP_STATUS.md` non-technical product status doc
- Added `lib/startupCheck.js` and tests
