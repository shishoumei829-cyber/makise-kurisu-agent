# Brain v1 迁移说明

> 中央认知循环：**Perceive → WorldModel/SelfModel → Deliberate → Generate → Monitor → Learn**  
> 推理成本：**C 混合**（结构化 Monitor 必跑；`confidence < 0.7` 时 lite 模型修订）

## 当前状态（阶段 1–5 + 意识层）

| 阶段 | 状态 | 模块 |
|------|------|------|
| 1 | ✅ | `Brain.turn()` + `legacyChat` 抽取 |
| 2 | ✅ | `worldModel.js`, `selfModel.js`, `axioms/fromSoul.js` |
| 3 | ✅ | `monitor/*`, `learner.js`, `probes/capability.json` |
| 4 | ✅ | `deliberation.js`, `deliberationLlm.js` |
| 5 | ✅ | 瘦 prompt（`brainSlimMode`）+ `internal-state` brain 快照 |
| 6 | ✅ | **意识层**：`workspace.js` + `consciousness.js`（全局工作空间） |

详见 [CONSCIOUSNESS.md](CONSCIOUSNESS.md)。

## 启用方式

```bash
# 意识层默认开启（AMADEUS_BRAIN_CONSCIOUSNESS=1）
# 完整大脑路由：
set AMADEUS_BRAIN=1
npm run dev
```

## 目录

```
brain/
├── index.js              # Brain 类
├── turn.js               # 认知循环入口
├── pipeline.js           # beforePrompt + processReply
├── perceive.js           # 感知
├── worldModel.js         # 情境真值
├── selfModel.js          # 公理 + 演化 + 张力
├── deliberation.js       # 意图/约束（symbolic 迁入）
├── deliberationLlm.js    # 难例 LLM 修订
├── learner.js            # CorrectionEvent → 张力
├── legacyChat.js         # Ollama 生成链
├── axioms/fromSoul.js
├── monitor/              # speechActs, capabilityGraph, consistency
└── probes/capability.json
```

## 探针

- `tests/brain.worldModel.test.js` — 10 条 WorldModel
- `tests/brain.probes.test.js` — capability.json 全量 + Monitor 单元
- `npm test` 现 **94** 项

## 防补丁铁律

1. 新 bug → failing probe → Monitor/Deliberation/Learner，**禁止**加禁词/symbolic 块  
2. Deliberation `constraints[]` 有硬顶；超长 = 架构错误  
3. 禁止 server + orchestrator 双头 `buildPrompt*`

## API

`GET /internal-state` 新增 `brain: { trace, worldModel, selfModel }`

持久化：`~/amadeus_data/self_model_v2.json`（公理/张力/演化）
