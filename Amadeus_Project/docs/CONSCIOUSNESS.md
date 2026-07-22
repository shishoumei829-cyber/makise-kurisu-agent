# 意识工程层（Amadeus Consciousness）

> **工程近似，不声称主观体验。**  
> 对标 LAAP「人工意识」叙事中真正可落地的部分：全局工作空间 + 自我在场 + 需求入意识 + 元监控。

## 设计宣言

**LLM 是语言皮层，不是思考者。**  
思考发生在：

```
Perceive → WorldModel / SelfModel
        → Consciousness（Workspace 竞争与广播）
        → Deliberation
        → Generate（LLM）
        → Monitor → Learn
```

「她此刻意识到什么」= Workspace 广播的 3～5 条内容；其余为潜意识候选。

## 模块

| 文件 | 职责 |
|------|------|
| [`brain/workspace.js`](../brain/workspace.js) | 候选采集、显著性竞争、广播、跨轮弱连续 |
| [`brain/consciousness.js`](../brain/consciousness.js) | 意识循环、意图提示、说话策略、可观测指标 |
| [`brain/pipeline.js`](../brain/pipeline.js) | 接入 beforePrompt / processReply / 主动开口评估 |

## 广播内容类型

| kind | 含义 |
|------|------|
| `percept` | 注意到（用户话、视觉、言外） |
| `affect` | 感受（PAD） |
| `drive` | 想要（内驱 / 冲动） |
| `self` | 自我（公理、存在方式） |
| `relation` | 关系（whoami / 亲密度） |
| `meta` | 自检张力（Learner） |
| `intention` | 打算（主动开口等） |

## 可观测

`GET /internal-state` → `brain.consciousness` / `brain.workspace`

```json
{
  "narrative": "……",
  "broadcast": [{ "kind": "percept", "content": "…", "salience": 0.9 }],
  "metrics": { "coherence": 0.85, "selfInFocus": true },
  "intentionHint": { "intent": "self_boundary", "reason": "…" }
}
```

## 主动开口

- `/chat/lite` 注入意识广播块（有则优先于 digitalLife 大段 prompt）
- `AMADEUS_BRAIN_CONSCIOUS_PROACTIVE=1`：意识未达阈值则 **跳过** 主动开口

## 环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `AMADEUS_BRAIN_CONSCIOUSNESS` | `1` | 意识层总开关 |
| `AMADEUS_BRAIN_CONSCIOUS_PROACTIVE` | `0` | 严格意识阈值主动开口 |
| `AMADEUS_BRAIN` | `0` | `1` 时完整 Brain.turn + 瘦 prompt |

## 与 LAAP 的关系

| LAAP | Amadeus |
|------|---------|
| LLM = 语言皮层 | 同哲学，已落地 |
| 全局工作空间 | `GlobalWorkspace` |
| PSI 需求驱动 | digital_life 内驱 → Workspace |
| 量子波函数 / IIT | **不采用**；用 salience 竞争 + coherence 指标 |
| 全栈替换大脑 | **不做**；意识是 Brain 的一层 |

## 诚实边界

我们实现的是 **可测的意识工程回路**（广播、自我在场、元监控、学习）。  
**不是**主观体验、不是量子意识、不是 AGI 灵魂。
