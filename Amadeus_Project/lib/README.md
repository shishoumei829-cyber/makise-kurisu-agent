# lib/ 模块说明

## 现状

- `memory.js` — 由 `server.js` 引用，是唯一留在此目录的运行时模块。

## 迁移历史

`behavior.js`、`goals.js`、`motivation.js`、`self_model.js` 原为旧的轻量草案提取，
与 `server.js` 内联实现存在分歧，已于重构时删除。

真正的实现已迁移至 **`cognitive/`** 目录：

| 模块 | 路径 |
|------|------|
| MotivationSystem   | `cognitive/motivation.js` |
| InternalGoalSystem | `cognitive/goals.js` |
| StrategyLayer      | `cognitive/strategy.js` |
| SelfModel          | `cognitive/selfModel.js` |
| PAD 状态管理        | `cognitive/pad.js` |
| Prompt 构建工具     | `cognitive/prompts.js` |

## 规则

- 新增认知子系统请放入 `cognitive/`，不要再放 `lib/`。
- `lib/memory.js` 与 `server.js` 同步维护，改动须同步测试。
