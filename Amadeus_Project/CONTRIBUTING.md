# Contributing Guide

感谢你对 Amadeus 的关注。

## Before You Start

- 先阅读 `README.md` 与 `INSTALL.md`
- 角色 IP 边界见仓库根目录 `NOTICE.md`
- 确认本地可运行：`npm run dev`
- 若改动 RAG 数据，记得执行 `npm run ingest`

## Branch and PR

- 新功能请使用独立分支
- PR 标题建议使用动词开头（例如：`add`, `fix`, `refactor`, `docs`）
- PR 描述请包含：
  - 变更动机
  - 主要改动点
  - 本地验证步骤
  - 风险与回滚方式（如有）

## Code Style

- 保持模块边界清晰（`cognitive/`、`lib/`、`server.js`）
- 优先小步提交，避免超大改动
- 不要在无测试的情况下重构核心链路
- 涉及行为变化时补最小测试

## Testing

- 先跑已有测试（若存在）
- 新增测试优先覆盖纯函数与边界条件
- 避免将外部服务可用性作为单测前提

## Security and Privacy

- 不要提交密钥、凭据、个人数据样本
- 不要把本地运行数据目录内容直接提交到仓库
- 涉及摄像头、语音、身份信息改动时，更新 `SECURITY.md`
