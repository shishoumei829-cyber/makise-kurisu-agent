# Security and Privacy Notes

本文件描述 Amadeus 的安全边界与隐私实践，帮助开发者和用户在本地部署时减少风险。

## Security Model

- 默认运行方式是本机本地服务（`localhost`）
- 主要推理依赖本地 Ollama
- 可选接入本地 SoVITS / Stable Diffusion WebUI
- 项目不包含远端账号系统与多租户权限模型

## Data Storage

运行时状态默认写入：

- `%USERPROFILE%/amadeus_data/`

其中可能包括：

- 对话派生状态（PAD、策略、记忆事件等）
- 用户画像与关系状态
- 可选模块状态快照

请勿将该目录内容直接上传到公开仓库。

## Camera and Voice

项目可选使用摄像头与语音能力：

- 摄像头：由前端授权后采集并提交到本地接口
- TTS：请求转发至本地 SoVITS 服务

建议：

- 仅在可信机器上启用
- 明确告知使用者何时采集与处理
- 关闭不使用的能力与端口

## Hardening Suggestions

- 生产部署时限制接口访问范围（仅本机或内网）
- 给关键 API 增加输入校验与速率限制
- 对日志输出进行脱敏（避免写入敏感信息）
- 发布前执行依赖漏洞扫描与最小权限检查

## Reporting

如发现安全问题，请先私下反馈维护者，不要直接公开利用细节。
