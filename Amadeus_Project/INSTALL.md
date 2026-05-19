# Amadeus 安装与使用指南（给使用者）

本指南面向**非开发者**：按步骤安装后，打开 Amadeus 并开始对话。

## 方式 A：Windows 安装包（推荐）

1. 在 `Amadeus_Project/dist/` 找到 **`Amadeus Setup 1.0.0.exe`**（或 Releases 页下载）
2. 双击安装，完成后从开始菜单或桌面打开 **Amadeus**
3. 首次打开若顶部出现黄条，按提示操作（通常是启动 Ollama）

开发者本地打包：

```bash
cd Amadeus_Project
scripts\build-release.bat
```

> 安装包会自带 Electron 壳；**仍需本机安装 Ollama** 才能对话。

## 方式 B：一键脚本（开发者/源码用户）

1. 安装 [Node.js 18+](https://nodejs.org)
2. 安装 [Ollama](https://ollama.com) 并启动
3. 在项目目录双击 `一键启动.bat`

或命令行：

```bash
cd Amadeus_Project
npm install
npm run dev
```

浏览器访问 `http://localhost:3000`

## 方式 C：Electron 桌面壳（源码）

```bash
cd Amadeus_Project
npm install
npm start
```

## 必须准备（对话功能）

| 步骤 | 操作 |
|------|------|
| 1 | 安装并启动 Ollama |
| 2 | 拉取/创建对话模型 `kurisu:latest`（或修改 `.env` 中的 `AMADEUS_CHAT_MODEL`） |
| 3 | 打开 Amadeus，确认顶部**无黄条**或访问 `/health` 显示 ready |

检查命令（可选）：

```bash
ollama list
```

应能看到你的对话模型。

## 可选功能

| 功能 | 需要 |
|------|------|
| 长期记忆 RAG | `npm run ingest` + 嵌入模型 `nomic-embed-text` |
| 语音 TTS | 本地 GPT-SoVITS（见 `docs/GPU_OLLAMA_SOVITS.md`） |
| 设计出图 | Stable Diffusion WebUI |

缺少可选项**不影响文字聊天**。

## 自检

- 浏览器：`http://localhost:3000/health`
- 脚本：双击 `scripts/install-check.bat`（Windows）

## 常见问题

**顶部黄条：Ollama 未就绪**
- 打开 Ollama 应用，或命令行运行 `ollama serve`

**缺少对话模型**
- 按你的模型文档执行 `ollama pull` / `ollama create`

**端口被占用**
- 修改 `.env` 中 `AMADEUS_BACKEND_PORT`，前后端需一致

**安装包打开后后端失败**
- 确认安装目录有写入权限，且 3000 端口未被占用

## 数据保存在哪

用户数据默认在：

- `%USERPROFILE%\amadeus_data\`

卸载程序不会自动删除该目录（保留你的对话记忆与状态）。
