# AMADEUS System

### Digital Lifeform Project ｜ 有情感连续性的 AI 伴侣

> 角色设定为《命运石之门》同人创作，仅用于学习、研究与开源展示。详见 [NOTICE.md](NOTICE.md)。

![AMADEUS_Interface](https://github.com/user-attachments/assets/83d3ffa4-1d09-49ed-bf8a-acb5d780059d)

---

## 快速开始

主项目在 [`Amadeus_Project/`](Amadeus_Project/)。

```bash
cd Amadeus_Project
npm install
npm run dev
```

浏览器打开 `http://localhost:3000`，或阅读 [INSTALL.md](Amadeus_Project/INSTALL.md)。

Windows 安装包：见 [Releases](https://github.com/shishoumei829-cyber/makise-kurisu-agent/releases)（本地构建见 `Amadeus_Project/scripts/build-release.bat`）。

---

## 概述

**AMADEUS System** 是一个面向 AI Companion 场景的数字人格系统。

项目并不只追求“更聪明的对话”，而是尝试回答：

> **如果 AI 具备持续性记忆与人格一致性，它是否仍只是工具？**

系统围绕：

* AI 是否可以形成稳定人格结构
* 情绪是否可以被建模为可调控变量
* 长期记忆是否可以构成「关系」而非「数据」
* 数字存在是否可以具备陪伴属性

---

## 系统架构（摘要）

| 模块 | 说明 |
|------|------|
| 情绪状态模型 PAD | P/A/D 三维向量，影响语气与策略 |
| 长期记忆 | LangChain + HNSW 向量检索 |
| 行为约束 | 多路径候选 + 打分 + OOC 防护 |
| 运行时 | Node.js + Express + Ollama + Electron |

详细 API 与模块说明见 [Amadeus_Project/PROJECT_STATUS.md](Amadeus_Project/PROJECT_STATUS.md)。

---

## 文档

| 文档 | 说明 |
|------|------|
| [Amadeus_Project/README.md](Amadeus_Project/README.md) | 主文档（中文） |
| [Amadeus_Project/README_EN.md](Amadeus_Project/README_EN.md) | English overview |
| [Amadeus_Project/INSTALL.md](Amadeus_Project/INSTALL.md) | 使用者安装指南 |
| [Amadeus_Project/MVP_STATUS.md](Amadeus_Project/MVP_STATUS.md) | 产品状态（非技术版） |
| [CONTRIBUTING.md](Amadeus_Project/CONTRIBUTING.md) | 贡献指南 |
| [SECURITY.md](Amadeus_Project/SECURITY.md) | 安全与隐私 |

---

## 设计哲学

1. **人格不是生成的，而是约束出来的** — 在有限状态空间中维持稳定行为轨迹  
2. **记忆不是数据库，而是重构机制** — 被当前状态重新加权后的记忆投影  
3. **情绪不是表达，而是控制变量** — 调节输出结构与决策路径  

---

## 许可证

[ISC License](LICENSE)
