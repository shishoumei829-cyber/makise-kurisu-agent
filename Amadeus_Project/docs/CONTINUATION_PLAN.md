# Amadeus 接力计划（下次打开先读这个）

> 更新于 2026-07-20。  
> 目的：下次换对话/重开 Cursor，不必再从头讲产品方向、Git 提交节奏和当前进度。  
> 上位文档：[PRODUCT_CHARTER.md](../PRODUCT_CHARTER.md) · [ROADMAP.md](ROADMAP.md)

---

## 一句话现状

Amadeus 正在从「红莉栖陪伴」升级为「贾维斯式本地管家」。  
**管家地基 v2（operator_v2）代码已写完并测过**，但大量改动还在本地，**按「一天一批」慢慢推 GitHub**，禁止一把梭。

今天刚修过：Electron `preload` 与页面抢用 `amadeus` 名字 → 整页 SyntaxError、卡在「初始化」。已改为 `amadeusDesktop`。

---

## 下次 Agent 开场该做什么

1. 先读本文件 + `PRODUCT_CHARTER.md` + `docs/ROADMAP.md`
2. 跑：`git status -sb` 和 `git log origin/main..HEAD --oneline`
3. **若用户没另说**：按下面「Git 日更节奏」执行**当天一批**（commit 或 push 二选一优先清队列，仍遵守一天一批）
4. 不要重新发明方向；不要一次 commit/push 全部剩余改动

---

## Git 日更节奏（用户硬性要求）

| 规则 | 说明 |
|------|------|
| 一天一批 | 每天最多 **1 次** push 到 GitHub（通常 1 个 commit） |
| 不要一把梭 | 禁止 `git push` 把本地超前的一长串全推上去 |
| 主题清晰 | 每个 commit 只覆盖一个主题 |
| 密钥 | 永不提交 `.env`、`AMADEUS_OPENAI_API_KEY`、真实 token |
| 运行时垃圾 | 不提交 `learning_state.json`、`user_model.json`、本机绝对路径隐私 |

### 推送方式（只推 1 个 commit）

```powershell
# 看队列（旧 → 新）
git rev-list --reverse origin/main..HEAD --oneline

# 只推最早的那一个
$oldest = (git rev-list --reverse origin/main..HEAD | Select-Object -First 1)
git push origin ${oldest}:main
```

### 今日已做（2026-07-20）

- 本地新 commit：`f3f91b9` — preload 改名修复初始化  
- GitHub 已推：`597da80` — digital life modules 接入主循环  
- 当时本地仍超前远程约 10 个 commit

---

## A. 待推送队列（已 commit、未上 GitHub）

按从旧到新，**每天 push 1 个**：

1. `c4ab966` feat(autonomy): deep module 01 — drive dynamics…
2. `e7ee1b6` feat(digital-life): 深度实现模块二至五…
3. `8c64e6e` docs: 数字生命五大模块深度实现更新日志
4. `538882c` feat(architecture): Q1/Q4 架构大更新
5. `9b900c1` feat(architecture): Phase 2 深化
6. `7820454` fix(chat): statePromise 未初始化
7. `c92e952` docs: Cursor Cloud Agent 接力文档
8. `6253480` docs: README 增加接力文档链接
9. `366fd5a` feat: 本机接力深化 — JP-first / 实录 / 冒烟
10. `f3f91b9` fix: preload 与页面抢用 amadeus（**启动修复，优先保留到队列末尾推**）

> 若队列与 `git rev-list` 不一致，以 git 为准，并改本文件。

---

## B. 待分批 commit 的工作区（尚未入库）

建议按天拆成下面主题，**一天最多新 commit 1 个主题**（可与「推送 1 个旧 commit」错开：例如今天只 push，明天只 commit）：

| 顺序 | 主题 | 大致包含 | 不要带上 |
|------|------|----------|----------|
| B1 | 桌面壳启动加固 | `main.js`（dotenv/健康检查/媒体权限/单实例）、`amadeus_work.html` 里摄像头非阻塞启动相关改动 | 整页无关大改可再拆 |
| B2 | 管家内核 v2 | `lib/butler/**`、`tests/butler.test.js`、`server.js` 中 butler/journal 接线 | `.env` |
| B3 | 脑管衔接 | `brain/**`（workspace 订阅任务、consciousness butler）、`legacyChat` getActiveTasks 相关 | — |
| B4 | 产品文档 | `PRODUCT_CHARTER.md`、`docs/ROADMAP.md`、本文件 | 简历 txt、resume_*.py |
| B5 | 意识/脑文档与测试 | `docs/CONSCIOUSNESS.md`、`docs/BRAIN_V1_MIGRATION.md`、`tests/brain.*.test.js` | — |
| B6 | 微调脚本与数据（可选） | `scripts/finetune/**`、`data/finetune/**`、`tests/finetuneDataset.test.js` | 大模型权重 `models/` 若过大勿推 |
| B7 | 日语语料 brain_data | `brain_data/kurisu_ja/**` | — |
| B8 | 其余杂项 | autostart、voice_daemon、repair 脚本等 | `assets/models/*.tflite` 视体积决定 |

**明确不要提交：**

- `Amadeus_Project/.env`
- `learning_state.json` / `user_model.json`（除非确认是模板）
- 根目录 `resume_*.txt`、`optimize_resume.py` 等简历实验文件（与 Amadeus 无关）
- 嵌套旧目录大删除 `Amadeus_Trae/Amadeus_Project/**`：单独一天做清理 commit，勿与功能混推

---

## 产品方向（不要跑偏）

### 永久目标

本地智能管家：理解用户、接住目标、操作本机、证据验证、失败恢复、连续人格 + 日语声音。

### 护城河（模型可换，这些不能丢）

1. 统一事件事实史  
2. 无证据不宣称完成 + 失败恢复  
3. 本机执行履历（userWorld）  
4. 人格与管家同一主体（脑管衔接）

### 版本位置

| 版本 | 状态 |
|------|------|
| 陪伴层（PAD/记忆/TTS/摄像头） | 可用 |
| 管家 v1 任务/证据/能力 | 已有 |
| **管家 v2** 统一 journal / 恢复循环 / 脑管衔接 | **代码完成，待分批入库** |
| v3 浏览器执行器 | 未开工 |
| v4 能力考试 + 长期个性化 | 未开工 |

### 真执行器（已有）

`system.inspect` · `file.search` · `file.create_text` · `file.trash` · `file.restore` · `reminder.create` · `app.launch` · `desktop.open_target`  

空壳：`web.research` 等。

---

## 已知坑

1. **禁止**再把 preload 暴露成 `window.amadeus`（必须 `amadeusDesktop`）  
2. 后端默认端口以 `.env` 的 `AMADEUS_BACKEND_PORT` 为准（常见 **3002**）  
3. 摄像头初始化必须 `void this._initCamera()`，不可阻塞 boot  
4. `kernel.getActiveTasks()` 必须存在（legacyChat / 意识管线依赖）  
5. 提交前检查：暂存区是否被旧的 `git add` 污染（曾误把 finetune/brain_data 打进启动修复 commit，已 soft reset 纠正）

---

## 验证口令

```powershell
cd Amadeus_Project
npm test
curl.exe -s http://localhost:3002/health
curl.exe -s http://localhost:3002/butler/status   # phase 应为 operator_v2
```

桌面端：能过「初始化」、能说话、能通话、摄像头非假 CAM READY。

---

## 给下一任 Agent 的最短指令

```
读 Amadeus_Project/docs/CONTINUATION_PLAN.md。
按「一天一批」处理 Git：今天只 push 队列里 origin/main..HEAD 最早的 1 个 commit，
或只 commit 表 B 里的下一个主题，不要全量提交/推送。
产品方向以 PRODUCT_CHARTER + ROADMAP 为准；下一步功能是管家 v3 浏览器只读能力，但先把 Git 队列清完再开大功能，除非用户另有指示。
```
