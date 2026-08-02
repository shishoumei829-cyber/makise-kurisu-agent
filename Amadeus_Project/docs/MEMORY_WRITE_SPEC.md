# 记忆系统写入规范

> 状态：Phase 1 落地中  
> 原则：**生成可以犯错；记忆按层、按来源、按证据写入。**  
> 中枢：**记忆宫殿**（长期记忆的空间索引）。毒句只是拒绝类之一，不是设计中心。

## 分层

| 层 | 角色 | 权威问题 |
|----|------|----------|
| L0 Working | 本轮工作记忆 | 默认不落盘 |
| L1 Dialogue | 对话实录 | 「你说过/我说过」逐字权威 |
| L2 Palace / Episodic | **记忆宫殿** + 任务/提醒/意图 | 「她记得的事」主索引 |
| L3 Identity | whoami / 关系锚点 | 身份高门槛 |
| L4 Self | 公理 / PAD / 动机 | 她的内部状态 |
| L5 Derived | 统计 / RL / 模式 | 禁止回流当事实 |
| L6 Journal | 审计流 | 对账，不直接喂角色记忆 |

一句话：L1 是速记本，宫殿是她怎么记事的房子，whoami 是户口本，journal 是监控录像。

## 宫殿四房

- `hall` — 日常与关系（默认）
- `lab` — 智识 / 课题
- `cafe` — 生活琐事 / 情绪
- `forbidden` — 高亲密 / 高代价（稀缺召回）

权威落盘：`~/amadeus_data/memory_palace.json`  
前端 localStorage 仅为缓存。

## 写入资格（WriteGate）

### 可进宫殿（archive）

- 实质互动（双方有信息量）
- 用户明确「请记住 / 我喜欢…」等
- 他接话后的主动开口段

### 不进宫殿

- 短闲聊 / 敷衍（嗯、哼、在吗）
- 未接话的主动碎句
- 内部控制 / 压缩器原文 / meta
- 身份崩坏或不完整的 assistant 定稿
- L5 推断倒灌

### 晋升

```
定稿 → L1（对话闸门）→ WriteGate.assessArchive
  reject → 不进宫殿（可记 journal 审计）
  archive → 压缩节点 → 分房 → memory_palace.json
  explicit → 同时可提案 L3 whoami
```

## 召回（必须可检索，不是储物间）

- 仅当需要长期记忆时注入宫殿摘录
- **相关度 top-k**：按关键词/子串打分取命中，禁止只塞「房间最近几条」
- 节点含 `detail` / `keywords` / `user` / `assistant`，保证问得回
- 后端 `navigate` 为权威；`legacyChat` 在需要长期记忆时兜底注入
- 「你说过什么」以 L1 为准；宫殿是长期线索
- prompt 中标明「记忆宫殿摘录」，不得冒充逐字实录
- 验收：`node scripts/memory-stress-retrieve.js`（大量干扰后短上下文已无种子事实，宫殿仍命中）

## API

- `GET /memory/palace` — 快照
- `POST /memory/palace/archive` — 提案归档（经 WriteGate）
- `POST /memory/palace/navigate` — 按用户句召回摘录
- `POST /memory/palace/clear` — 清空宫殿（运维/测试）
