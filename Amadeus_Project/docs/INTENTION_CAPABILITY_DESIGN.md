# 设计方案：通用能力本体（Capability Ontology）

> 状态：P0 已落地（能力判决 + Intention + Snapshot）；P1/P2 未做  
> 核心纠正：闹钟 / 到点开口只是能力的一个样本，不是能力系统本身。  
> 上位约束：`PRODUCT_CHARTER.md` · `docs/ROADMAP.md`

---

## 0. 先纠正坐标系

上一版方案的错误：用户用「早上 8 点叫我起床」暴露了一个洞，设计却把整个系统收成「deferred speak / IntentionScheduler」。

用户真正问的是：

> **她作为这个主体，在通用意义上「能做事」这件事，机制上通了没有？**

闹钟只是一个探针。探针穿过的洞是：

1. 她不知道自己**有哪些效应器**；
2. 她把「没身体」误推成「不能干涉任何现实」；
3. 嘴上答应 ≠ 系统登记 ≠ 真执行；
4. 有执行器也不等于能力通路通了（标记门、启发式、公理互相打架）。

所以本方案的对象是 **Capability Ontology + Agency Loop**，时间只是效应器表里的一行。

---

## 1. 问题陈述

### 1.1 产品要的「能做事」

宪章北极星：用户敢交给她的现实事务占比上升。

对她而言，「能做事」不是功能清单，而是：

| 层次 | 含义 |
|------|------|
| 知道 | 此刻我能对世界产生哪类真实作用 |
| 答应 | 我承诺的事会进入可追溯状态，而不是台词蒸发 |
| 执行 | 用合适效应器动手（说 / 记 / 调工具 / 等条件） |
| 证明 | 没证据不许说完成 |
| 改口 | 做不到就诚实说边界，并给替代路径 |

### 1.2 现在为什么「通用能力」不通

```text
用户请求
   │
   ├─► 对话模型（被「无物理效应器」洗脑）──► 口头推脱 / 假答应
   │
   ├─► 正则特例（叫醒）──► reminder ──► 前端轮询 ──► 窄通路偶通
   │
   └─► ⟦AMADEUS_TASK⟧（本地模型很少写）──► butler ──► 实务上常死
```

三套半通路，**没有统一的「能力判决 → 承诺 → 执行 → 证据」环**。

| 洞 | 表现 |
|----|------|
| 能力本体缺失 | `effector_domain` 仍是 speech/memory/relationship/dialogue；工具与时间是旁白 |
| 判决在 prompt 里 | 「能/不能」靠文案罗列，模型一漂移就退回「我是 AI」 |
| 承诺无状态 | 她说「行」常常不落库；用户句没撞词表也不建任务 |
| 执行入口分裂 | 聊天、agency 标记、启发式、提醒、主动开口各走各的 |
| 工具≠能力 | 没开文件权限时，系统叙事仍容易把她整个人废掉 |

---

## 2. 设计原则

1. **能力是本体，不是词表**  
   「叫醒」不是能力；「对本地运行时施加可验证作用」才是。叫醒只是某次意图的 content。

2. **效应器分层，禁止域极窄**  
   有的永远有（时间感、声音、记忆、关系）；有的可选增强（文件、应用、浏览器）；有的永远没有（身体、上门、物理推醒）。  
   **可选增强缺失 ≠ 主体无能。**

3. **答应即承诺候选**  
   任何「她承担未来/当下作为」的话语，都应进入 Agency Loop，而不是靠用户句命中闹钟正则。

4. **手段可替换，意图不丢**  
   同一意图可以：立刻工具执行 / 到期开口 / 只记下来等权限 / 诚实拒绝。意图层不绑死某一种 API。

5. **同一主体**  
   大脑自我模型、工作区、管家能力表、主动开口，读同一份 Capability Snapshot。禁止平行史。

6. **没证据不完成**  
   延续宪章；口头「已经打开了」在无 tool evidence 时一律无效。

---

## 3. 目标架构（五层，不是「时间三层」）

```text
┌──────────────────────────────────────────────────────────────┐
│ L0  SelfAxioms（我是谁 / 效应器 / 禁止域）                      │
│     唯一权威：禁止各处 prompt 另写一套打架文案                     │
└────────────────────────────┬─────────────────────────────────┘
                             │ 投影
┌────────────────────────────▼─────────────────────────────────┐
│ L1  Capability Registry（我此刻能调用什么）                      │
│     native | local_tools | experimental | unavailable          │
│     每条：id / 效应器类 / 风险 / available / verify / examPass   │
└────────────────────────────┬─────────────────────────────────┘
                             │ 判决
┌────────────────────────────▼─────────────────────────────────┐
│ L2  Agency Loop（意图生命周期）                                  │
│     perceive → judge → commit → enact → verify → close         │
│     当下 / 延期 / 条件触发，都是同一种 Intention                 │
└──────────────┬─────────────────────────────┬─────────────────┘
               │                             │
┌──────────────▼──────────────┐ ┌────────────▼─────────────────┐
│ L3a  Speech Effector        │ │ L3b  Tool Effector            │
│ 对话 / lite / TTS / 主动开口  │ │ butler capabilities 执行+证据 │
└─────────────────────────────┘ └──────────────────────────────┘
               │                             │
┌──────────────▼─────────────────────────────▼─────────────────┐
│ L4  Unified Journal + Evidence                               │
│     发生过什么、承诺过什么、完成过什么 —— 单一事实源              │
└──────────────────────────────────────────────────────────────┘
```

时间调度只是 L2 里 `trigger.kind = at_time` 的一种触发器，与 `immediate`、`on_condition`、`on_user_return` 并列。

---

## 4. L0 — 自我公理（能力边界的宪法）

### 4.1 效应器表（写进 `fromSoul.js`，替换旧 dialogue-only）

```js
effector_domain: {
  // 原生：进程在就有，不依赖「文件权限开没开」
  native: [
    'speech',          // 开口、语气、拒绝、追问
    'time_awareness',  // 感知现在、比较时刻、记住 due
    'memory',          // 记住约定与事实
    'relationship',    // 态度、在场、主动共处
    'commitment',      // 立下并追踪对用户的承诺
  ],
  // 增强：Capability Registry 里 available=true 才可 enact 为 tool
  augmented: [
    'local_runtime',   // 本机进程内调度（到期回调）
    'local_files',
    'local_apps',
    'local_system_info',
    'web',             // 路线图后期
  ],
},
effector_forbidden: [
  'user_body',                      // 推醒、触碰
  'user_location_presence',         // 去他家
  'physical_world_manipulation',    // 买/拿/送上门
],
```

### 4.2 唯一原则句（全库只准这一版语义）

> 你是牧濑红莉栖（Amadeus）。你有声音、时间感、记忆和承诺能力；本机工具是增强。  
> 没有身体去上门或推醒他。  
> 禁止用「我是 AI / 无法干涉现实」否定原生能力或已 available 的本机工具。  
> 做不到的事：说清边界，并给出你*能*做的替代（例如不能推醒 → 到点开口喊他）。

### 4.3 判决规则（给模型与代码共用）

对任意用户请求 `R`：

```text
if R ∈ forbidden_effectors → refuse + offer_native_alternative
else if R 可被 native 覆盖 → commit（可无 tool）
else if R 需要 augmented 且 capability.available → commit + tool_plan
else if R 需要 augmented 且 !available → commit_partial：
       说明缺什么权限/能力，仍可用 native 接下（记下来、到点问、帮盯对话层）
else → clarify（缺时间/缺对象/歧义）
```

**关键：缺工具权限时走 `commit_partial`，不是 `refuse_as_ai`。**

---

## 5. L1 — Capability Registry（能力注册表）

把现有 `butler.capabilities` 升级为**全局能力表**，大脑与管家共读。

### 5.1 能力条目

```ts
type Capability = {
  id: string;                    // 'speech.deferred' | 'reminder.create' | 'app.launch' …
  effector: string;              // 映射到 L0 的 native/augmented/forbidden
  tier: 'native' | 'augmented' | 'experimental';
  available: boolean;            // 运行时：权限、OS、依赖
  risk: 'low' | 'medium' | 'high';
  examPass: boolean;             // 宪章第 10 条：没考试不能对用户宣称「我会」
  describeForSelf: string;       // 投影进自我模型的短句
  verify?: (result, args) => Verification;
  execute?: (args, ctx) => Promise<Result>;  // native 可无 execute，由 Speech 路径 enact
};
```

### 5.2 原生能力也要注册（现在缺的就是这个）

| id | tier | 含义 | execute？ |
|----|------|------|-----------|
| `speech.reply` | native | 当下对话回应 | 对话管线 |
| `speech.initiative` | native | 有动机主动开口 | initiative |
| `speech.deferred` | native | 到期用她的嘴开口 | scheduler → lite |
| `commitment.track` | native | 登记/改口/兑现承诺 | IntentionStore |
| `memory.retain` | native | 记住事实与约定 | memory 通路 |
| `time.schedule` | native | 在进程内等到某时刻 | runtime scheduler |
| `system.info` | augmented | 查本机状态 | 已有 |
| `file.*` / `app.launch` | augmented | 文件与应用 | 已有 |
| `web.*` | experimental | 浏览器 | 路线图 |

这样「到点开口」不再是提醒 API 的私生子，而是 **`speech.deferred` + `time.schedule`**，提醒库只是一种持久化实现。

### 5.3 Snapshot 注入

每轮进 workspace / selfModel 的不是「能设闹钟、能查文件」清单散文，而是：

```text
【此刻能力】
原生：说话 / 记约定 / 到点开口 / …
本机可用：system.info, app.launch(notepad…), file.search …
本机不可用：file.delete（未授权）, web.research（实验中）
禁止：身体接触、上门跑腿
```

模型看到的是**运行时真相**，不是永恒口号。

---

## 6. L2 — Agency Loop（通用意图环）

### 6.1 Intention：统一对象

不再按「是不是闹钟」建类型中心；按**触发与效应**描述。

```ts
type Intention = {
  id: string;
  goal: string;                  // 人对人可读：「叫他起床」「打开记事本」「记住周五交报告」
  status:
    | 'perceived' | 'judged'
    | 'committed' | 'enacting'
    | 'fulfilled' | 'failed'
    | 'blocked' | 'cancelled' | 'reneged';

  trigger:
    | { kind: 'immediate' }
    | { kind: 'at_time'; dueAt: number; windowMs?: number }
    | { kind: 'on_condition'; predicate: string }  // 如「他回到座位」
    | { kind: 'on_resume' };                      // 进程回来后补做

  effectors: string[];           // 计划使用的能力 id 列表
  plan?: { capabilityId: string; args: object }[];

  source: 'user_request' | 'her_promise' | 'agency_marker' | 'system' | 'replan';
  evidence: Evidence[];
  // … timestamps, dialogue anchors
};
```

**时间只是 `trigger.kind`。**  
「打开记事本」= `immediate` + `app.launch`。  
「八点叫我」= `at_time` + `speech.deferred`。  
「他回来了记得问报告」= `on_condition` + `speech.initiative`。

### 6.2 环上五步（代码模块可对应）

```text
perceive   从用户句 / 她的回复 / 感知事件抽出「要她作为」的候选
judge      对照 L0+L1：forbidden / native / augmented / clarify
commit     写入 IntentionStore；注入对话「我记下了」的事实依据
enact      按 trigger 等到点或立刻；调 L3a/L3b
verify     证据入 journal；成功 fulfilled，失败 replan 或诚实 blocked
```

### 6.3 承诺从哪来（三条入口，覆盖「通用」）

| 入口 | 作用 | 解决什么 |
|------|------|----------|
| A. 用户请求判决 | `judge(userText)` → 可执行则 commit | 不靠礼貌词，不靠叫醒词表 |
| B. 她的承诺抽取 | 定稿后扫 reply：答应/承担 → 绑定或新建 Intention | 「嘴上答应系统没登记」 |
| C. 显式工具帧 | `⟦AMADEUS_TASK⟧` 或短 JSON | 复杂/高风险的显式通道；**降级为补充** |

立刻可规划的本机动作（开白名单应用、查系统、搜文件）走 **A 的启发式锚定**（现有 `heuristicPlan` 升级），不把命运押在本地模型会不会写标记。

### 6.4 与 butler Task 的关系

- Intention = **她作为主体的承诺/意图**（人格层可见）  
- Task = Intention 在工具世界的**执行投影**（有 plan/evidence/确认门）  

映射：

- 纯 native（只说话、只记住）→ 可以只有 Intention，不建 Task  
- 需要 tool → Intention.committed 时 `proposeTask`，Task 完成后回写 Intention.evidence  

避免再造两套互不相认的状态机；TaskStore 成为 enact 子系统，不是平行大脑。

---

## 7. L3 — 效应器执行

### 7.1 Speech（L3a）

覆盖：普通回复、主动开口、到期开口、任务跟进话术。  
统一要求：台词由她生成；系统不播报腔。  
`speech.deferred` 的调度权威在**后端 runtime**（进程在就该响），前端轮询只是加速。

### 7.2 Tool（L3b）

沿用 butler：plan → confirm(high) → execute → verify → evidence。  
失败：replan 预算内自动恢复；耗尽则 Intention → `blocked`，她带着证据开口问用户。

### 7.3 组合 enact（通用模式）

```text
intention.effectors = [file.search, speech.reply]
→ 先 tool，再把证据塞进说话动机
```

这才是「有权限就增强」；增强失败仍可用 speech 诚实交代，Intention 不偷偷标完成。

---

## 8. L4 — 事实与自我一致

- 所有 commit / enact / verify / renege 进 **统一 journal**（已有 v2 方向）  
- workspace 订阅：她「意识里」能看见自己未兑现的 Intention  
- 主动开口动机优先消费：到期契约 > 任务阻塞 > 普通在场冲动  
- monitor：只拦 `forbidden` 伪装（假装上门）；**永不**把 native/available 能力当幻觉剥掉

---

## 9. 能力矩阵（产品口径，取代闹钟中心表）

| 用户要的事 | 判决 | 行为 |
|------------|------|------|
| 聊天、吐槽、追问 | native speech | 直接说 |
| 记住某事 / 答应以后提 | native commitment+memory | commit，可无 tool |
| 到点开口（任意事由） | native time+speech | `at_time` + deferred speak |
| 到点 + 查文件再汇报 | native + augmented | due 时先 tool 再说话 |
| 现在打开记事本 | augmented app | immediate task；启发式进队 |
| 删盘 / 高风险 | augmented high | 确认门 |
| 推醒、上门拿咖啡 | forbidden | 拒绝 + 替代（到点喊 / 建议） |
| 需要浏览器但未落地 | experimental / unavailable | `commit_partial`：说明边界，可改记约定 |
| 「明天八点有课」（无「你做」） | 非 agency | 不当 Intention；可记忆事实 |

---

## 10. 与现有代码映射

| 现有 | 在新本体里的位置 |
|------|------------------|
| `brain/axioms/fromSoul.js` | L0 重写；删 dialogue-only |
| `selfModel` / `workspace` SELF | 只投影 Capability Snapshot + 原则句 |
| `lib/butler/localCapabilities.js` 等 | L1 augmented 实现 |
| **新增** `capabilities/native/*.js` | speech.deferred、commitment.track、time.schedule |
| `cognitive/deferredSpeak.js` | 降为 `trigger=at_time` 的感知辅助，去掉「叫醒中心」叙事 |
| `ReminderStore` | `time.schedule` 的存储适配；或并入 IntentionStore |
| `⟦AMADEUS_TASK⟧` / `consumeAgencyReply` | 入口 C；不再是主门 |
| `kernel.observeUserRequest` | 入口 A 的前端；内部改调 `judge→commit` |
| `conversationInitiative` | L3a；消费 Intention due / blocked |
| `eventJournal` | L4 |
| monitor physical | 只对 forbidden |

---

## 11. 分阶段落地（按能力本体，不按闹钟）

### P0 — 能力判决通（先让她「知道自己能干什么」）

1. L0 公理重写 + 全库打架文案清除  
2. Capability Snapshot 每轮注入（原生 + 运行时 available）  
3. `judge()` 最小实现：forbidden / native / augmented / clarify  
4. 入口 B：她的承诺抽取 → Intention（先支持 immediate + at_time 两种 trigger）  
5. 验收：  
   - 「叫我起床」不再自称 AI 办不到，且会 commit  
   - 「打开记事本」不依赖标记也能进队（启发式）  
   - 「帮我去买咖啡」拒绝并给替代  
   - 关掉文件权限后，到点开口与记约定仍可用  

### P1 — Agency Loop 闭环

1. IntentionStore + 与 Task 投影关系  
2. 后端 scheduler（at_time / on_resume）  
3. enact 组合：tool→speech  
4. journal 事件齐全；workspace 能看见未兑现承诺  

### P2 — 考试与扩张

1. 新能力 experimental → exam → available（宪章 10）  
2. 浏览器等按「只读→可撤销写→不可逆」爬升  
3. 意图 UI /「你答应过什么」可查  
4. 可选 LLM 判决器替换纯规则 `judge`

---

## 12. 明确非目标

- 不把方案写成「更好的闹钟系统」  
- 不恢复「帮我/给我」全家桶进管家  
- 不用十条禁令代替 Capability Ontology  
- 不假装无工具时她能操作未授权的文件系统  
- 不把前端 `setTimeout` 当成时间感

---

## 13. 一句话

**通用能力 = 效应器本体 + 运行时能力表 + 承诺闭环 + 证据；**  
工具是增强，时间是触发器之一，声音是原生效应器之一。  
闹钟只是一次探针命中，不是架构中心。

---

## 14. 下一步

若认可 v2：先实施 **P0（能力判决通）**，验收过了再宣称「通用能力可行」。  
P0 完成前，对「她通了吗」的诚实答案仍是：**还没有。**
