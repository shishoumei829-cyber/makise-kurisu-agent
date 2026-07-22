# AMADEUS 立绘生产与接入规范

## 目标

立绘不是聊天界面的装饰轮播，而是角色状态的可见结果。所有资产必须先经过角色一致性、动作语义和透明边缘审核，再允许进入运行时 registry。

## 不再采用的做法

- 不用单条 prompt 直接批量生成 50-100 张。
- 不用“有背景图生成后随手抠图”作为正式资产。
- 不按时间、轮次或每句话切换立绘。
- 不让模型自由决定夸张动作、卖萌姿势或与角色性格冲突的表情。

## 推荐生产管线

1. 角色设计圣经
   - 固定脸型、眼睛、发型、发色、头身比例、服装结构、常用站姿、禁用动作。
   - 同一套服装必须统一光源、线条、画幅、脚底基线和裁切区域。

2. 核心母版
   - 先制作 1 张 neutral 全身/半身母版。
   - 再制作 8-12 张核心状态，不直接扩到 100 张。
   - 核心状态建议：neutral、thinking、skeptical、annoyed、shy_denial、soft、tired、worried、smug。

3. 生成约束
   - 身份一致性：角色 LoRA 优先；可叠加 IP-Adapter、PuLID 或 InstantID 保脸。
   - 姿势控制：ControlNet OpenPose 或等价姿势条件。
   - 服装控制：同一 outfit 单独成组，禁止在一个批次里混服装。
   - 局部修复：FaceDetailer/手部修复/统一超分。
   - 透明输出：优先生成透明 PNG；否则只允许统一纯色背景 + 同一 RMBG/matting 流程。

4. 人工审核
   - 身份：脸型、发型、发色、眼睛、身体比例必须像同一个人。
   - 服装：领带、外套、袖口、裙摆/裤装、配饰不能漂。
   - OOC：动作必须符合红莉栖式克制、理性、嘴硬、别扭；禁止过度软妹、偶像、媚态、夸张撒娇。
   - 技术：透明边缘无脏边；尺寸、锚点、脚底基线一致；面部局部无崩坏。

## 资产 metadata

正式资产入库时应记录：

```json
{
  "id": "casual_shy_denial_01",
  "outfit": "casual_red_tie",
  "emotion": "shy",
  "pose": "arms_crossed_look_away",
  "intensity": 0.65,
  "path": "assets/Live2d/kurisu/sprites/casual_shy_denial_01.png",
  "approved": true,
  "allowed_triggers": ["intimacy_tease", "praise_with_tension"],
  "forbidden_triggers": ["timer", "every_turn", "normal_reply"]
}
```

## 运行时规则

- `SpritePolicy` 只允许切到 registry 中 `approved: true` 且有 `path` 的资产。
- 没有合格资产时，表情状态可以记录，但前端保持当前安全立绘。
- 默认最短保持时间为 45 秒，避免每轮对话抖动。
- 允许触发的事件包括：亲密调侃、技术争论、边界冒犯、疲惫/深夜关心、温和陪伴、高唤醒专注。
- 小幅 PAD 波动只影响滤镜和 TTS hint，不直接换整张图。

## 扩展步骤

1. 把合格 PNG 放入 `assets/Live2d/kurisu/sprites/`。
2. 在 `digital_life/embodiment/sprite_policy.js` 的 `SPRITE_REGISTRY` 中补充 `path` 并设为 `approved: true`。
3. 为新增状态补测试：触发条件、冷却保持、未审核资产阻断。
4. 在真实聊天里检查切换是否只发生在语义事件上。
