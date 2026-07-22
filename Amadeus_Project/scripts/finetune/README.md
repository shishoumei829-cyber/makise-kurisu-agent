# Kurisu v4 微调

当前微调负责牧濑红莉栖的日语口吻、反应节奏和角色边界，不单独承担复杂推理。
生产对话由 Amadeus 的记忆/行为层组装上下文，再由推理模型生成答案；`kurisu-v4-candidate`
作为本地角色模型和云端不可用时的回退。

## 当前配置

- 基座：`Qwen2.5-3B-Instruct`
- 数据：54 条训练样本、8 条评测样本
- 方法：QLoRA，completion-only loss
- 适配器：`D:/Amadeus_Trae/model_artifacts/kurisu-v4-lora`
- 合并模型：`D:/Amadeus_Trae/model_artifacts/kurisu-v4-merged`
- Ollama 模型：`kurisu-v4-candidate`

## 数据与验证

```powershell
npm run finetune:dataset
npm run finetune:validate
```

## 训练

已具备本地基座权重时：

```powershell
E:\amadeus_finetune\.venv-finetune\Scripts\python.exe scripts/finetune/train_lora.py
```

需要检查并下载权重时：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/finetune/run_train.ps1
```

`start_train.bat` 是同一流程的快捷入口，不会结束其他 Python 进程。

## 合并与发布

```powershell
E:\amadeus_finetune\.venv-finetune\Scripts\python.exe scripts/finetune/merge_lora.py
```

将合并模型转换为 GGUF 后，更新 `models/kurisu-v4/Modelfile` 的 `FROM` 路径，再执行：

```powershell
ollama create kurisu-v4-candidate -f models/kurisu-v4/Modelfile
npm run finetune:evaluate
```

运行时保持：

```dotenv
AMADEUS_CHAT_MODEL=kurisu-v4-candidate
AMADEUS_WORK_USE_REASONER=1
```

不要恢复 v2/v3 的模板或模型标签。旧数据配对方式会强化口吻，却会损害逻辑、计算和记忆真实性。
