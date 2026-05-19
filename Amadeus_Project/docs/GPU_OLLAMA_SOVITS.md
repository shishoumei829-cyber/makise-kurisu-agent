# 8GB 显存：Ollama GPU 与 SoVITS 排障

## 现象

- 回复或「译日」极慢（数十秒），GPU 利用率很低。
- 日志：`TTS 自检: SoVITS 不可用 502`、`译日失败`。

通常 **不是换推理引擎能解决的**：Ollama 若回退到 CPU，换 llama.cpp 仍会在 CPU 上慢。先把权重装进 GPU。

## 1. 诊断（必做）

在项目目录执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\diag_ollama_gpu.ps1
```

关注：

- `ollama ps` 最右列：`100% GPU` 为佳；`100% CPU` 或 `xx% CPU` 表示在 CPU 上跑。
- `nvidia-smi`：`Memory-Usage` 是否已满。

## 2. 释放显存（操作）

- Chrome：`chrome://settings` → 关闭「使用硬件加速模式」。
- 关闭 Stable Diffusion、ComfyUI、OBS 等占 GPU 的程序。
- 任务管理器 → 性能 → GPU：**专用 GPU 内存**尽量留出给 Ollama（7B Q4 约需 5–7GB）。

## 3. 强制 Ollama 使用 GPU（环境变量）

以下变量必须作用于 **运行 `ollama serve` 的进程**（Windows 可设「用户环境变量」后重启 Ollama / 注销重登）：

| 变量 | 建议值 | 说明 |
|------|--------|------|
| `OLLAMA_NUM_GPU` | `999` | 尽量把层放到 GPU |
| `OLLAMA_FLASH_ATTENTION` | `1` | 省显存（Ollama 版本需支持） |

Amadeus 后端默认已收紧（见下），无需再改代码即可受益：

| 变量 | 默认（代码内） | 说明 |
|------|----------------|------|
| `AMADEUS_OLLAMA_NUM_CTX` | `2048` | 比 4096 省显存 |
| `AMADEUS_MAX_PROMPT_CHARS` | `6000` | 限制 prompt 长度 |
| `AMADEUS_OLLAMA_KEEP_ALIVE` | `2m` | 避免长期占满与 SoVITS 冲突 |

完整示例见 [`env.example`](../env.example)（与 `server.js` 同目录）。

## 3.1 译日专用小模型（前端 `amadeus_work.html`）

主对话仍用 `mainModel`（如 `kurisu:latest`）；**译日与日→中修复**直连 Ollama，默认使用 **`translateModel: 'qwen2.5:3b'`**，减轻与红莉栖大模型抢 GPU、也避免微调模型「只会中文」。

```bash
ollama pull qwen2.5:3b
```

名称以 `ollama list` 为准（个别环境为 `qwen2.5:3b-instruct` 等，则在配置里改成一致）。若该模型未安装，请求会 **404**，前端会自动 **回退到 `mainModel`**（控制台会提示）。

可调：`translateTimeoutMs`（默认 90000）、`translateModel`（设为空字符串则全程用 `mainModel`）。

## 4. SoVITS（9880）

后端默认代理：`AMADEUS_SOVITS_URL`（默认 `http://localhost:9880`）。

- **502 / 无法连接**：在 GPT-SoVITS 目录单独启动 `api_v2`（或官方启动方式），看终端报错。
- **CUDA OOM**：LLM 与 SoVITS 同卡争显存。可选：SoVITS 用 CPU 推理、或把主模型降到 Q3（见下）、或先关其它占 GPU 程序。

## 5. 仍部分 CPU：降量化（`kurisu:q3`）

```powershell
ollama show kurisu:latest --modelfile > kurisu.Modelfile
# 编辑 Modelfile：将 FROM 行中的量化改为 q3_K_M（或更小）
ollama create kurisu:q3 -f kurisu.Modelfile
```

然后把 [`amadeus_work.html`](../amadeus_work.html) 里 `mainModel` 改为 `kurisu:q3` 试跑。

## 6. 验收清单

- [ ] `ollama ps` 显示 `100% GPU`（或接近）。
- [ ] 主对话首包延迟在数秒内（非数十秒）。
- [ ] TTS 自检无 502；能听见合成语音。
- [ ] 译日日志不再长时间卡在「译日中」。

## 何时才考虑换引擎

仅在已确认 **GPU 已满载** 且延迟仍不可接受时，再评估 vLLM / TabbyAPI 等；8GB 上多数问题来自 **CPU 回退** 或 **SoVITS 未启动**，与引擎品牌无关。
