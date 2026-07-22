#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并 LoRA 到基座并写出 HuggingFace 全量模型，供 GGUF / Ollama 转换。

用法：python scripts/finetune/merge_lora.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).resolve().parent / "config.json"


def main() -> int:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    adapter_dir = ROOT / cfg["adapter_dir"]
    merged_dir = ROOT / cfg["merged_dir"]
    meta_path = adapter_dir / "kurisu_train_meta.json"

    if not adapter_dir.exists():
        print(f"[merge] 适配器目录不存在: {adapter_dir}", file=sys.stderr)
        return 1

    base_model = cfg["base_model"]
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            base_model = json.load(f).get("base_model", base_model)

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        print("[merge] pip install -r scripts/finetune/requirements.txt", file=sys.stderr)
        print(e, file=sys.stderr)
        return 1

    print(f"[merge] 加载 base={base_model}")
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir), trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model = model.merge_and_unload()

    merged_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(merged_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_dir))

    print(f"[merge] merged model written to {merged_dir}")
    print("[merge] next: convert the merged model to GGUF and update models/kurisu-v4/Modelfile")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
