#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kurisu LoRA / QLoRA 微调（HuggingFace + PEFT + TRL）

用法（在 Amadeus_Project 目录）：
  pip install -r scripts/finetune/requirements.txt
  node scripts/finetune/build_sft_dataset.js
  python scripts/finetune/train_lora.py

8GB 显存默认：Qwen2.5-3B + 4bit QLoRA
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
_DEFAULT_HF = Path("E:/amadeus_finetune/hf-cache")
if not os.environ.get("HF_HOME") and _DEFAULT_HF.parent.exists():
    os.environ.setdefault("HF_HOME", str(_DEFAULT_HF))
    os.environ.setdefault("HF_HUB_CACHE", str(_DEFAULT_HF / "hub"))


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> int:
    cfg = load_config()
    dataset_path = ROOT / cfg["dataset"]
    if not dataset_path.exists():
        print(f"[train] 数据集不存在: {dataset_path}", file=sys.stderr)
        print("[train] 请先运行: node scripts/finetune/build_sft_dataset.js", file=sys.stderr)
        return 1

    try:
        # datasets 须在 torch 之前导入，否则部分 Windows+CUDA 环境会原生崩溃
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            TrainingArguments,
        )
        from trl import SFTTrainer, SFTConfig
        import torch
    except ImportError as e:
        print("[train] 缺少依赖，请执行: pip install -r scripts/finetune/requirements.txt", file=sys.stderr)
        print(e, file=sys.stderr)
        return 1

    if not torch.cuda.is_available():
        print("[train] 警告：未检测到 CUDA，QLoRA 在 CPU 上极慢。建议 NVIDIA GPU。", file=sys.stderr)

    rows = load_jsonl(dataset_path)
    max_samples = int(cfg.get("max_samples") or 0)
    if max_samples > 0:
        rows = rows[:max_samples]

    base_model = cfg["base_model"]
    adapter_dir = ROOT / cfg["adapter_dir"]
    adapter_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_items = []
    for row in rows:
        messages = row["messages"]
        train_items.append({
            "prompt": messages[:-1],
            "completion": [messages[-1]],
        })
    ds = Dataset.from_list(train_items)

    eval_ds = None
    eval_path = ROOT / cfg.get("eval_dataset", "")
    if eval_path.is_file():
        eval_items = []
        for row in load_jsonl(eval_path):
            messages = row["messages"]
            eval_items.append({
                "prompt": messages[:-1],
                "completion": [messages[-1]],
            })
        if eval_items:
            eval_ds = Dataset.from_list(eval_items)

    bnb_config = None
    if cfg.get("use_4bit", True):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=int(cfg["lora_r"]),
        lora_alpha=int(cfg["lora_alpha"]),
        lora_dropout=float(cfg["lora_dropout"]),
        target_modules=list(cfg["lora_target_modules"]),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    training_args = SFTConfig(
        output_dir=str(adapter_dir),
        num_train_epochs=float(cfg["epochs"]),
        per_device_train_batch_size=int(cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
        learning_rate=float(cfg["learning_rate"]),
        warmup_ratio=float(cfg["warmup_ratio"]),
        weight_decay=float(cfg["weight_decay"]),
        logging_steps=int(cfg["logging_steps"]),
        save_strategy="epoch",
        save_total_limit=2,
        eval_strategy="epoch" if eval_ds is not None else "no",
        load_best_model_at_end=eval_ds is not None,
        metric_for_best_model="eval_loss" if eval_ds is not None else None,
        greater_is_better=False if eval_ds is not None else None,
        fp16=False,
        bf16=False,
        max_length=int(cfg["max_seq_length"]),
        completion_only_loss=True,
        packing=False,
        report_to=[],
        seed=int(cfg.get("seed", 42)),
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    print(f"[train] 开始微调 base={base_model} samples={len(rows)} -> {adapter_dir}")
    trainer.train()
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    meta = {
        "base_model": base_model,
        "ollama_base": cfg.get("ollama_base"),
        "output_model_tag": cfg.get("output_model_tag"),
        "samples": len(rows),
        "adapter_dir": str(adapter_dir),
    }
    with open(adapter_dir / "kurisu_train_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("[train] 完成。下一步: python scripts/finetune/merge_lora.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
