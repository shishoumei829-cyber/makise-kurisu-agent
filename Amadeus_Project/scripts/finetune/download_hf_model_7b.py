#!/usr/bin/env python3
"""Download the Qwen2.5 7B instruct weights used by Kurisu v6 QLoRA."""

from __future__ import annotations

import os
from pathlib import Path

HF_HOME = Path("E:/amadeus_finetune/hf-cache")
LOCAL_DIR = Path("E:/amadeus_finetune/models/Qwen2.5-7B-Instruct")
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("HF_HUB_CACHE", str(HF_HOME / "hub"))
os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from huggingface_hub import snapshot_download


def main() -> None:
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[download-7b] {MODEL_ID} -> {LOCAL_DIR}", flush=True)
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(LOCAL_DIR),
        local_dir_use_symlinks=False,
        resume_download=True,
        ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
    )
    shards = sorted(LOCAL_DIR.glob("*.safetensors"))
    if not shards:
        raise RuntimeError("7B download completed without safetensors")
    total_gb = sum(item.stat().st_size for item in shards) / (1024 ** 3)
    print(f"[download-7b] ready shards={len(shards)} size={total_gb:.2f} GiB", flush=True)


if __name__ == "__main__":
    main()
