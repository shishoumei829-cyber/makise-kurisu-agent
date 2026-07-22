#!/usr/bin/env python3
"""预下载 Qwen 基座到 E 盘（HF 镜像 + 逐文件断点续传）。"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HF = "E:/amadeus_finetune/hf-cache"
LOCAL = Path("E:/amadeus_finetune/models/Qwen2.5-3B-Instruct")
MODEL = "Qwen/Qwen2.5-3B-Instruct"

os.environ.setdefault("HF_HOME", HF)
os.environ.setdefault("HF_HUB_CACHE", f"{HF}/hub")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import hf_hub_download, list_repo_files

SMALL_FILES = [
    ".gitattributes",
    "config.json",
    "generation_config.json",
    "LICENSE",
    "merges.txt",
    "model.safetensors.index.json",
    "README.md",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
]
WEIGHT_FILES = [
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
]


def has_weights(dir_path: Path) -> bool:
    if not dir_path.is_dir():
        return False
    return any(p.suffix == ".safetensors" for p in dir_path.iterdir())


def download_file(filename: str) -> None:
    dest = LOCAL / filename
    if dest.exists() and dest.stat().st_size > 1024:
        print(f"[download] skip existing {filename} ({dest.stat().st_size // (1024*1024)} MB)", flush=True)
        return
    print(f"[download] fetching {filename} ...", flush=True)
    path = hf_hub_download(
        MODEL,
        filename,
        local_dir=str(LOCAL),
        local_dir_use_symlinks=False,
    )
    size_mb = Path(path).stat().st_size // (1024 * 1024)
    print(f"[download] ok {filename} ({size_mb} MB)", flush=True)


def main() -> int:
    LOCAL.mkdir(parents=True, exist_ok=True)
    print(f"[download] model={MODEL}", flush=True)
    print(f"[download] local_dir={LOCAL}", flush=True)
    print(f"[download] endpoint={os.environ.get('HF_ENDPOINT')}", flush=True)

    if has_weights(LOCAL):
        print("[download] weights already present, skip", flush=True)
        return 0

    try:
        remote_files = set(list_repo_files(MODEL))
    except Exception as exc:
        print(f"[download] list_repo_files failed: {exc}", flush=True)
        remote_files = set(SMALL_FILES)

    for name in SMALL_FILES:
        if name not in remote_files:
            continue
        try:
            download_file(name)
        except Exception as exc:
            print(f"[download] ERROR {name}: {exc}", flush=True)
            return 1

    # 大权重由 download_weights_curl.bat 断点续传
    missing = [w for w in WEIGHT_FILES if not (LOCAL / w).exists() or (LOCAL / w).stat().st_size < 1024 * 1024]
    if missing:
        print(f"[download] weight shards pending (use curl): {', '.join(missing)}", flush=True)
        return 0

    if not has_weights(LOCAL):
        print("[download] ERROR: no weight files", flush=True)
        return 1

    print(f"[download] done: {LOCAL}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
