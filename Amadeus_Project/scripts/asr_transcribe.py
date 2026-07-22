#!/usr/bin/env python3
"""One-shot local ASR for Amadeus call mode.

Usage:
  python asr_transcribe.py --wav path/to/audio.wav [--language zh]
Prints one JSON line: {"ok": true, "text": "..."} or {"ok": false, "error": "..."}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav', required=True)
    parser.add_argument('--language', default='zh')
    parser.add_argument('--model', default='base')
    args = parser.parse_args()

    wav = Path(args.wav)
    if not wav.is_file():
        print(json.dumps({'ok': False, 'error': f'missing wav: {wav}'}, ensure_ascii=False))
        return 2

    try:
        from faster_whisper import WhisperModel
    except Exception as exc:  # pragma: no cover
        print(json.dumps({'ok': False, 'error': f'faster_whisper import failed: {exc}'}, ensure_ascii=False))
        return 3

    try:
        model = WhisperModel(args.model, device='cpu', compute_type='int8')
        segments, _info = model.transcribe(
            str(wav),
            language=args.language or None,
            vad_filter=True,
            beam_size=1,
        )
        text = ''.join(seg.text for seg in segments).strip()
        print(json.dumps({'ok': True, 'text': text}, ensure_ascii=False))
        return 0
    except Exception as exc:  # pragma: no cover
        print(json.dumps({'ok': False, 'error': str(exc)}, ensure_ascii=False))
        return 4


if __name__ == '__main__':
    sys.exit(main())
