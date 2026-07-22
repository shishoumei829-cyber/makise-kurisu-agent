#!/usr/bin/env python3
"""Always-on local voice daemon for Amadeus.

Pipeline:
microphone -> Silero VAD -> SpeechBrain speaker verification -> faster-whisper
-> POST /ambient-hearing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import tempfile
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import requests
import sounddevice as sd
import soundfile as sf
import torch
import torchaudio
from faster_whisper import WhisperModel
from speechbrain.inference.speaker import EncoderClassifier


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT.parent / "amadeus_data" / "voice"
PROFILE_PATH = DATA_DIR / "speaker_profiles.json"
SAMPLE_RATE = 16000
FRAME_SAMPLES = 512
FRAME_MS = int(round(FRAME_SAMPLES * 1000 / SAMPLE_RATE))


@dataclass
class Profile:
    speaker: str
    embedding: list[float]
    created_at: float
    seconds: float


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_profiles() -> dict[str, Profile]:
    if not PROFILE_PATH.exists():
        return {}
    raw = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    return {k: Profile(**v) for k, v in raw.items()}


def save_profiles(profiles: dict[str, Profile]) -> None:
    ensure_data_dir()
    payload = {k: vars(v) for k, v in profiles.items()}
    PROFILE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def list_devices() -> None:
    print(sd.query_devices())


def pcm_to_float32(data: bytes) -> np.ndarray:
    audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    return np.clip(audio, -1.0, 1.0)


def record_seconds(seconds: float, device: int | None = None) -> np.ndarray:
    print(f"[voice] recording {seconds:.1f}s at {SAMPLE_RATE}Hz...")
    audio = sd.rec(
        int(seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        device=device,
    )
    sd.wait()
    return audio.reshape(-1)


class SpeakerVerifier:
    def __init__(self, threshold: float) -> None:
        self.threshold = threshold
        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(DATA_DIR / "models" / "spkrec-ecapa-voxceleb"),
            run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        )

    def embed(self, audio: np.ndarray) -> np.ndarray:
        signal = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            emb = self.classifier.encode_batch(signal).squeeze().detach().cpu().numpy()
        norm = np.linalg.norm(emb)
        return emb / max(norm, 1e-9)

    def verify(self, audio: np.ndarray, profile: Profile | None) -> tuple[bool, float]:
        if profile is None:
            return False, 0.0
        current = self.embed(audio)
        ref = np.asarray(profile.embedding, dtype=np.float32)
        ref = ref / max(np.linalg.norm(ref), 1e-9)
        score = float(np.dot(current, ref))
        return score >= self.threshold, score


class Transcriber:
    def __init__(self, model: str, device: str, compute_type: str, language: str) -> None:
        self.language = language
        self.model = WhisperModel(model, device=device, compute_type=compute_type)

    def transcribe(self, audio: np.ndarray) -> str:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = Path(f.name)
        try:
            sf.write(tmp, audio, SAMPLE_RATE)
            segments, _info = self.model.transcribe(
                str(tmp),
                language=self.language or None,
                vad_filter=False,
                beam_size=1,
                temperature=0.0,
            )
            return "".join(seg.text for seg in segments).strip()
        finally:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass


class SileroVad:
    def __init__(self, threshold: float) -> None:
        self.threshold = threshold
        self.model, _utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
            verbose=False,
        )
        self.model.eval()

    def is_speech(self, frame: np.ndarray) -> bool:
        if frame.shape[0] != FRAME_SAMPLES:
            frame = np.pad(frame, (0, max(0, FRAME_SAMPLES - frame.shape[0])))[:FRAME_SAMPLES]
        tensor = torch.from_numpy(frame.astype(np.float32))
        with torch.no_grad():
            prob = float(self.model(tensor, SAMPLE_RATE).item())
        return prob >= self.threshold


def audio_to_wav_bytes(audio: np.ndarray) -> bytes:
    pcm = np.clip(audio, -1.0, 1.0)
    pcm16 = (pcm * 32767).astype(np.int16)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp = Path(f.name)
    try:
        with wave.open(str(tmp), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm16.tobytes())
        return tmp.read_bytes()
    finally:
        tmp.unlink(missing_ok=True)


def post_ambient(
    backend: str,
    text: str,
    speaker: str,
    verified: bool,
    score: float,
    source: str,
) -> None:
    url = backend.rstrip("/") + "/ambient-hearing"
    payload = {
        "text": text,
        "speaker": speaker,
        "verified": verified,
        "source": source,
        "voiceScore": score,
    }
    r = requests.post(url, json=payload, timeout=8)
    r.raise_for_status()
    data = r.json()
    status = "accepted" if data.get("accepted") else f"ignored:{data.get('reason')}"
    print(f"[voice] posted {status}: {text}")


def install_audio_callback(q: queue.Queue[bytes], device: int | None):
    def callback(indata, frames, _time_info, status):
        if status:
            print(f"[audio] {status}")
        q.put(bytes(indata))

    return sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=FRAME_SAMPLES,
        dtype="int16",
        channels=1,
        callback=callback,
        device=device,
    )


def concat_frames(frames: Iterable[np.ndarray]) -> np.ndarray:
    if not frames:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(list(frames)).astype(np.float32)


def listen(args: argparse.Namespace) -> None:
    ensure_data_dir()
    profiles = load_profiles()
    profile = profiles.get(args.speaker)
    if profile is None:
        raise SystemExit(f"missing speaker profile: run enroll --speaker {args.speaker}")

    vad = SileroVad(args.vad_threshold)
    verifier = SpeakerVerifier(args.speaker_threshold)
    transcriber = Transcriber(args.model, args.asr_device, args.compute_type, args.language)

    q: queue.Queue[bytes] = queue.Queue(maxsize=80)
    speech_frames: list[np.ndarray] = []
    silence_ms = 0
    in_speech = False
    min_frames = math.ceil(args.min_seconds * 1000 / FRAME_MS)
    max_frames = math.ceil(args.max_seconds * 1000 / FRAME_MS)

    print("[voice] listening. Ctrl+C to stop.")
    with install_audio_callback(q, args.device):
        while True:
            frame = pcm_to_float32(q.get())
            speech = vad.is_speech(frame)
            if speech:
                if not in_speech:
                    print("[voice] speech start")
                    in_speech = True
                    speech_frames = []
                silence_ms = 0
                speech_frames.append(frame)
            elif in_speech:
                silence_ms += FRAME_MS
                speech_frames.append(frame)

            too_long = len(speech_frames) >= max_frames
            ended = in_speech and silence_ms >= args.end_silence_ms
            if not (ended or too_long):
                continue

            audio = concat_frames(speech_frames)
            in_speech = False
            silence_ms = 0
            speech_frames = []
            if audio.shape[0] < min_frames * FRAME_SAMPLES:
                continue

            verified, score = verifier.verify(audio, profile)
            print(f"[voice] speaker score={score:.3f} verified={verified}")
            if not verified and not args.post_unverified:
                continue

            text = transcriber.transcribe(audio)
            if not text:
                continue
            if verified or args.post_unverified:
                post_ambient(
                    args.backend,
                    text,
                    args.speaker if verified else "unknown",
                    verified,
                    score,
                    "voice_daemon",
                )


def enroll(args: argparse.Namespace) -> None:
    ensure_data_dir()
    verifier = SpeakerVerifier(args.speaker_threshold)
    audio = record_seconds(args.seconds, args.device)
    emb = verifier.embed(audio).astype(float).tolist()
    profiles = load_profiles()
    profiles[args.speaker] = Profile(
        speaker=args.speaker,
        embedding=emb,
        created_at=time.time(),
        seconds=float(args.seconds),
    )
    save_profiles(profiles)
    print(f"[voice] saved profile '{args.speaker}' -> {PROFILE_PATH}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Amadeus local always-on voice daemon")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("devices", help="list audio input/output devices")
    d.set_defaults(func=lambda _args: list_devices())

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--device", type=int, default=None)
    common.add_argument("--speaker", default="owner")
    common.add_argument("--speaker-threshold", type=float, default=0.62)

    e = sub.add_parser("enroll", parents=[common], help="record and save speaker profile")
    e.add_argument("--seconds", type=float, default=25.0)
    e.set_defaults(func=enroll)

    l = sub.add_parser("listen", parents=[common], help="start always-on listening")
    l.add_argument("--backend", default=os.environ.get("AMADEUS_BACKEND", "http://127.0.0.1:3002"))
    l.add_argument("--model", default=os.environ.get("AMADEUS_ASR_MODEL", "small"))
    l.add_argument("--language", default=os.environ.get("AMADEUS_ASR_LANGUAGE", "zh"))
    l.add_argument("--asr-device", choices=["auto", "cpu", "cuda"], default=os.environ.get("AMADEUS_ASR_DEVICE", "auto"))
    l.add_argument("--compute-type", default=os.environ.get("AMADEUS_ASR_COMPUTE", "int8"))
    l.add_argument("--vad-threshold", type=float, default=0.5)
    l.add_argument("--end-silence-ms", type=int, default=850)
    l.add_argument("--min-seconds", type=float, default=0.8)
    l.add_argument("--max-seconds", type=float, default=18.0)
    l.add_argument("--post-unverified", action="store_true")
    l.set_defaults(func=listen)
    return p


def main() -> None:
    args = build_parser().parse_args()
    if getattr(args, "asr_device", "") == "auto":
        args.asr_device = "cuda" if torch.cuda.is_available() else "cpu"
    args.func(args)


if __name__ == "__main__":
    main()
