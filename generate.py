"""Generate local Qwen3-TTS narration and optional Studio word timing."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import shutil
import subprocess
import gc
import secrets
from io import TextIOWrapper
from typing import Any
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
MODEL_REVISION = "0c0e3051f131929182e2c023b9537f8b1c68adfe"

PRESETS = {
    "aiden-teacher": {
        "speaker": "Aiden",
        "label": "Aiden — warm, excited teacher",
        "instruction": (
            "Speak in a warm, enthusiastic and conversational American English voice, "
            "like a friendly teacher excited to explain an amazing discovery. "
            "Sound curious when asking questions. Emphasize surprising facts with natural "
            "variation in pitch and rhythm. Clear pronunciation, lively but unhurried. "
            "Use brief natural pauses. Do not shout."
        ),
    },
    "ryan-explainer": {
        "speaker": "Ryan",
        "label": "Ryan — energetic science explainer",
        "instruction": (
            "Speak in an excited, engaging conversational English voice, like a science "
            "teacher sharing a surprising discovery with students. Use energetic rhythm, "
            "expressive intonation and clear emphasis on the most interesting words. "
            "Ask the question with genuine curiosity. Keep the explanation easy to follow. "
            "Sound friendly and natural, not like a shouting advertisement. "
            "Use crisp, precise diction, as if teaching English learners. "
            "Pronounce short connecting words with clear vowels instead of blending them."
        ),
    },
    "ryan-storyteller": {
        "speaker": "Ryan",
        "label": "Ryan — curious storyteller",
        "instruction": (
            "Speak in a curious, animated storytelling voice. Invite the listener into "
            "a fascinating mystery. Ask the question with wonder, pause briefly before "
            "revealing the answer, then explain with warm excitement. Vary the rhythm "
            "and emphasis naturally. Clear conversational English without shouting."
        ),
    },
    "ryan-calm": {
        "speaker": "Ryan", "label": "Ryan — calm teacher",
        "instruction": "Speak in a calm, gentle, reassuring conversational voice. Explain thoughtfully, with an unhurried pace and soft natural emphasis. Sound warm and interested. Use clear precise diction, natural pauses and a relaxed tone. Do not whisper or sound sleepy.",
    },
    "ryan-happy": {
        "speaker": "Ryan", "label": "Ryan — happy and upbeat",
        "instruction": "Speak with a happy, upbeat, smiling voice. Sound delighted to share a wonderful discovery. Use bright lively intonation, warmth, and playful enthusiasm while pronouncing every word clearly. Keep it conversational and easy to follow. Do not shout or laugh over the words.",
    },
    "ryan-whisper": {
        "speaker": "Ryan", "label": "Ryan — quiet whisper",
        "instruction": "Whisper the entire narration in a soft, breathy, intimate voice, as if sharing a fascinating secret with someone right beside you. Maintain a genuine hushed whisper throughout. Speak clearly and gently with subtle curiosity and unhurried pauses. Do not switch to a normal speaking voice.",
    },
}


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, TextIOWrapper):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=["all", *PRESETS], default="ryan-explainer")
    parser.add_argument("--text-file", type=Path, default=ROOT / "examples/antelope-hook.txt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output")
    parser.add_argument("--seed", type=int, help="Reproduce a take. Studio timing runs choose a new seed by default.")
    parser.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--decoder", choices=["gpu", "cpu"], default="cpu")
    parser.add_argument("--timing", action="store_true", help="Extract local word timestamps for Studio after generation.")
    args = parser.parse_args()
    if args.seed is None:
        args.seed = secrets.randbelow(2**31) if args.timing else 42
    text = args.text_file.read_text(encoding="utf-8-sig").strip()
    if not text or len(text) > 4000:
        parser.error("Supply 1–4,000 characters of narration in a UTF-8 text file.")
    if args.timing and args.preset == "all":
        parser.error("Choose one preset when requesting Studio timing.")
    uv = shutil.which("uv")
    if args.timing:
        if not uv:
            raise RuntimeError("uv is required for the existing local transcription runtime.")
        print("Checking offline word-timing runtime...", flush=True)
        subprocess.run([uv, "run", "--offline", "--script", str(ROOT / "align.py"), "--check"], check=True, timeout=90)

    import numpy as np
    import soundfile as sf
    import torch
    from huggingface_hub import snapshot_download
    from qwen_tts import Qwen3TTSModel

    torch.set_num_threads(8)
    if args.device == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("AMD GPU is unavailable. Check the ROCm setup or explicitly use --device cpu.")
    device = "cuda:0" if args.device == "gpu" else "cpu"
    dtype = torch.bfloat16 if args.device == "gpu" else torch.float32
    gpu_name = torch.cuda.get_device_name(0) if args.device == "gpu" else None
    run_dir = args.output_dir.resolve() / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "script.txt").write_text(text + "\n", encoding="utf-8")

    print(f"Loading {MODEL_ID} on {gpu_name or 'CPU'}...", flush=True)
    started = perf_counter()
    model_path = snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION, local_files_only=args.timing)
    model = Qwen3TTSModel.from_pretrained(
        model_path, device_map=device, dtype=dtype,
        attn_implementation="sdpa",
    )
    decoder_device = "cpu" if args.device == "cpu" else args.decoder
    tokenizer = None
    if args.device == "gpu" and decoder_device == "cpu":
        # Keep speech generation on the GPU; avoid slow ROCm audio convolutions.
        tokenizer = model.model.speech_tokenizer
        if tokenizer is None:
            raise RuntimeError("The speech waveform decoder is missing from the local model.")
        tokenizer.model.to(device="cpu", dtype=torch.float32)
        tokenizer.device = torch.device("cpu")
    report: dict[str, Any] = {
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "torch": torch.__version__,
        "gpu": gpu_name,
        "device": args.device,
        "decoder_device": decoder_device,
        "dtype": str(dtype),
        "seed": args.seed,
        "load_seconds": round(perf_counter() - started, 2),
        "text": text,
        "samples": [],
    }
    names = list(PRESETS) if args.preset == "all" else [args.preset]
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    for name in names:
        preset = PRESETS[name]
        print(f"Generating {preset['label']}...", flush=True)
        started = perf_counter()
        pieces = []
        paragraph_report = []
        sample_rate = None
        for index, paragraph in enumerate(paragraphs):
            print(f"  Paragraph {index + 1}/{len(paragraphs)}", flush=True)
            torch.manual_seed(args.seed + index)
            with torch.inference_mode():
                waves, rate = model.generate_custom_voice(
                    text=paragraph, language="English", speaker=preset["speaker"],
                    instruct=preset["instruction"], max_new_tokens=2048,
                )
            if sample_rate is not None and sample_rate != rate:
                raise RuntimeError("The model returned inconsistent audio sample rates.")
            sample_rate = rate
            piece = np.asarray(waves[0], dtype=np.float32).reshape(-1)
            if not piece.size:
                raise RuntimeError(f"{name}: paragraph {index + 1} produced no audio.")
            if pieces:
                pieces.append(np.zeros(round(sample_rate * 0.2), dtype=np.float32))
            pieces.append(piece)
            paragraph_report.append({
                "text": paragraph, "seed": args.seed + index,
                "duration_seconds": round(piece.size / sample_rate, 3),
            })
        if args.device == "gpu":
            torch.cuda.synchronize()
        elapsed = perf_counter() - started
        wave = np.concatenate(pieces)
        if not wave.size or not np.isfinite(wave).all():
            raise RuntimeError(f"{name}: generated audio is empty or non-finite.")
        peak = float(np.max(np.abs(wave)))
        rms = float(np.sqrt(np.mean(wave * wave)))
        if peak < 0.001 or rms < 0.0001:
            raise RuntimeError(f"{name}: generated audio is effectively silent.")
        if peak >= 1.0:
            raise RuntimeError(f"{name}: generated audio would clip; regenerate this take.")
        audio_path = run_dir / f"{name}.wav"
        if sample_rate is None:
            raise RuntimeError("The model returned no audio sample rate.")
        sf.write(audio_path, wave, sample_rate, subtype="PCM_16")
        duration = wave.size / sample_rate
        sample = {
            "preset": name, **preset, "file": audio_path.name,
            "sample_rate": sample_rate, "duration_seconds": round(duration, 3),
            "generation_seconds": round(elapsed, 2),
            "generation_seconds_per_audio_second": round(elapsed / duration, 3),
            "peak_dbfs": round(float(20 * np.log10(peak)), 2),
            "rms_dbfs": round(float(20 * np.log10(rms)), 2),
            "paragraphs": paragraph_report,
            "paragraph_gap_seconds": 0.2 if len(paragraphs) > 1 else 0,
        }
        report["samples"].append(sample)
        (run_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Saved {audio_path} ({duration:.1f}s audio in {elapsed:.1f}s)", flush=True)
    print(f"Completed. Report: {run_dir / 'report.json'}", flush=True)
    if args.timing:
        # Release synthesis memory before the separate CPU transcription process.
        del model
        del tokenizer
        gc.collect()
        if args.device == "gpu":
            torch.cuda.empty_cache()
        if not uv:
            raise RuntimeError("uv is required for the existing local transcription runtime.")
        print("Measuring word timing and checking the spoken script...", flush=True)
        subprocess.run([uv, "run", "--offline", "--script", str(ROOT / "align.py"), str(run_dir)], check=True)


if __name__ == "__main__":
    from runtime_lock import generation_lock
    with generation_lock(ROOT / "data" / "generation.lock"):
        main()
