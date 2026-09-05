# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["faster-whisper==1.2.1"]
# ///
"""Independently transcribe each audition to check for missing or repeated words."""

import argparse
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower().replace("’", "'"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, nargs="?")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--model", choices=["base.en", "small.en", "medium.en"], default="small.en")
    args = parser.parse_args()
    if not args.prepare and args.run_dir is None:
        parser.error("Provide the output run folder, or use --prepare to download the checker.")

    from faster_whisper import WhisperModel

    model = WhisperModel(
        args.model, device="cpu", compute_type="int8", cpu_threads=8,
        download_root=str(ROOT / "data" / "verification-model"),
    )
    if args.prepare:
        print("Local transcription checker ready.")
        return
    report = json.loads((args.run_dir / "report.json").read_text(encoding="utf-8"))
    expected = words(report["text"])
    results = []
    for sample in report["samples"]:
        audio_path = args.run_dir / sample["file"]
        segments, info = model.transcribe(
            str(audio_path), language="en", beam_size=5, condition_on_previous_text=False,
        )
        transcript = " ".join(segment.text.strip() for segment in segments)
        result = {
            "file": sample["file"], "transcript": transcript,
            "normalized_words_match": words(transcript) == expected,
            "audio_duration_seconds": info.duration,
        }
        results.append(result)
        print(json.dumps(result), flush=True)
    payload = {
        "checker": f"faster-whisper {args.model} CPU int8",
        "note": "An independent transcription check; does not rate acting quality or guarantee perfect pronunciation.",
        "all_words_match": bool(results) and all(item["normalized_words_match"] for item in results),
        "samples": results,
    }
    (args.run_dir / "verification.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if not payload["all_words_match"]:
        raise SystemExit("Transcription differences found. Review the audio and verification.json.")


if __name__ == "__main__":
    main()
