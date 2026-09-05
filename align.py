# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["faster-whisper==1.2.1"]
# ///
"""Extract measured local Whisper word spans and flag transcript discrepancies."""
import argparse
import json
import os
from pathlib import Path
import sys
import wave
from io import TextIOWrapper

from alignment import map_alignment, Word

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def main() -> None:
    if isinstance(sys.stdout, TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, nargs="?")
    parser.add_argument("--check", action="store_true", help="Check the offline alignment runtime before synthesis.")
    args = parser.parse_args()
    # This dependency belongs to the PEP 723 uv environment, not the GPU venv.
    from faster_whisper import WhisperModel  # pyright: ignore[reportMissingImports]

    model = WhisperModel("medium.en", device="cpu", compute_type="int8", cpu_threads=8,
                         download_root=str(ROOT / "data/verification-model"), local_files_only=True)
    if args.check:
        print("Offline word-timing runtime ready.", flush=True)
        return
    if args.run_dir is None:
        parser.error("Supply a run folder or --check.")
    report = json.loads((args.run_dir / "report.json").read_text(encoding="utf-8"))
    if len(report["samples"]) != 1:
        parser.error("Studio timing requires a single preset per run.")
    audio = args.run_dir / report["samples"][0]["file"]
    with wave.open(str(audio), "rb") as stream:
        duration = stream.getnframes() / stream.getframerate()
    segments, _ = model.transcribe(str(audio), language="en", beam_size=5,
                                  condition_on_previous_text=False, word_timestamps=True)
    observed: list[Word] = [{"text": word.word.strip(), "start": word.start, "end": word.end}
                for segment in segments for word in (segment.words or [])]
    result = map_alignment(report["text"], observed, duration)
    timing = result["timing"]
    verification = {key: value for key, value in result.items() if key != "timing"}
    verification.update({"checker": "faster-whisper medium.en CPU int8", "audio_duration_seconds": duration,
                   "observed_words": observed,
                   "note": "Word boundaries are estimated from final audio by Whisper. Script spelling is restored; transcription substitutions require listening review."})
    if timing is not None:
        (args.run_dir / "timing.json").write_text(json.dumps(timing, indent=2)+"\n", encoding="utf-8")
    (args.run_dir / "verification.json").write_text(json.dumps(verification, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in verification.items() if key != "observed_words"}), flush=True)


if __name__ == "__main__":
    main()
