# Local Voice Generator

A local text-to-speech tool built with **Qwen3-TTS**, with expressive voice presets and optional word-level timing. It uses pretrained synthetic speakers rather than recordings of a user's voice.

## How it was built

The tool combines three components:

- **Qwen3-TTS CustomVoice** generates speech from text, a speaker name, and delivery instructions.
- **PyTorch** runs speech-token generation; the waveform decoder can run separately on the CPU.
- **faster-whisper** independently transcribes the audio so the tool can compare spoken words with the input and extract timestamps.

The Qwen model snapshot is pinned in `generate.py`. Python packages are recorded in `requirements.txt`. Voice presets are instruction strings, not separately trained models.

## Available presets

| Preset | Delivery |
| --- | --- |
| `ryan-explainer` | Energetic, curious explanation; default |
| `ryan-calm` | Warm and relaxed |
| `ryan-happy` | Bright and upbeat |
| `ryan-whisper` | Soft and intimate |
| `ryan-storyteller` | Expressive storytelling |
| `aiden-teacher` | Alternative warm male speaker |

Delivery can vary between takes. Listen to the output before using it.

## Processing pipeline

1. Read a UTF-8 script and select a preset.
2. Load the pinned pretrained model.
3. Generate each blank-line-separated paragraph and join the audio with short pauses.
4. Save a WAV and a report containing the script, settings, seed, duration, and audio measurements.
5. Optionally transcribe the final audio and align recognized word boundaries to the original script.

A process lock prevents two generator instances from using the runtime simultaneously. No paid speech API is called. Initial installation and model downloads require internet access; prepared timing runs use local caches offline.

## Installation

The included dependency snapshot targets **Windows, Python 3.12, and AMD ROCm PyTorch**. Other platforms or GPU vendors require an appropriate PyTorch installation and separate validation. Consult the [AMD installation documentation](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2.1/docs/install/installrad/windows/install-pytorch.html) for supported drivers and hardware.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), download or clone this repository, and run these commands from its root in PowerShell:

```powershell
uv venv --python 3.12 --seed .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m pip check
```

Prepare the model caches and the separate alignment environment:

```powershell
# Download Qwen and generate a short example.
.venv/Scripts/python.exe generate.py --text-file examples/antelope-hook.txt

# Download the word checker.
uv run --script verify.py --prepare --model medium.en

# Prepare and check the timing environment.
uv run --script align.py --check
uv run --offline --script align.py --check
```

Model downloads and the runtime require several GB of storage. A fresh clone alone is not ready for offline generation.

## Usage

```powershell
# Default voice
.venv/Scripts/python.exe generate.py --text-file examples/antelope-full.txt

# Select a tone
.venv/Scripts/python.exe generate.py --preset ryan-happy --text-file examples/antelope-full.txt

# Compare presets on a short script
.venv/Scripts/python.exe generate.py --preset all --text-file examples/antelope-hook.txt

# Generate a single take with offline word timing
.venv/Scripts/python.exe generate.py --preset ryan-explainer --text-file examples/antelope-full.txt --timing
```

Scripts support up to 4,000 characters. Keep paragraphs short. Each run creates a separate timestamped directory under `output/`. Use `--output-dir PATH` for another destination.

Timing runs select a fresh random seed. `--seed NUMBER` fixes the seed; ordinary audition runs default to 42. Output may still differ across hardware or dependency versions.

The default uses GPU token generation and CPU waveform decoding. `--device cpu` selects CPU generation; `--decoder gpu` changes waveform decoding. Run `generate.py --help` for available options.

## Output files

| File | Contents |
| --- | --- |
| `script.txt` | Input narration |
| `PRESET.wav` | Generated audio |
| `report.json` | Model, settings, seed, and measurements |
| `verification.json` | Transcription and differences, when checking is run |
| `timing.json` | Word timestamps when alignment is usable |

With `--timing`, check `alignment_usable`, `requires_review`, and `all_words_match`. A successful process exit does not by itself mean the spoken words matched.

Substitutions require listening review because either the generator or the recognizer may be wrong. Missing/extra words or unusable boundaries produce no valid timing output. Regenerate rather than inventing timestamps. Word boundaries are model estimates, not guarantees of exact pronunciation.

The standalone checker can inspect an existing output folder:

```powershell
uv run --script verify.py output/REPLACE_WITH_RUN_FOLDER --model medium.en
```

It exits nonzero on transcription differences. Both checking workflows write `verification.json`; preserve an existing report if needed before running another checker.

## Integration

Other applications can invoke `generate.py` as a subprocess with `--preset`, `--text-file`, `--output-dir`, and `--timing`. Consume the saved WAV, report, verification flags, and timing file after completion. The caller owns its queue, listening approval, and any project selection; this tool does not update application databases.

## Privacy and repository contents

Scripts and generated reports can contain the text supplied to the tool. Keep sensitive input and output private. Virtual environments, model caches, generated audio, logs, databases, and common credential files are excluded from Git. Those exclusions are safeguards, not a substitute for reviewing files before committing.

This repository backs up source code, presets, tests, and example text. Generated recordings and external application data require separate backups. No API credentials are required for this tool's local generation workflow.

## Tests

```powershell
.venv/Scripts/python.exe -m unittest -v test_alignment.py test_runtime.py
.venv/Scripts/python.exe -m py_compile generate.py alignment.py align.py runtime_lock.py verify.py
```

Tests cover alignment, transcript differences, invalid timestamps, process locking, and missing-runtime preflight. They do not rate acting quality or replace a GPU generation check.

## Upstream projects

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
- [Qwen CustomVoice model and license](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)

Model weights are downloaded from upstream and are not redistributed here. Refer to the upstream projects for their licenses.
