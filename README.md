# New Voice Clone Project

 This project uses modern TTS (Text‑to‑Speech) models to synthesize natural‑sounding speech from text, with support for multiple speakers, batch generation, analysis tools, and a simple CLI.

Highlights:
- Realistic voice cloning (XTTS v2 by default)
- Clean data pipeline for podcasts and custom audio
- One‑liners for generation, batch jobs, and comparisons
- Handy demos to explore features
- Tools for voice quality analysis and model evaluation

## Quick start (5 minutes)

1) Set up Python and deps

```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

2) List voices and generate your first sample

```bash
# See known voices (auto‑detected from data directories)
python scripts/cli.py list-voices

# Make the model talk
python scripts/cli.py generate "Hello from the voice cloner!" chris -o outputs/samples/hello_chris.wav
```

3) Prefer a guided tour?

```bash
python scripts/demo.py
```

That will walk you through single samples, speed tweaks, comparisons, and more.

## What this project is

- A practical voice cloning toolkit built around XTTS v2
- A data pipeline for turning long‑form audio (e.g., podcasts) into clean training segments
- An approachable CLI and demo layer for everyday use
- A research sandbox for evaluating and improving TTS quality

## Current status 

- The cloning system works end‑to‑end. You can generate audio right now.
- If you’re chasing Chris’s exact voice: the “processed” dataset included here was created via a simple even/odd segment split and still contains mixed speakers. For accurate cloning, run real speaker diarization (see “Get clean voice data” below).
- There’s a ready‑to‑run diarization path and a quick “just test it” path.

See: `CHRIS_VOICE_STATUS.md` for the detailed status and next steps.

## Project structure

- src/
  - voice_cloner.py — main TTS interface (XTTS v2), batch generation, analysis
  - speecht5_cloner.py — alt model (for comparison)
  - model_comparison.py — evaluation framework
  - metrics/audio_scorer.py — objective audio metrics
- scripts/
  - cli.py — command‑line interface (generate/list/analyze/batch/compare)
  - demo.py — interactive demos
  - process_audio.py — segmenting, cleaning, normalizing audio
  - extract_podcast_episodes.py — download/curate podcast source audio
  - speaker_diarization.py — proper diarization with pyannote
  - evaluate_models.py, test_filtering.py, speech_brain.py — extra tools
- data/
  - raw/ — your source audio per speaker (you create this)
  - processed/ — segmented audio produced by the pipeline
- outputs/
  - samples/, demos/, comparisons/ — generated results
- requirements.txt, README.md, test_voice.py, select_best_segments.py, extract_chris_proper.py

## Common tasks

- Generate speech for a voice

```bash
python scripts/cli.py generate "A short message." chris -o outputs/samples/chris_msg.wav
```

- Compare two speakers with the same text

```bash
python scripts/cli.py compare "Same line for both." --speakers chris,daniel
```

- Batch generate from a file of lines

```bash
python scripts/cli.py batch texts.txt chris --output-dir outputs/samples/batch/
```

- Analyze a voice’s characteristics

```bash
python scripts/cli.py analyze chris
```

## Use your own audio

1) Create a folder and drop in clean .wav/.mp3 files (speech only, minimal music/background)

```bash
mkdir -p data/raw/speaker_name
# Copy your audio into data/raw/speaker_name/
```

2) Or add via CLI

```bash
python scripts/cli.py add-voice "Speaker Name" path/to/audio1.wav path/to/audio2.wav
```

3) Process audio into segments

```bash
python scripts/process_audio.py
```

Now you can generate with `speaker_name`.

## Get clean voice data (with diarization)

If you want a truly accurate voice clone from mixed‑source audio (e.g., two hosts on a show), run proper speaker diarization using `pyannote.audio`.

- Accept access to required models on Hugging Face:
  - `pyannote/segmentation-3.0`
  - `pyannote/speaker-diarization-3.1`

- Set a token as an environment variable

```bash
export HF_TOKEN={{HF_TOKEN}}
```

- Run the extractor

```bash
source venv/bin/activate
python extract_chris_proper.py
```

This will build a clean dataset of just the target speaker.

Note: If you can’t use diarization right now, you can still test the pipeline with the mixed segments to confirm everything works technically (see `CHRIS_VOICE_STATUS.md` and `test_voice.py`).

## Python API (optional)

```python
from src.voice_cloner import VoiceCloner, GenerationConfig

cloner = VoiceCloner()
config = GenerationConfig(language="en", speed=1.1, temperature=0.7)

wav_path = cloner.generate_speech(
    "This is a custom message.",
    speaker_id="chris",
    output_path="outputs/samples/custom.wav",
    config=config
)
print("Saved to", wav_path)
```

## Audio processing knobs

Tune these in `scripts/process_audio.py`:
- `target_sr`: 22050 Hz (default)
- `min_segment_length`: 3.0s
- `max_segment_length`: 15.0s
- `silence_thresh`: -40 dB
- Normalization target: -20 dB

## Models

- Default: XTTS v2 (multilingual, voice cloning via reference audio)
- Optional: SpeechT5 (comparison/evaluation)

GPU is optional; CPU works fine but is slower.

## Troubleshooting

- Models won’t download or load
  - Double‑check internet and Hugging Face access for gated repos
  - Clear local caches, then retry
  ```bash
  rm -rf ~/.cache/tts/
  python scripts/cli.py list-voices
  ```

- Audio quality isn’t great
  - Use cleaner source audio (no music/crosstalk)
  - Keep segments between 3–15 seconds
  - Ensure sample rate/format compatibility

- Memory errors
  - Reduce batch size or segment lengths
  - Prefer CPU if your GPU is constrained

## Ethics and consent

Only clone voices you’re allowed to clone. Always obtain consent, and label synthetic audio clearly if sharing it.

## Contributing

- Open issues with clear repro steps
- PRs welcome (prefer small, focused changes)
- If adding features, consider a CLI command and a demo entry

## References

- Coqui TTS (XTTS v2)
- Librosa
- pyannote.audio (diarization)
