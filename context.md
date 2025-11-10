# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Common Development Commands

### Setup & Installation
```bash
# Setup virtual environment
python3 -m venv venv
source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Voice Data Management
```bash
# Extract podcast episodes (automatic data collection)
python scripts/extract_podcast_episodes.py

# Process audio files into training segments
python scripts/process_audio.py

# Add custom voice profile
python scripts/cli.py add-voice "Speaker Name" audio1.wav audio2.wav
```

### Voice Generation & Testing
```bash
# List available voices
python scripts/cli.py list-voices

# Generate single speech sample
python scripts/cli.py generate "Your text here" speaker_id

# Run interactive demo
python scripts/demo.py

# Batch generation from text file
python scripts/cli.py batch texts.txt speaker_id --output-dir batch_output/

# Compare speakers
python scripts/cli.py compare "Same text" --speakers chris,daniel

# Analyze voice characteristics
python scripts/cli.py analyze speaker_id
```

### Model Development & Testing
```bash
# Test individual components
python src/voice_cloner.py
python src/model_comparison.py

# Run filtering tests
python scripts/test_filtering.py

# Evaluate models
python scripts/evaluate_models.py
```

## Architecture Overview

### Core Components

**Voice Cloning Pipeline:**
1. **Audio Preprocessing** (`scripts/process_audio.py`) - Segments raw audio, normalizes levels, filters by quality
2. **Voice Profile Management** (`src/voice_cloner.py`) - Manages speaker embeddings and reference audio
3. **Speech Generation** (`src/voice_cloner.py`) - XTTS v2 model integration for synthesis
4. **Model Comparison** (`src/model_comparison.py`) - Evaluation framework comparing XTTS v2 vs SpeechT5

**Data Flow:**
```
Raw Podcast Audio → Audio Processing → Speaker Segments → Voice Profiles → Speech Generation
```

### Key Classes & Architecture Patterns

**VoiceCloner** (Primary Interface):
- Manages XTTS v2 model lifecycle and device selection
- Handles voice profile loading from `data/raw/speaker_name/` directories
- Provides batch generation, speaker comparison, and voice analysis
- Uses dataclasses for `VoiceProfile` and `GenerationConfig`

**AudioProcessor** (Data Pipeline):
- Implements silence-based segmentation with configurable thresholds
- Normalizes audio to -20dB target, filters segments by duration (3-15s)
- Creates speaker datasets through basic alternating assignment (placeholder for real speaker diarization)

**CLI Architecture**:
- Click-based command structure with context passing
- Supports voice management, generation, batch processing, and analysis
- Error handling with proper exit codes

**Demo System** (`scripts/demo.py`):
- Interactive menu-driven demonstration
- Multiple demo types: basic, speed control, speaker comparison, batch generation, voice analysis
- Educational showcase of all system capabilities

### Model Integration

**XTTS v2 (Primary):**
- Multilingual TTS with voice cloning capabilities
- Uses reference audio for voice embedding
- Configurable generation parameters (temperature, speed, language)

**SpeechT5 (Comparison):**
- Alternative TTS model for evaluation
- Requires separate speaker embedding creation
- Used in model comparison framework

### Data Organization

```
data/
├── raw/                    # Raw audio files
│   ├── speaker_name/       # Per-speaker directories
│   └── practical_ai_episodes/  # Podcast source data
├── processed/              # Segmented audio
│   ├── episodes/           # Episode-based segments
│   ├── chris/              # Speaker-specific datasets
│   └── daniel/
└── models/                 # Model storage

outputs/
├── samples/                # Individual generations
├── demos/                  # Demo outputs
└── comparisons/            # Speaker comparison results
```

### Configuration & Parameters

**Audio Processing:**
- Target sample rate: 22050 Hz
- Segment length: 3-15 seconds
- Silence threshold: -40 dB
- Audio normalization: -20 dB target

**Voice Generation:**
- Default language: English
- Speed range: 0.8x - 1.5x
- Temperature: 0.75 (default)
- Device selection: Auto-detects CUDA availability

## Development Patterns

### Adding New Speakers
1. Create directory: `data/raw/new_speaker/`
2. Add audio files (.wav/.mp3)
3. System auto-loads profiles on initialization
4. Or use CLI: `python scripts/cli.py add-voice "Speaker Name" audio_files...`

### Model Evaluation Workflow
1. Use `ModelComparison` class for systematic evaluation
2. Metrics include PESQ, STOI, MFCC similarity, F0 similarity
3. Generates comprehensive JSON reports with quality scores
4. Supports reference audio comparison for objective metrics

### Error Handling Strategy
- Graceful degradation when models fail to load
- Comprehensive logging throughout pipeline
- CLI tools provide detailed error messages with exit codes
- Demo system continues operation even if individual components fail

### Extending Generation Capabilities
- Implement new generation configs by extending `GenerationConfig`
- Add CLI commands by extending the Click command structure
- New demo types can be added to `VoiceCloningDemo` class
- Model comparison supports pluggable metric calculation

## Project Context

This is a research/educational voice cloning system focused on:
- High-quality speech synthesis using XTTS v2
- Podcast data extraction and processing (specifically Practical AI podcast)  
- Comparative model evaluation
- Interactive demonstration of voice cloning capabilities

The system emphasizes ease of use through comprehensive CLI tools and interactive demos, while maintaining flexibility for research and experimentation through its modular architecture.