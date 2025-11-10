# Chris Voice Cloning Status

## Current Issue

The voice generated is **not Chris's actual voice** because the current audio data in `data/processed/chris/` was created using a **simple alternating assignment** method, not real speaker diarization. This means the audio contains a mix of both Chris and Daniel's voices.

## What We Have

**Voice cloning system working** - XTTS v2 model loads and generates speech successfully
**1,099 processed audio segments** - But they're incorrectly labeled 
**30 verified host-only podcast episodes** - Raw audio ready for processing
**Speaker diarization code** - Ready to use for proper voice extraction

## The Problem

From `scripts/process_audio.py` (lines 173-182):
```python
# Simple alternating assignment (not ideal, but works for demo)
# In practice, you'd use speaker diarization here
if i % 2 == 0:  # Even segments to Chris
    dest_path = chris_dir / Path(file_path).name
    chris_files.append(str(dest_path))
else:  # Odd segments to Daniel
    dest_path = daniel_dir / Path(file_path).name
    daniel_files.append(str(dest_path))
```

This just alternates segments between speakers, which doesn't accurately separate voices.

## Solution: Proper Speaker Diarization

We have proper speaker diarization code in `scripts/speaker_diarization.py` that uses **pyannote.audio** to:
1. Identify which speaker is Chris vs Daniel
2. Extract only Chris's voice segments
3. Create clean training data

### Requirements

To run proper speaker diarization, you need:

1. **Hugging Face Account**: https://huggingface.co/
2. **Accept Model License**: https://huggingface.co/pyannote/speaker-diarization-3.1
3. **Get API Token**: https://huggingface.co/settings/tokens
4. **Set Environment Variable**:
   ```bash
   export HF_TOKEN='your_token_here'
   ```

### Steps to Get Chris's Real Voice

1. **Set up Hugging Face token** (see requirements above)

2. **Run speaker diarization**:
   ```bash
   cd /Users/keshikaa/new-voice-clone-project
   source venv/bin/activate
   python extract_chris_proper.py
   ```

3. **Update voice profile**:
   ```bash
   # Remove old incorrect data
   rm -rf data/raw/chris
   
   # Link to new diarized Chris audio
   ln -s data/raw/chris_diarized data/raw/chris
   ```

4. **Test again**:
   ```bash
   python test_voice.py
   ```

## Alternative: Manual Selection

If you can't get a Hugging Face token, you can manually identify Chris's voice:

1. Listen to a few episodes to identify voice patterns
2. Manually select segments where only Chris is speaking
3. Copy those segments to `data/raw/chris/`

## Quick Test Right Now

To verify the system works with ANY voice, you can:

```bash
# Use verified episodes audio as-is (will be mixed voices but tests the system)
cd /Users/keshikaa/new-voice-clone-project
source venv/bin/activate
python test_voice.py
```

The generated audio at `outputs/samples/chris_test.wav` demonstrates that the voice cloning **system works**, but needs proper input data.

## Summary

- Voice cloning technology: **Working**
- Input data quality: **Mixed speakers (needs diarization)**
- Solution available: **Speaker diarization script ready**
- Action needed: **Get HF token and run proper extraction**

The 1,099 audio files you have are valuable raw material, they just need to be properly separated by speaker using the diarization script.
