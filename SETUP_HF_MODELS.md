# Hugging Face Model Setup Instructions

## Required Model Licenses

You need to accept the user agreements for these models:

1. **pyannote/segmentation-3.0**
   - Visit: https://huggingface.co/pyannote/segmentation-3.0
   - Click "Agree and access repository"

2. **pyannote/speaker-diarization-3.1**
   - Visit: https://huggingface.co/pyannote/speaker-diarization-3.1
   - Click "Agree and access repository"

## Steps

1. **Log in to Hugging Face**: https://huggingface.co/login
   - Use your existing account

2. **Accept each model's terms**:
   - Visit each URL above
   - Read and accept the terms
   - You should see "Access granted" or similar

3. **Set your Hugging Face token**: 
   - Get your token from: https://huggingface.co/settings/tokens
   - Set it as an environment variable

4. **Run extraction again** after accepting licenses:
   ```bash
   cd /Users/keshikaa/new-voice-clone-project
   source venv/bin/activate
   export HF_TOKEN='your_token_here'
   python extract_chris_proper.py
   ```

## Note

Once you accept the terms, it may take a few minutes for access to be granted. If it still doesn't work immediately, wait 5-10 minutes and try again.
