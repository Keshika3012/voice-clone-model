#!/usr/bin/env python3
"""
Simple test script for voice cloning
"""
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Patch the bangla module before importing TTS
import sys
if 'bangla' not in sys.modules:
    # Create a dummy bangla module
    import types
    bangla_module = types.ModuleType('bangla')
    sys.modules['bangla'] = bangla_module

print("Initializing voice cloner...")

try:
    from voice_cloner import VoiceCloner, GenerationConfig
    
    print("Imports successful")
    
    # Initialize cloner
    cloner = VoiceCloner(data_dir="data", output_dir="outputs", device="auto")
    
    print(f"\nLoaded {len(cloner.voice_profiles)} voice profiles:")
    for speaker_id, profile in cloner.voice_profiles.items():
        print(f"  - {profile.name} (ID: {speaker_id}, {len(profile.audio_files)} files)")
    
    # Test generation with chris
    if 'chris' in cloner.voice_profiles:
        print("\nTesting voice generation with Chris...")
        text = "Hello, this is a test of the voice cloning system. I'm excited to see how this sounds!"
        output_path = "outputs/samples/chris_test.wav"
        
        config = GenerationConfig(language="en", speed=1.0)
        result = cloner.generate_speech(text, "chris", output_path, config)
        
        print(f"Generated audio: {result}")
        print("\nVoice cloning test successful!")
    else:
        print("\n⚠️  Chris voice profile not found")
        print("Available profiles:", list(cloner.voice_profiles.keys()))
        
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
