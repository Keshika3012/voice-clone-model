#!/usr/bin/env python3
"""
Select best quality audio segments from processed Chris folder
Prioritizes longer, cleaner segments that are more likely to be single-speaker
"""

import os
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from tqdm import tqdm

def analyze_segment(file_path):
    """Analyze audio segment quality"""
    try:
        y, sr = librosa.load(file_path, sr=22050)
        
        duration = len(y) / sr
        
        # Get energy
        energy = np.mean(librosa.feature.rms(y=y))
        
        # Get spectral features
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        
        # Score based on duration (prefer 5-10 seconds) and energy
        duration_score = 1.0 if 5 <= duration <= 10 else 0.5
        energy_score = min(energy / 0.1, 1.0)  # Normalize energy
        
        total_score = duration_score * 0.6 + energy_score * 0.4
        
        return {
            'duration': duration,
            'energy': energy,
            'spectral_centroid': spectral_centroid,
            'score': total_score
        }
    except Exception as e:
        return None

def main():
    print("Selecting best audio segments for Chris voice cloning\n")
    
    processed_dir = Path("data/processed/chris")
    output_dir = Path("data/raw/chris_selected")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all audio files
    audio_files = list(processed_dir.glob("*.wav"))
    print(f"Found {len(audio_files)} audio segments\n")
    
    # Analyze all segments
    print("Analyzing segments...")
    segments_with_scores = []
    
    for audio_file in tqdm(audio_files):
        analysis = analyze_segment(audio_file)
        if analysis:
            segments_with_scores.append((audio_file, analysis))
    
    # Sort by score
    segments_with_scores.sort(key=lambda x: x[1]['score'], reverse=True)
    
    # Select top 20 segments
    top_n = 20
    print(f"\nSelecting top {top_n} segments...")
    
    for i, (audio_file, analysis) in enumerate(segments_with_scores[:top_n]):
        print(f"{i+1}. {audio_file.name}")
        print(f"   Duration: {analysis['duration']:.1f}s, Energy: {analysis['energy']:.4f}, Score: {analysis['score']:.2f}")
        
        # Copy to output directory
        output_path = output_dir / f"chris_segment_{i+1:02d}.wav"
        y, sr = librosa.load(audio_file, sr=22050)
        sf.write(output_path, y, sr)
    
    print(f"\nSelected {top_n} best segments saved to: {output_dir}")
    print("\nNext step: Test voice cloning with these segments")
    print("Run: python test_voice.py")

if __name__ == "__main__":
    main()
