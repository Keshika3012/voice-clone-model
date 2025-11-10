#!/usr/bin/env python3
"""
Speaker Diarization Script for Chris Voice Extraction

This script uses pyannote.audio to perform speaker diarization on dual-speaker
podcast episodes and extract segments where Chris is speaking.
"""

import os
import sys
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import argparse
from tqdm import tqdm
import json

try:
    from pyannote.audio import Pipeline
    from pyannote.core import Annotation, Segment
except ImportError:
    print("pyannote.audio not installed. Please install with: pip install pyannote.audio")
    sys.exit(1)


class ChrisSpeakerExtractor:
    """
    Extract Chris's voice segments from dual-speaker audio files using speaker diarization.
    """
    
    def __init__(self, model_name: str = "pyannote/speaker-diarization-3.1"):
        """
        Initialize the speaker extractor.
        
        Args:
            model_name: Hugging Face model name for speaker diarization
        """
        self.model_name = model_name
        self.pipeline = None
        self.chris_speaker_id = None
        self.speaker_profiles = {}
        
    def load_pipeline(self):
        """Load the pyannote speaker diarization pipeline."""
        print(f"Loading speaker diarization pipeline: {self.model_name}")
        
        # Get token from environment
        hf_token = os.environ.get('HF_TOKEN')
        if not hf_token:
            print("Error: HF_TOKEN environment variable not set")
            print("Please set your Hugging Face token: export HF_TOKEN=your_token")
            sys.exit(1)
        
        try:
            # Use the token explicitly
            self.pipeline = Pipeline.from_pretrained(
                self.model_name,
                use_auth_token=hf_token
            )
        except Exception as e:
            print(f"Error loading pipeline: {e}")
            print("Please make sure you've accepted the model licenses at:")
            print("  - https://huggingface.co/pyannote/speaker-diarization-3.1")
            print("  - https://huggingface.co/pyannote/segmentation-3.0")
            sys.exit(1)
            
    def diarize_audio(self, audio_path: str) -> Annotation:
        """
        Perform speaker diarization on audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Annotation object with speaker segments
        """
        if self.pipeline is None:
            self.load_pipeline()
            
        print(f"Diarizing: {audio_path}")
        diarization = self.pipeline(audio_path)
        return diarization
        
    def identify_chris_speaker(self, diarization: Annotation, 
                             audio_path: str, 
                             sample_duration: float = 30.0) -> str:
        """
        Identify which speaker is Chris based on voice characteristics.
        
        This is a simplified version - in practice, you might need to:
        1. Use a pre-trained speaker identification model
        2. Compare against known Chris voice samples
        3. Use manual annotation for initial setup
        
        Args:
            diarization: Speaker diarization results
            audio_path: Path to audio file
            sample_duration: Duration of sample to analyze
            
        Returns:
            Speaker ID for Chris
        """
        # Get speaker statistics
        speaker_stats = {}
        
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            if speaker not in speaker_stats:
                speaker_stats[speaker] = {'duration': 0, 'segments': []}
            speaker_stats[speaker]['duration'] += segment.duration
            speaker_stats[speaker]['segments'].append(segment)
            
        print("Speaker statistics:")
        for speaker, stats in speaker_stats.items():
            print(f"  {speaker}: {stats['duration']:.1f}s across {len(stats['segments'])} segments")
            
        # For now, assume the speaker with more speaking time is Chris
        # This is a heuristic that may need adjustment
        chris_speaker = max(speaker_stats.keys(), 
                          key=lambda x: speaker_stats[x]['duration'])
        
        print(f"Identified Chris as speaker: {chris_speaker}")
        return chris_speaker
        
    def extract_chris_segments(self, audio_path: str, 
                             output_dir: str,
                             min_segment_length: float = 2.0,
                             max_segment_length: float = 15.0) -> List[str]:
        """
        Extract Chris's voice segments from audio file.
        
        Args:
            audio_path: Path to input audio file
            output_dir: Directory to save Chris segments
            min_segment_length: Minimum segment duration in seconds
            max_segment_length: Maximum segment duration in seconds
            
        Returns:
            List of paths to extracted Chris segments
        """
        # Perform diarization
        diarization = self.diarize_audio(audio_path)
        
        # Identify Chris speaker
        if self.chris_speaker_id is None:
            self.chris_speaker_id = self.identify_chris_speaker(diarization, audio_path)
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=22050)
        
        # Extract Chris segments
        chris_segments = []
        segment_paths = []
        
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            if speaker == self.chris_speaker_id:
                # Check segment duration
                duration = segment.end - segment.start
                if min_segment_length <= duration <= max_segment_length:
                    chris_segments.append(segment)
                    
        print(f"Found {len(chris_segments)} Chris segments")
        
        # Save segments
        os.makedirs(output_dir, exist_ok=True)
        audio_name = Path(audio_path).stem
        
        for i, segment in enumerate(chris_segments):
            start_sample = int(segment.start * sr)
            end_sample = int(segment.end * sr)
            
            # Extract audio segment
            segment_audio = y[start_sample:end_sample]
            
            # Save segment
            segment_filename = f"{audio_name}_chris_segment_{i:03d}.wav"
            segment_path = os.path.join(output_dir, segment_filename)
            
            sf.write(segment_path, segment_audio, sr)
            segment_paths.append(segment_path)
            
        print(f"Saved {len(segment_paths)} Chris segments to {output_dir}")
        return segment_paths
        
    def process_all_files(self, input_dir: str, output_dir: str) -> Dict:
        """
        Process all audio files in directory to extract Chris segments.
        
        Args:
            input_dir: Directory containing dual-speaker audio files
            output_dir: Directory to save Chris segments
            
        Returns:
            Dictionary with processing results
        """
        input_path = Path(input_dir)
        audio_files = list(input_path.glob("*.mp3")) + list(input_path.glob("*.wav"))
        
        results = {
            'processed_files': [],
            'total_segments': 0,
            'failed_files': []
        }
        
        print(f"Processing {len(audio_files)} audio files...")
        
        for audio_file in tqdm(audio_files, desc="Extracting Chris segments"):
            try:
                segment_paths = self.extract_chris_segments(
                    str(audio_file), 
                    output_dir
                )
                
                results['processed_files'].append({
                    'file': str(audio_file),
                    'segments': len(segment_paths),
                    'paths': segment_paths
                })
                results['total_segments'] += len(segment_paths)
                
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")
                results['failed_files'].append(str(audio_file))
                
        return results


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(
        description="Extract Chris's voice segments from dual-speaker audio files"
    )
    parser.add_argument(
        "input_dir",
        help="Directory containing dual-speaker audio files"
    )
    parser.add_argument(
        "output_dir", 
        help="Directory to save Chris segments"
    )
    parser.add_argument(
        "--model",
        default="pyannote/speaker-diarization-3.1",
        help="Hugging Face model for speaker diarization"
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=2.0,
        help="Minimum segment duration in seconds"
    )
    parser.add_argument(
        "--max-duration", 
        type=float,
        default=15.0,
        help="Maximum segment duration in seconds"
    )
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = ChrisSpeakerExtractor(args.model)
    
    # Process files
    results = extractor.process_all_files(args.input_dir, args.output_dir)
    
    # Save results
    results_file = os.path.join(args.output_dir, "extraction_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*50)
    print("EXTRACTION SUMMARY")
    print("="*50)
    print(f"Processed files: {len(results['processed_files'])}")
    print(f"Failed files: {len(results['failed_files'])}")
    print(f"Total Chris segments: {results['total_segments']}")
    print(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()