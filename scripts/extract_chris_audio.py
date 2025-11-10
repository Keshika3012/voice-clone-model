#!/usr/bin/env python3
"""
Chris Audio Extraction Pipeline

This script processes all dual-speaker podcast episodes to extract Chris's voice segments.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.speaker_diarization import ChrisSpeakerExtractor


def main():
    """Main extraction pipeline for Chris's voice."""
    
    # Paths
    input_dir = "data/raw/verified_host_only_episodes"
    output_dir = "data/processed/chris"
    
    print("="*60)
    print("CHRIS VOICE EXTRACTION PIPELINE")
    print("="*60)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' not found!")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize extractor
    print("Initializing speaker extractor...")
    extractor = ChrisSpeakerExtractor()
    
    # Check for HuggingFace token
    if not os.getenv('HF_TOKEN'):
        print("\nWARNING: No HuggingFace token found!")
        print("Please set your HF token: export HF_TOKEN=your_token")
        print("You can get a token from: https://huggingface.co/settings/tokens")
        print()
        
        # Ask if they want to continue anyway
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Process all files
    results = extractor.process_all_files(input_dir, output_dir)
    
    # Print detailed results
    print("\n" + "="*60)
    print("EXTRACTION RESULTS")
    print("="*60)
    
    print(f"Successfully processed: {len(results['processed_files'])} files")
    print(f"Failed to process: {len(results['failed_files'])} files")
    print(f"Total Chris segments extracted: {results['total_segments']}")
    
    if results['failed_files']:
        print("\nFailed files:")
        for failed_file in results['failed_files']:
            print(f"  - {failed_file}")
    
    print(f"\nExtracted segments saved to: {output_dir}")
    
    # Calculate total duration estimate
    avg_segment_duration = 8.0  # Average segment length
    total_duration_minutes = (results['total_segments'] * avg_segment_duration) / 60
    
    print(f"Estimated total Chris audio: {total_duration_minutes:.1f} minutes")
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("1. Review extracted Chris segments in", output_dir)
    print("2. Run model training on Chris's voice data")
    print("3. Compare SpeechT5 vs XTTS v2 models")
    print("="*60)


if __name__ == "__main__":
    main()