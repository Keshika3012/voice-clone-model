#!/usr/bin/env python3
"""
Properly extract Chris's voice using speaker diarization
Requires Hugging Face token set as environment variable: HF_TOKEN
"""

import os
import sys
from pathlib import Path

# Check for HF token
if 'HF_TOKEN' not in os.environ:
    print("Hugging Face token not found!")
    print("\nTo use speaker diarization, you need to:")
    print("1. Create account at https://huggingface.co/")
    print("2. Accept license at https://huggingface.co/pyannote/speaker-diarization-3.1")
    print("3. Get token from https://huggingface.co/settings/tokens")
    print("4. Set environment variable:")
    print("   export HF_TOKEN='your_token_here'")
    print("\nThen run: python extract_chris_proper.py")
    sys.exit(1)

sys.path.append(str(Path(__file__).parent / "scripts"))

from speaker_diarization import ChrisSpeakerExtractor

def main():
    print("Extracting Chris's voice using speaker diarization\n")
    
    # Initialize extractor
    extractor = ChrisSpeakerExtractor()
    
    # Process verified episodes
    input_dir = "data/raw/verified_host_only_episodes"
    output_dir = "data/raw/chris_diarized"
    
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}\n")
    
    # Process all files
    results = extractor.process_all_files(input_dir, output_dir)
    
    print("\nProcessing complete!")
    print(f"Processed: {len(results['processed_files'])} files")
    print(f"Total Chris segments: {results['total_segments']}")
    if results['failed_files']:
        print(f"Failed: {len(results['failed_files'])} files")
        
    print(f"\nChris audio segments saved to: {output_dir}")

if __name__ == "__main__":
    main()
