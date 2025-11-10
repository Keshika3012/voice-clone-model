#!/usr/bin/env python3
"""
Comprehensive Model Evaluation Script

This script orchestrates the entire Chris voice cloning evaluation pipeline:
1. Extract Chris's voice segments from dual-speaker audio
2. Train/setup both SpeechT5 and XTTS v2 models
3. Run comprehensive comparison between models
4. Generate detailed evaluation reports
"""

import os
import sys
import argparse
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.speaker_diarization import ChrisSpeakerExtractor
from src.model_comparison import ModelComparison
from src.voice_cloner import VoiceCloner
from src.speecht5_cloner import SpeechT5VoiceCloner


class VoiceCloningEvaluator:
    """
    Comprehensive evaluation system for Chris voice cloning project.
    """
    
    def __init__(self, 
                 raw_audio_dir: str = "data/raw/verified_host_only_episodes",
                 processed_dir: str = "data/processed/chris",
                 output_dir: str = "outputs/evaluation"):
        """
        Initialize the evaluator.
        
        Args:
            raw_audio_dir: Directory with dual-speaker audio files
            processed_dir: Directory to save Chris's extracted audio
            output_dir: Directory for evaluation results
        """
        self.raw_audio_dir = Path(raw_audio_dir)
        self.processed_dir = Path(processed_dir)
        self.output_dir = Path(output_dir)
        
        # Create directories
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.speaker_extractor = None
        self.model_comparison = None
        
    def step_1_extract_chris_voice(self):
        """Step 1: Extract Chris's voice segments from dual-speaker audio."""
        print("="*60)
        print("STEP 1: EXTRACTING CHRIS'S VOICE SEGMENTS")
        print("="*60)
        
        if not self.raw_audio_dir.exists():
            print(f" Raw audio directory not found: {self.raw_audio_dir}")
            return False
            
        # Count audio files
        audio_files = list(self.raw_audio_dir.glob("*.mp3"))
        print(f"Found {len(audio_files)} dual-speaker audio files")
        
        if len(audio_files) == 0:
            print("No audio files found!")
            return False
            
        # Initialize speaker extractor
        print("Initializing speaker diarization system...")
        self.speaker_extractor = ChrisSpeakerExtractor()
        
        # Process all files
        results = self.speaker_extractor.process_all_files(
            str(self.raw_audio_dir),
            str(self.processed_dir)
        )
        
        # Save extraction results
        results_file = self.output_dir / "chris_extraction_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"\nStep 1 Complete:")
        print(f"   - Processed: {len(results['processed_files'])} files")
        print(f"   - Chris segments extracted: {results['total_segments']}")
        print(f"   - Results saved to: {results_file}")
        
        return results['total_segments'] > 0
        
    def step_2_setup_models(self):
        """Step 2: Setup and prepare both voice cloning models."""
        print("\n" + "="*60)
        print("STEP 2: SETTING UP VOICE CLONING MODELS")
        print("="*60)
        
        # Check if Chris audio exists
        chris_files = list(self.processed_dir.glob("*.wav"))
        if len(chris_files) == 0:
            print("No Chris audio files found! Run step 1 first.")
            return False
            
        print(f"Found {len(chris_files)} Chris audio segments")
        
        # Initialize model comparison framework
        comparison_dir = self.output_dir / "model_comparison"
        self.model_comparison = ModelComparison(str(comparison_dir))
        
        # Load models
        print("Loading voice cloning models...")
        self.model_comparison.load_models()
        
        # Setup Chris voice for both models
        print("Setting up Chris's voice profile...")
        self.model_comparison.prepare_chris_voice(str(self.processed_dir))
        
        print("Step 2 Complete: Models loaded and Chris voice configured")
        return True
        
    def step_3_run_comparison(self):
        """Step 3: Run comprehensive model comparison."""
        print("\n" + "="*60)
        print("STEP 3: RUNNING MODEL COMPARISON")
        print("="*60)
        
        if self.model_comparison is None:
            print("Models not initialized! Run step 2 first.")
            return False
            
        # Get reference audio for quality metrics
        chris_files = list(self.processed_dir.glob("*.wav"))
        reference_audio = str(chris_files[0]) if chris_files else None
        
        print("Starting comprehensive model comparison...")
        print("This will test both models with multiple text samples")
        
        # Run comparison
        results = self.model_comparison.run_model_comparison(
            speaker_id="chris",
            reference_audio=reference_audio
        )
        
        # Generate detailed report
        print("Generating comparison report...")
        report = self.model_comparison.generate_comparison_report(results)
        
        # Save report
        report_file = self.model_comparison.save_report(report)
        
        # Print summary
        self.model_comparison.print_summary(results)
        
        print(f"\nStep 3 Complete:")
        print(f"   - XTTS v2 tests: {len(results.get('xtts_v2', []))}")
        print(f"   - SpeechT5 tests: {len(results.get('speecht5', []))}")
        print(f"   - Detailed report: {report_file}")
        
        return True
        
    def step_4_generate_final_report(self):
        """Step 4: Generate final comprehensive evaluation report."""
        print("\n" + "="*60)
        print("STEP 4: GENERATING FINAL EVALUATION REPORT")
        print("="*60)
        
        # Collect all results
        final_report = {
            "project_name": "Chris Voice Cloning Evaluation",
            "evaluation_summary": {},
            "model_comparison": {},
            "recommendations": [],
            "next_steps": []
        }
        
        # Load extraction results
        extraction_file = self.output_dir / "chris_extraction_results.json"
        if extraction_file.exists():
            with open(extraction_file, 'r') as f:
                extraction_data = json.load(f)
                final_report["voice_extraction"] = {
                    "total_files_processed": len(extraction_data.get('processed_files', [])),
                    "chris_segments_extracted": extraction_data.get('total_segments', 0),
                    "failed_files": len(extraction_data.get('failed_files', []))
                }
        
        # Load comparison results
        comparison_dir = self.output_dir / "model_comparison"
        comparison_files = list(comparison_dir.glob("model_comparison_report_*.json"))
        if comparison_files:
            latest_comparison = max(comparison_files, key=lambda x: x.stat().st_mtime)
            with open(latest_comparison, 'r') as f:
                comparison_data = json.load(f)
                final_report["model_comparison"] = comparison_data
        
        # Generate recommendations
        if "model_comparison" in final_report and final_report["model_comparison"]:
            summary = final_report["model_comparison"].get("summary", {})
            
            # Speed recommendation
            if "speed_winner" in summary:
                speed_winner = summary["speed_winner"]
                final_report["recommendations"].append(
                    f"For fastest generation: Use {speed_winner}"
                )
            
            # Quality recommendations
            quality_comp = summary.get("quality_comparison", {})
            if quality_comp:
                quality_winners = {}
                for metric, data in quality_comp.items():
                    winner = data.get("winner")
                    if winner:
                        quality_winners[winner] = quality_winners.get(winner, 0) + 1
                
                if quality_winners:
                    best_quality = max(quality_winners.keys(), key=lambda x: quality_winners[x])
                    final_report["recommendations"].append(
                        f"For best quality: Use {best_quality}"
                    )
        
        # Next steps
        final_report["next_steps"] = [
            "Review generated audio samples in outputs/comparisons/",
            "Fine-tune the winning model with more Chris audio data",
            "Test with longer text passages",
            "Experiment with different generation parameters",
            "Consider ensemble approaches combining both models"
        ]
        
        # Save final report
        final_report_file = self.output_dir / "final_evaluation_report.json"
        with open(final_report_file, 'w') as f:
            json.dump(final_report, f, indent=2)
            
        # Print final summary
        print("FINAL EVALUATION SUMMARY")
        print("-" * 40)
        
        if "voice_extraction" in final_report:
            extraction = final_report["voice_extraction"]
            print(f"Chris segments extracted: {extraction['chris_segments_extracted']}")
            print(f"Source files processed: {extraction['total_files_processed']}")
            
        if final_report["recommendations"]:
            print("\nRECOMMENDATIONS:")
            for rec in final_report["recommendations"]:
                print(f"   • {rec}")
                
        print(f"\nFinal report saved to: {final_report_file}")
        print(f"Evaluation complete! Check {self.output_dir} for all results.")
        
        return True
        
    def run_full_evaluation(self):
        """Run the complete evaluation pipeline."""
        print("Starting Complete Chris Voice Cloning Evaluation")
        print("This will take some time depending on your hardware...")
        
        success_count = 0
        
        # Step 1: Extract Chris voice
        if self.step_1_extract_chris_voice():
            success_count += 1
        else:
            print("Step 1 failed. Cannot continue.")
            return False
            
        # Step 2: Setup models  
        if self.step_2_setup_models():
            success_count += 1
        else:
            print("Step 2 failed. Cannot continue.")
            return False
            
        # Step 3: Run comparison
        if self.step_3_run_comparison():
            success_count += 1
        else:
            print("Step 3 failed, but continuing...")
            
        # Step 4: Generate final report
        if self.step_4_generate_final_report():
            success_count += 1
            
        print(f"\nEvaluation Complete: {success_count}/4 steps successful")
        return success_count >= 3


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Chris Voice Cloning Evaluation"
    )
    parser.add_argument(
        "--raw-dir",
        default="data/raw/verified_host_only_episodes",
        help="Directory with dual-speaker audio files"
    )
    parser.add_argument(
        "--processed-dir", 
        default="data/processed/chris",
        help="Directory to save Chris's extracted audio"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/evaluation", 
        help="Directory for evaluation results"
    )
    parser.add_argument(
        "--step",
        type=int,
        choices=[1, 2, 3, 4],
        help="Run specific step only (1-4)"
    )
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = VoiceCloningEvaluator(
        raw_audio_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        output_dir=args.output_dir
    )
    
    # Run specific step or full evaluation
    if args.step:
        step_functions = {
            1: evaluator.step_1_extract_chris_voice,
            2: evaluator.step_2_setup_models,
            3: evaluator.step_3_run_comparison,
            4: evaluator.step_4_generate_final_report
        }
        
        success = step_functions[args.step]()
        print(f"\nStep {args.step} {'Success' if success else 'Failed'}")
    else:
        # Run full evaluation
        success = evaluator.run_full_evaluation()
        if success:
            print("\nFull evaluation completed successfully!")
        else:
            print("\nEvaluation completed with some issues. Check logs above.")


if __name__ == "__main__":
    main()