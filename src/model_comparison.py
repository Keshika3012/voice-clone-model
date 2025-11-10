#!/usr/bin/env python3
"""
Voice Cloning Model Comparison Framework

This module provides comprehensive evaluation and comparison between different
voice cloning models (SpeechT5 vs XTTS v2) using multiple audio quality metrics.
"""

import os
import json
import time
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Import our model implementations
from voice_cloner import VoiceCloner, GenerationConfig
from speecht5_cloner import SpeechT5VoiceCloner

# Evaluation metrics
try:
    from metrics.audio_scorer import compare_audio as score_compare
except Exception:
    score_compare = None

# Fallback: attempt direct metrics if scorer not available
try:
    from pesq import pesq
    from pystoi import stoi
except ImportError:
    print("Warning: PESQ and STOI not available. Install with: pip install pesq pystoi")
    pesq = None
    stoi = None


@dataclass
class ComparisonResult:
    """Results from model comparison."""
    model_name: str
    speaker_id: str
    text: str
    audio_path: str
    generation_time: float
    metrics: Dict[str, float]
    reference_audio: Optional[str] = None


class ModelComparison:
    """
    Comprehensive model comparison framework for voice cloning systems.
    """
    
    def __init__(self, output_dir: str = "outputs/comparisons"):
        """
        Initialize the model comparison framework.
        
        Args:
            output_dir: Directory to save comparison results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize models
        self.xtts_model = None
        self.speecht5_model = None
        
        # Results storage
        self.results: List[ComparisonResult] = []
        
        # Test texts for evaluation
        self.test_texts = [
            "Hello, this is a test of the voice cloning system.",
            "Artificial intelligence is transforming how we interact with technology.",
            "The weather today is absolutely beautiful with clear blue skies.",
            "Machine learning models require careful evaluation and validation.",
            "Voice cloning technology has made remarkable advances in recent years."
        ]
        
    def load_models(self):
        """Load both XTTS v2 and SpeechT5 models."""
        print("Loading XTTS v2 model...")
        try:
            self.xtts_model = VoiceCloner()
            print("XTTS v2 loaded successfully")
        except Exception as e:
            print(f"Failed to load XTTS v2: {e}")
            
        print("Loading SpeechT5 model...")
        try:
            self.speecht5_model = SpeechT5VoiceCloner()
            print("SpeechT5 loaded successfully")
        except Exception as e:
            print(f"Failed to load SpeechT5: {e}")
            
    def prepare_chris_voice(self, chris_audio_dir: str):
        """
        Prepare Chris's voice for both models.
        
        Args:
            chris_audio_dir: Directory containing Chris's extracted audio segments
        """
        chris_dir = Path(chris_audio_dir)
        if not chris_dir.exists():
            print(f"Chris audio directory not found: {chris_dir}")
            return
            
        # Get Chris audio files
        audio_files = list(chris_dir.glob("*.wav"))
        if not audio_files:
            print("No audio files found in Chris directory")
            return
            
        print(f"📁 Found {len(audio_files)} Chris audio files")
        
        # Setup for XTTS v2
        if self.xtts_model:
            try:
                self.xtts_model.add_voice_profile(
                    name="Chris",
                    audio_files=[str(f) for f in audio_files[:5]],  # Use first 5 files
                    speaker_id="chris"
                )
                print("Chris voice added to XTTS v2")
            except Exception as e:
                print(f"Failed to add Chris to XTTS v2: {e}")
        
        # Setup for SpeechT5
        if self.speecht5_model:
            try:
                self.speecht5_model.create_speaker_embedding(
                    audio_files=[str(f) for f in audio_files[:10]], 
                    speaker_name="chris"
                )
                print("Chris voice added to SpeechT5")
            except Exception as e:
                print(f"Failed to add Chris to SpeechT5: {e}")
                
    def calculate_audio_metrics(self, 
                              reference_audio: str, 
                              generated_audio: str,
                              sample_rate: int = 16000) -> Dict[str, float]:
        """
        Calculate comprehensive audio quality metrics.
        Uses improved preprocessing/alignment for PESQ/STOI if available.
        """
        metrics = {}
        
        try:
            # Preferred path: our scorer with trim/normalize/align
            if score_compare is not None:
                s = score_compare(reference_audio, generated_audio, target_sr=sample_rate, align=True)
                metrics['pesq'] = float(s.get('pesq', 0.0))
                metrics['stoi'] = float(s.get('stoi', 0.0))
            else:
                # Fallback simple load/score
                ref_audio, _ = librosa.load(reference_audio, sr=sample_rate)
                gen_audio, _ = librosa.load(generated_audio, sr=sample_rate)
                min_len = min(len(ref_audio), len(gen_audio))
                ref_audio = ref_audio[:min_len]
                gen_audio = gen_audio[:min_len]
                if pesq is not None and len(ref_audio) > 0 and len(gen_audio) > 0:
                    try:
                        metrics['pesq'] = float(pesq(sample_rate, ref_audio, gen_audio, 'wb'))
                    except:
                        metrics['pesq'] = 0.0
                else:
                    metrics['pesq'] = 0.0
                if stoi is not None:
                    try:
                        metrics['stoi'] = float(stoi(ref_audio, gen_audio, sample_rate, extended=False))
                    except:
                        metrics['stoi'] = 0.0
                else:
                    metrics['stoi'] = 0.0

            # Spectral similarity metrics (librosa features)
            ref_audio, _ = librosa.load(reference_audio, sr=sample_rate)
            gen_audio, _ = librosa.load(generated_audio, sr=sample_rate)
            min_len = min(len(ref_audio), len(gen_audio))
            ref_audio = ref_audio[:min_len]
            gen_audio = gen_audio[:min_len]

            ref_mfcc = librosa.feature.mfcc(y=ref_audio, sr=sample_rate, n_mfcc=13)
            gen_mfcc = librosa.feature.mfcc(y=gen_audio, sr=sample_rate, n_mfcc=13)
            min_frames = min(ref_mfcc.shape[1], gen_mfcc.shape[1])
            ref_mfcc = ref_mfcc[:, :min_frames]
            gen_mfcc = gen_mfcc[:, :min_frames]
            if min_frames > 0:
                mfcc_corr = np.corrcoef(ref_mfcc.flatten(), gen_mfcc.flatten())[0, 1]
                metrics['mfcc_similarity'] = float(mfcc_corr) if not np.isnan(mfcc_corr) else 0.0
            else:
                metrics['mfcc_similarity'] = 0.0

            ref_f0 = librosa.yin(ref_audio, fmin=50, fmax=400)
            gen_f0 = librosa.yin(gen_audio, fmin=50, fmax=400)
            ref_f0_voiced = ref_f0[ref_f0 > 0]
            gen_f0_voiced = gen_f0[gen_f0 > 0]
            if len(ref_f0_voiced) > 0 and len(gen_f0_voiced) > 0:
                f0_diff = abs(np.mean(ref_f0_voiced) - np.mean(gen_f0_voiced))
                f0_similarity = max(0.0, 1 - f0_diff / np.mean(ref_f0_voiced))
                metrics['f0_similarity'] = float(f0_similarity)
            else:
                metrics['f0_similarity'] = 0.0

            ref_sc = np.mean(librosa.feature.spectral_centroid(y=ref_audio, sr=sample_rate))
            gen_sc = np.mean(librosa.feature.spectral_centroid(y=gen_audio, sr=sample_rate))
            sc_similarity = 1 - abs(ref_sc - gen_sc) / max(ref_sc, gen_sc)
            metrics['spectral_centroid_similarity'] = float(sc_similarity)

        except Exception as e:
            print(f"⚠️ Error calculating metrics: {e}")
            metrics = {
                'pesq': 0.0,
                'stoi': 0.0,
                'mfcc_similarity': 0.0,
                'f0_similarity': 0.0,
                'spectral_centroid_similarity': 0.0
            }
        
        return metrics
        
    def run_model_comparison(self, 
                           speaker_id: str = "chris",
                           reference_audio: Optional[str] = None) -> Dict[str, List[ComparisonResult]]:
        """
        Run comprehensive comparison between models.
        
        Args:
            speaker_id: Speaker to use for comparison
            reference_audio: Reference audio for quality metrics
            
        Returns:
            Dictionary with results for each model
        """
        results = {
            'xtts_v2': [],
            'speecht5': []
        }
        
        print(" Starting model comparison...")
        print(f" Testing with {len(self.test_texts)} different texts")
        
        for i, text in enumerate(self.test_texts):
            print(f"\n--- Testing text {i+1}/{len(self.test_texts)} ---")
            print(f"Text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            
            # Test XTTS v2
            if self.xtts_model and speaker_id in self.xtts_model.voice_profiles:
                try:
                    print("🎤 Generating with XTTS v2...")
                    start_time = time.time()
                    
                    output_path = str(self.output_dir / f"xtts_v2_test_{i+1:02d}.wav")
                    self.xtts_model.generate_speech(text, speaker_id, output_path)
                    
                    generation_time = time.time() - start_time
                    
                    # Calculate metrics if reference available
                    metrics = {}
                    if reference_audio and os.path.exists(reference_audio):
                        metrics = self.calculate_audio_metrics(reference_audio, output_path)
                    
                    result = ComparisonResult(
                        model_name="XTTS v2",
                        speaker_id=speaker_id,
                        text=text,
                        audio_path=output_path,
                        generation_time=generation_time,
                        metrics=metrics,
                        reference_audio=reference_audio
                    )
                    results['xtts_v2'].append(result)
                    print(f"XTTS v2 completed in {generation_time:.2f}s")
                    
                except Exception as e:
                    print(f" XTTS v2 failed: {e}")
            
            # Test SpeechT5
            if self.speecht5_model and speaker_id in self.speecht5_model.speaker_embeddings:
                try:
                    print(" Generating with SpeechT5...")
                    start_time = time.time()
                    
                    output_path = str(self.output_dir / f"speecht5_test_{i+1:02d}.wav")
                    self.speecht5_model.generate_speech(text, speaker_id, output_path)
                    
                    generation_time = time.time() - start_time
                    
                    # Calculate metrics if reference available
                    metrics = {}
                    if reference_audio and os.path.exists(reference_audio):
                        metrics = self.calculate_audio_metrics(reference_audio, output_path)
                    
                    result = ComparisonResult(
                        model_name="SpeechT5",
                        speaker_id=speaker_id,
                        text=text,
                        audio_path=output_path,
                        generation_time=generation_time,
                        metrics=metrics,
                        reference_audio=reference_audio
                    )
                    results['speecht5'].append(result)
                    print(f" SpeechT5 completed in {generation_time:.2f}s")
                    
                except Exception as e:
                    print(f" SpeechT5 failed: {e}")
        
        self.results = results['xtts_v2'] + results['speecht5']
        return results
        
    def generate_comparison_report(self, results: Dict[str, List[ComparisonResult]]) -> Dict[str, Any]:
        """
        Generate comprehensive comparison report.
        
        Args:
            results: Results from model comparison
            
        Returns:
            Report dictionary
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {},
            'detailed_results': {},
            'model_stats': {}
        }
        
        for model_name, model_results in results.items():
            if not model_results:
                continue
                
            # Calculate averages
            avg_gen_time = np.mean([r.generation_time for r in model_results])
            
            # Average metrics (if available)
            metric_averages = {}
            if model_results[0].metrics:
                for metric_name in model_results[0].metrics.keys():
                    metric_values = [r.metrics.get(metric_name, 0.0) for r in model_results]
                    metric_averages[metric_name] = float(np.mean(metric_values))
            
            report['model_stats'][model_name] = {
                'average_generation_time': float(avg_gen_time),
                'total_tests': len(model_results),
                'successful_generations': len([r for r in model_results if os.path.exists(r.audio_path)]),
                'average_metrics': metric_averages
            }
            
            report['detailed_results'][model_name] = [
                {
                    'text': r.text,
                    'generation_time': r.generation_time,
                    'metrics': r.metrics,
                    'audio_path': r.audio_path
                } for r in model_results
            ]
        
        # Generate summary comparison
        if len(results) >= 2:
            models = list(results.keys())
            model_a, model_b = models[0], models[1]
            
            if results[model_a] and results[model_b]:
                stats_a = report['model_stats'][model_a]
                stats_b = report['model_stats'][model_b]
                
                report['summary'] = {
                    'speed_winner': model_a if stats_a['average_generation_time'] < stats_b['average_generation_time'] else model_b,
                    'quality_comparison': {}
                }
                
                # Compare quality metrics
                for metric in stats_a.get('average_metrics', {}):
                    if metric in stats_b.get('average_metrics', {}):
                        score_a = stats_a['average_metrics'][metric]
                        score_b = stats_b['average_metrics'][metric]
                        winner = model_a if score_a > score_b else model_b
                        report['summary']['quality_comparison'][metric] = {
                            'winner': winner,
                            f'{model_a}_score': score_a,
                            f'{model_b}_score': score_b
                        }
        
        return report
        
    def save_report(self, report: Dict[str, Any], filename: str = None):
        """Save comparison report to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"model_comparison_report_{timestamp}.json"
            
        report_path = self.output_dir / filename
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f" Report saved to: {report_path}")
        return str(report_path)
        
    def print_summary(self, results: Dict[str, List[ComparisonResult]]):
        """Print a summary of comparison results."""
        print("\n" + "="*60)
        print("MODEL COMPARISON SUMMARY")
        print("="*60)
        
        for model_name, model_results in results.items():
            if not model_results:
                print(f"\n {model_name.upper()}: No results")
                continue
                
            print(f"\n🤖 {model_name.upper()}:")
            print(f"   Tests completed: {len(model_results)}")
            
            avg_time = np.mean([r.generation_time for r in model_results])
            print(f"   Average generation time: {avg_time:.2f}s")
            
            if model_results[0].metrics:
                print("   Average quality metrics:")
                for metric_name in model_results[0].metrics.keys():
                    values = [r.metrics.get(metric_name, 0.0) for r in model_results]
                    avg_score = np.mean(values)
                    print(f"     {metric_name}: {avg_score:.3f}")
        
        print("\n" + "="*60)


def main():
    """Example usage of model comparison framework."""
    # Initialize comparison
    comparison = ModelComparison()
    
    # Load models
    comparison.load_models()
    
    # Setup Chris voice (you'll need to run speaker extraction first)
    chris_audio_dir = "data/processed/chris"
    comparison.prepare_chris_voice(chris_audio_dir)
    
    # Run comparison
    results = comparison.run_model_comparison(speaker_id="chris")
    
    # Generate and save report
    report = comparison.generate_comparison_report(results)
    comparison.save_report(report)
    
    # Print summary
    comparison.print_summary(results)


if __name__ == "__main__":
    main()