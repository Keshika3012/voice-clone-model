#!/usr/bin/env python3
"""
Audio Processing for Voice Cloning
Process podcast episodes to extract clean voice samples for cloning
"""

import os
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import logging
from pydub import AudioSegment
from pydub.silence import split_on_silence
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioProcessor:
    """Process audio files for voice cloning training"""
    
    def __init__(self, input_dir: str = "data/raw", output_dir: str = "data/processed"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Processing parameters
        self.target_sr = 22050  # Target sample rate for TTS
        self.min_segment_length = 3.0  # Minimum segment length in seconds
        self.max_segment_length = 15.0  # Maximum segment length in seconds
        self.silence_thresh = -40  # dB threshold for silence detection
        self.min_silence_len = 500  # Minimum silence length in ms
        
    def process_episode(self, audio_file: str) -> Dict:
        """Process a single episode file"""
        logger.info(f"🎵 Processing: {Path(audio_file).name}")
        
        try:
            # Load audio
            audio = AudioSegment.from_file(audio_file)
            
            # Normalize audio
            audio = self._normalize_audio(audio)
            
            # Split on silence to get speech segments
            segments = self._split_audio_segments(audio)
            
            # Filter and clean segments
            clean_segments = self._filter_segments(segments)
            
            # Save segments
            saved_files = self._save_segments(clean_segments, audio_file)
            
            return {
                'original_file': audio_file,
                'total_segments': len(segments),
                'clean_segments': len(clean_segments),
                'saved_files': saved_files,
                'total_duration': len(clean_segments) * self.max_segment_length
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to process {audio_file}: {e}")
            return None
    
    def _normalize_audio(self, audio: AudioSegment) -> AudioSegment:
        """Normalize audio levels"""
        # Normalize to -20dB
        target_dBFS = -20
        change_in_dBFS = target_dBFS - audio.dBFS
        return audio.apply_gain(change_in_dBFS)
    
    def _split_audio_segments(self, audio: AudioSegment) -> List[AudioSegment]:
        """Split audio into segments based on silence"""
        logger.info("✂️ Splitting audio on silence")
        
        segments = split_on_silence(
            audio,
            min_silence_len=self.min_silence_len,
            silence_thresh=self.silence_thresh,
            keep_silence=250  # Keep 250ms of silence at edges
        )
        
        logger.info(f"📊 Found {len(segments)} segments")
        return segments
    
    def _filter_segments(self, segments: List[AudioSegment]) -> List[AudioSegment]:
        """Filter segments by duration and quality"""
        logger.info("🔍 Filtering segments")
        
        filtered = []
        
        for i, segment in enumerate(segments):
            duration = len(segment) / 1000.0  # Convert to seconds
            
            # Check duration
            if duration < self.min_segment_length:
                continue
                
            # Truncate if too long
            if duration > self.max_segment_length:
                segment = segment[:int(self.max_segment_length * 1000)]
            
            # Check audio quality (basic)
            if self._is_good_quality(segment):
                filtered.append(segment)
        
        logger.info(f"✅ Kept {len(filtered)} segments after filtering")
        return filtered
    
    def _is_good_quality(self, segment: AudioSegment) -> bool:
        """Basic audio quality check"""
        # Check if segment is not too quiet
        if segment.dBFS < -45:
            return False
        
        # Check if segment is not too loud (clipping)
        if segment.max_dBFS > -1:
            return False
        
        return True
    
    def _save_segments(self, segments: List[AudioSegment], original_file: str) -> List[str]:
        """Save audio segments to files"""
        base_name = Path(original_file).stem
        episode_dir = self.output_dir / "episodes" / base_name
        episode_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        
        for i, segment in enumerate(segments):
            filename = f"{base_name}_segment_{i+1:03d}.wav"
            file_path = episode_dir / filename
            
            # Export as WAV with target sample rate
            segment.export(
                str(file_path),
                format="wav",
                parameters=["-ar", str(self.target_sr)]
            )
            
            saved_files.append(str(file_path))
        
        logger.info(f"💾 Saved {len(saved_files)} segments to {episode_dir}")
        return saved_files
    
    def create_speaker_datasets(self, processed_episodes: List[Dict]):
        """Create separate datasets for Chris and Daniel"""
        logger.info("👥 Creating speaker-specific datasets")
        
        # Create speaker directories
        chris_dir = self.output_dir / "chris"
        daniel_dir = self.output_dir / "daniel"
        chris_dir.mkdir(exist_ok=True)
        daniel_dir.mkdir(exist_ok=True)
        
        # For now, we'll put all segments in both directories
        # In a real scenario, you'd need speaker diarization
        # This is a simplified approach
        
        chris_files = []
        daniel_files = []
        
        for episode_data in processed_episodes:
            if not episode_data:
                continue
                
            for i, file_path in enumerate(episode_data['saved_files']):
                # Simple alternating assignment (not ideal, but works for demo)
                # In practice, you'd use speaker diarization here
                if i % 2 == 0:  # Even segments to Chris
                    dest_path = chris_dir / Path(file_path).name
                    self._copy_audio_file(file_path, dest_path)
                    chris_files.append(str(dest_path))
                else:  # Odd segments to Daniel
                    dest_path = daniel_dir / Path(file_path).name
                    self._copy_audio_file(file_path, dest_path)
                    daniel_files.append(str(dest_path))
        
        logger.info(f"👨 Chris: {len(chris_files)} files")
        logger.info(f"👨 Daniel: {len(daniel_files)} files")
        
        return {
            'chris': chris_files,
            'daniel': daniel_files
        }
    
    def _copy_audio_file(self, src: str, dest: Path):
        """Copy audio file to destination"""
        try:
            audio = AudioSegment.from_wav(src)
            audio.export(str(dest), format="wav")
        except Exception as e:
            logger.error(f"❌ Failed to copy {src}: {e}")
    
    def process_all_episodes(self, episode_dir: str = None) -> Dict:
        """Process all episodes in the input directory"""
        if episode_dir is None:
            episode_dir = self.input_dir / "practical_ai_episodes"
        else:
            episode_dir = Path(episode_dir)
        
        if not episode_dir.exists():
            logger.error(f"❌ Episode directory not found: {episode_dir}")
            return None
        
        # Find audio files
        audio_files = list(episode_dir.glob("*.mp3")) + list(episode_dir.glob("*.wav"))
        
        if not audio_files:
            logger.error("❌ No audio files found")
            return None
        
        logger.info(f"🎵 Found {len(audio_files)} audio files to process")
        
        processed_episodes = []
        
        for i, audio_file in enumerate(audio_files, 1):
            logger.info(f"[{i}/{len(audio_files)}] Processing episode")
            result = self.process_episode(str(audio_file))
            if result:
                processed_episodes.append(result)
        
        # Create speaker datasets
        speaker_datasets = self.create_speaker_datasets(processed_episodes)
        
        # Save processing summary
        summary = {
            'total_episodes': len(audio_files),
            'processed_episodes': len(processed_episodes),
            'total_segments': sum(ep['clean_segments'] for ep in processed_episodes if ep),
            'speaker_datasets': speaker_datasets,
            'processing_params': {
                'target_sr': self.target_sr,
                'min_segment_length': self.min_segment_length,
                'max_segment_length': self.max_segment_length,
                'silence_thresh': self.silence_thresh
            }
        }
        
        summary_file = self.output_dir / "processing_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"💾 Saved processing summary: {summary_file}")
        
        return summary


class SpeakerDiarization:
    """Simple speaker diarization using basic audio features"""
    
    def __init__(self):
        self.features_cache = {}
    
    def extract_features(self, audio_file: str) -> np.ndarray:
        """Extract basic audio features for speaker identification"""
        if audio_file in self.features_cache:
            return self.features_cache[audio_file]
        
        y, sr = librosa.load(audio_file, sr=22050)
        
        # Extract features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
        
        # Compute statistics
        features = np.hstack([
            np.mean(mfccs, axis=1),
            np.std(mfccs, axis=1),
            np.mean(spectral_centroids),
            np.std(spectral_centroids),
            np.mean(spectral_rolloff),
            np.std(spectral_rolloff),
            np.mean(zero_crossing_rate),
            np.std(zero_crossing_rate)
        ])
        
        self.features_cache[audio_file] = features
        return features
    
    def cluster_speakers(self, audio_files: List[str], n_speakers: int = 2) -> Dict[str, int]:
        """Simple clustering to separate speakers"""
        from sklearn.cluster import KMeans
        
        logger.info(f"🎭 Clustering {len(audio_files)} segments into {n_speakers} speakers")
        
        # Extract features for all files
        features = []
        for audio_file in audio_files:
            feat = self.extract_features(audio_file)
            features.append(feat)
        
        features = np.array(features)
        
        # Cluster
        kmeans = KMeans(n_clusters=n_speakers, random_state=42)
        labels = kmeans.fit_predict(features)
        
        # Map files to speakers
        speaker_assignment = {}
        for audio_file, label in zip(audio_files, labels):
            speaker_assignment[audio_file] = label
        
        return speaker_assignment


def main():
    """Main function"""
    processor = AudioProcessor()
    
    try:
        results = processor.process_all_episodes()
        
        if results:
            print("\n" + "="*60)
            print("🎵 AUDIO PROCESSING SUMMARY")
            print("="*60)
            print(f"Episodes Processed: {results['processed_episodes']}/{results['total_episodes']}")
            print(f"Total Segments: {results['total_segments']}")
            print(f"Chris Files: {len(results['speaker_datasets']['chris'])}")
            print(f"Daniel Files: {len(results['speaker_datasets']['daniel'])}")
            print("\n✅ Processing complete!")
        else:
            print("❌ Processing failed")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())