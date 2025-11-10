#!/usr/bin/env python3
"""
Advanced Voice Cloning System
A comprehensive voice cloning solution using state-of-the-art TTS models
"""

import os
import json
import time
import torch
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
import logging
from tqdm import tqdm

# TTS imports
try:
    from TTS.api import TTS
except Exception as e:
    print("Failed to import TTS. Install with: pip install TTS (macOS may need: brew install mecab mecab-ipadic && pip install mecab-python3 unidic-lite)")
    raise

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VoiceProfile:
    """Data class for storing voice profile information"""
    name: str
    speaker_id: str
    reference_audio: str
    audio_files: List[str]
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class GenerationConfig:
    """Configuration for voice generation"""
    language: str = "en"
    speed: float = 1.0
    temperature: float = 0.75
    length_penalty: float = 1.0
    repetition_penalty: float = 5.0
    top_k: int = 50
    top_p: float = 0.85


class VoiceCloner:
    """Advanced Voice Cloning System"""
    
    def __init__(self, 
                 model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
                 device: str = "auto",
                 data_dir: str = "data",
                 output_dir: str = "outputs",
                 # Optional local model overrides (Coqui TTS API)
                 model_path: Optional[str] = None,
                 config_path: Optional[str] = None,
                 vocoder_path: Optional[str] = None,
                 vocoder_config_path: Optional[str] = None,
                 encoder_path: Optional[str] = None,
                 encoder_config_path: Optional[str] = None):
        """
        Initialize the Voice Cloner
        
        Args:
            model_name: Remote model identifier (default: XTTS v2)
            device: Device to run on ('cpu', 'cuda', or 'auto')
            data_dir: Directory containing voice data
            output_dir: Directory for generated outputs
            model_path: Local path to a TTS model checkpoint directory/file
            config_path: Local path to the TTS model config JSON
            vocoder_path: Optional local path to a vocoder checkpoint
            vocoder_config_path: Optional local path to a vocoder config JSON
            encoder_path: Optional local path to an encoder checkpoint
            encoder_config_path: Optional local path to an encoder config JSON
        """
        self.model_name = model_name
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        
        # Optional local model overrides (env vars take effect if args not provided)
        self.model_path = model_path or os.getenv("TTS_MODEL_PATH")
        self.config_path = config_path or os.getenv("TTS_CONFIG_PATH")
        self.vocoder_path = vocoder_path or os.getenv("TTS_VOCODER_PATH")
        self.vocoder_config_path = vocoder_config_path or os.getenv("TTS_VOCODER_CONFIG_PATH")
        self.encoder_path = encoder_path or os.getenv("TTS_ENCODER_PATH")
        self.encoder_config_path = encoder_config_path or os.getenv("TTS_ENCODER_CONFIG_PATH")
        
        # Setup device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"🚀 Initializing VoiceCloner on {self.device}")
        
        # Initialize TTS model
        self._load_model()
        
        # Voice profiles storage
        self.voice_profiles: Dict[str, VoiceProfile] = {}
        
        # Create directories
        self._setup_directories()
        
        # Load existing profiles
        self._load_voice_profiles()
    
    def _setup_directories(self):
        """Create necessary directories"""
        directories = [
            self.data_dir / "raw",
            self.data_dir / "processed", 
            self.data_dir / "models",
            self.output_dir / "samples",
            self.output_dir / "demos",
            self.output_dir / "comparisons"
        ]
        
        for dir_path in directories:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def _load_model(self):
        """Load the TTS model.
        Prefers local paths if provided; otherwise falls back to remote model_name.
        """
        try:
            # Allowlist XTTS config class for PyTorch safe deserialization (PyTorch >=2.5)
            try:
                from torch.serialization import add_safe_globals  # PyTorch 2.5+
            except Exception:
                add_safe_globals = None

            # Allowlist classes used in XTTS checkpoints for torch.load safe mode
            if add_safe_globals is not None:
                try:
                    from TTS.tts.configs.xtts_config import XttsConfig  # type: ignore
                    add_safe_globals([XttsConfig])
                except Exception:
                    pass
                try:
                    from TTS.tts.models.xtts import XttsAudioConfig  # type: ignore
                    add_safe_globals([XttsAudioConfig])
                except Exception:
                    pass
                try:
                    from TTS.config.shared_configs import BaseDatasetConfig  # type: ignore
                    add_safe_globals([BaseDatasetConfig])
                except Exception:
                    pass

            # As a fallback, force torch.load to use weights_only=False to allow legacy checkpoints
            try:
                import torch  # type: ignore
                import torch.serialization as _ts  # type: ignore
                _orig_load = _ts.load
                def _patched_load(*args, **kwargs):
                    kwargs.setdefault('weights_only', False)
                    return _orig_load(*args, **kwargs)
                _ts.load = _patched_load  # type: ignore
                torch.load = _patched_load  # type: ignore
            except Exception:
                pass

            use_local = (
                self.model_path and self.config_path and
                Path(self.model_path).expanduser().exists() and
                Path(self.config_path).expanduser().exists()
            )

            if use_local:
                mp = str(Path(self.model_path).expanduser())
                cp = str(Path(self.config_path).expanduser())
                vp = str(Path(self.vocoder_path).expanduser()) if self.vocoder_path else None
                vcp = str(Path(self.vocoder_config_path).expanduser()) if self.vocoder_config_path else None
                ep = str(Path(self.encoder_path).expanduser()) if self.encoder_path else None
                ecp = str(Path(self.encoder_config_path).expanduser()) if self.encoder_config_path else None

                logger.info(f"Loading local model: model_path={mp}, config_path={cp}")
                self.tts = TTS(
                    model_path=mp,
                    config_path=cp,
                    vocoder_path=vp,
                    vocoder_config_path=vcp,
                    encoder_path=ep,
                    encoder_config_path=ecp,
                    gpu=(self.device == "cuda"),
                )
            else:
                logger.info(f"Loading remote model: {self.model_name}")
                self.tts = TTS(model_name=self.model_name, gpu=(self.device == "cuda"))

            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _load_voice_profiles(self):
        """Load voice profiles from data directory"""
        raw_data_dir = self.data_dir / "raw"
        
        if not raw_data_dir.exists():
            logger.warning(f"Raw data directory not found: {raw_data_dir}")
            return
        
        # Look for speaker directories
        for speaker_dir in raw_data_dir.iterdir():
            if speaker_dir.is_dir() and not speaker_dir.name.startswith('.'):
                self._load_speaker_profile(speaker_dir)
        
        logger.info(f"Loaded {len(self.voice_profiles)} voice profiles")
    
    def _load_speaker_profile(self, speaker_dir: Path):
        """Load a single speaker profile"""
        speaker_name = speaker_dir.name
        audio_files = list(speaker_dir.glob("*.wav")) + list(speaker_dir.glob("*.mp3"))
        
        if not audio_files:
            logger.warning(f"No audio files found in {speaker_dir}")
            return
        
        # Use first audio file as reference
        reference_audio = str(audio_files[0])
        
        # Create profile
        profile = VoiceProfile(
            name=speaker_name,
            speaker_id=speaker_name.lower().replace(" ", "_"),
            reference_audio=reference_audio,
            audio_files=[str(f) for f in audio_files]
        )
        
        self.voice_profiles[profile.speaker_id] = profile
        logger.info(f"👤 Loaded profile: {speaker_name} ({len(audio_files)} files)")
    
    def add_voice_profile(self, 
                         name: str, 
                         audio_files: Union[str, List[str]], 
                         speaker_id: str = None) -> str:
        """
        Add a new voice profile
        
        Args:
            name: Speaker name
            audio_files: Path to audio file(s)
            speaker_id: Custom speaker ID (optional)
            
        Returns:
            speaker_id: The assigned speaker ID
        """
        if speaker_id is None:
            speaker_id = name.lower().replace(" ", "_")
        
        # Handle single file or list of files
        if isinstance(audio_files, str):
            audio_files = [audio_files]
        
        # Validate files exist
        valid_files = []
        for file_path in audio_files:
            if os.path.exists(file_path):
                valid_files.append(file_path)
            else:
                logger.warning(f"⚠️ Audio file not found: {file_path}")
        
        if not valid_files:
            raise ValueError("No valid audio files provided")
        
        # Create profile
        profile = VoiceProfile(
            name=name,
            speaker_id=speaker_id,
            reference_audio=valid_files[0],
            audio_files=valid_files
        )
        
        self.voice_profiles[speaker_id] = profile
        logger.info(f"Added voice profile: {name} (ID: {speaker_id})")
        
        return speaker_id
    
    def list_voices(self):
        """List all available voice profiles"""
        if not self.voice_profiles:
            print("No voice profiles loaded")
            return
        
        print("Available Voice Profiles:")
        print("-" * 50)
        for speaker_id, profile in self.voice_profiles.items():
            print(f"• {profile.name}")
            print(f"  ID: {speaker_id}")
            print(f"  Files: {len(profile.audio_files)}")
            print(f"  Reference: {Path(profile.reference_audio).name}")
            print()
    
    def analyze_voice(self, speaker_id: str) -> Dict:
        """
        Analyze voice characteristics of a speaker
        
        Args:
            speaker_id: Speaker ID to analyze
            
        Returns:
            Dictionary with voice analysis results
        """
        if speaker_id not in self.voice_profiles:
            raise ValueError(f"Speaker '{speaker_id}' not found")
        
        profile = self.voice_profiles[speaker_id]
        audio_path = profile.reference_audio
        
        logger.info(f"Analyzing voice: {profile.name}")
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=None)
        
        # Extract features
        analysis = {
            'speaker_id': speaker_id,
            'name': profile.name,
            'duration': len(y) / sr,
            'sample_rate': sr,
        }
        
        # Fundamental frequency (pitch)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')
        )
        analysis['f0_mean'] = float(np.nanmean(f0)) if not np.all(np.isnan(f0)) else 0
        analysis['f0_std'] = float(np.nanstd(f0)) if not np.all(np.isnan(f0)) else 0
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
        analysis['spectral_centroid'] = float(np.mean(spectral_centroids))
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        analysis['spectral_rolloff'] = float(np.mean(spectral_rolloff))
        
        # Tempo
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        analysis['tempo'] = float(tempo)
        
        # Energy
        rms = librosa.feature.rms(y=y)
        analysis['energy_mean'] = float(np.mean(rms))
        
        return analysis
    
    def generate_speech(self, 
                       text: str, 
                       speaker_id: str,
                       output_path: str = None,
                       config: GenerationConfig = None) -> str:
        """
        Generate speech for given text and speaker
        
        Args:
            text: Text to synthesize
            speaker_id: Speaker ID to use
            output_path: Output file path (optional)
            config: Generation configuration
            
        Returns:
            Path to generated audio file
        """
        if speaker_id not in self.voice_profiles:
            raise ValueError(f"Speaker '{speaker_id}' not found. Available: {list(self.voice_profiles.keys())}")
        
        if config is None:
            config = GenerationConfig()
        
        profile = self.voice_profiles[speaker_id]
        
        # Generate output path if not provided
        if output_path is None:
            timestamp = int(time.time())
            filename = f"{speaker_id}_{timestamp}.wav"
            output_path = str(self.output_dir / "samples" / filename)
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Generating speech for {profile.name}")
        logger.info(f"Text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        
        start_time = time.time()
        
        try:
            # Generate speech
            self.tts.tts_to_file(
                text=text,
                speaker_wav=profile.reference_audio,
                language=config.language,
                file_path=output_path
            )
            
            # Apply speed adjustment if needed
            if config.speed != 1.0:
                self._adjust_speed(output_path, config.speed)
            
            generation_time = time.time() - start_time
            logger.info(f"Speech generated in {generation_time:.2f}s")
            logger.info(f"Saved to: {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    def _adjust_speed(self, audio_path: str, speed_factor: float):
        """Adjust audio playback speed"""
        y, sr = librosa.load(audio_path)
        y_stretched = librosa.effects.time_stretch(y, rate=speed_factor)
        sf.write(audio_path, y_stretched, sr)
        logger.info(f"⚡ Applied speed adjustment: {speed_factor}x")
    
    def batch_generate(self, 
                      texts: List[str], 
                      speaker_id: str,
                      output_dir: str = None,
                      config: GenerationConfig = None) -> List[str]:
        """
        Generate multiple audio files in batch
        
        Args:
            texts: List of texts to synthesize
            speaker_id: Speaker ID to use
            output_dir: Output directory for batch files
            config: Generation configuration
            
        Returns:
            List of generated file paths
        """
        if output_dir is None:
            output_dir = str(self.output_dir / "demos" / "batch")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        generated_files = []
        
        logger.info(f"Starting batch generation: {len(texts)} files for {speaker_id}")
        
        for i, text in enumerate(tqdm(texts, desc="Generating")):
            output_path = os.path.join(output_dir, f"{speaker_id}_batch_{i+1:03d}.wav")
            
            try:
                generated_path = self.generate_speech(text, speaker_id, output_path, config)
                generated_files.append(generated_path)
            except Exception as e:
                logger.error(f"Failed to generate file {i+1}: {e}")
                continue
        
        logger.info(f"🎉 Batch complete: {len(generated_files)}/{len(texts)} files generated")
        return generated_files
    
    def compare_speakers(self, 
                        text: str, 
                        speaker_ids: List[str] = None,
                        output_dir: str = None) -> Dict[str, str]:
        """
        Generate the same text with multiple speakers for comparison
        
        Args:
            text: Text to synthesize
            speaker_ids: List of speaker IDs (uses all if None)
            output_dir: Output directory
            
        Returns:
            Dictionary mapping speaker_id to output file path
        """
        if speaker_ids is None:
            speaker_ids = list(self.voice_profiles.keys())
        
        if output_dir is None:
            timestamp = int(time.time())
            output_dir = str(self.output_dir / "comparisons" / f"comparison_{timestamp}")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        logger.info(f"Generating comparison with {len(speaker_ids)} speakers")
        
        for speaker_id in tqdm(speaker_ids, desc="Comparing speakers"):
            if speaker_id not in self.voice_profiles:
                logger.warning(f" Speaker '{speaker_id}' not found, skipping")
                continue
            
            output_path = os.path.join(output_dir, f"{speaker_id}.wav")
            
            try:
                generated_path = self.generate_speech(text, speaker_id, output_path)
                results[speaker_id] = generated_path
            except Exception as e:
                logger.error(f"Failed to generate for {speaker_id}: {e}")
        
        logger.info(f"Comparison complete: {len(results)} files generated")
        return results
    
    def save_profile_config(self, config_path: str = None):
        """Save current voice profiles configuration to file"""
        if config_path is None:
            config_path = str(self.data_dir / "voice_profiles.json")
        
        config_data = {}
        for speaker_id, profile in self.voice_profiles.items():
            config_data[speaker_id] = {
                'name': profile.name,
                'reference_audio': profile.reference_audio,
                'audio_files': profile.audio_files,
                'metadata': profile.metadata
            }
        
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        logger.info(f"Saved voice profiles config to: {config_path}")
    
    def load_profile_config(self, config_path: str):
        """Load voice profiles configuration from file"""
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        for speaker_id, data in config_data.items():
            profile = VoiceProfile(
                name=data['name'],
                speaker_id=speaker_id,
                reference_audio=data['reference_audio'],
                audio_files=data['audio_files'],
                metadata=data.get('metadata', {})
            )
            self.voice_profiles[speaker_id] = profile
        
        logger.info(f"Loaded {len(config_data)} voice profiles from config")


if __name__ == "__main__":
    # Example usage
    cloner = VoiceCloner()
    cloner.list_voices()