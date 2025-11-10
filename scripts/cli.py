#!/usr/bin/env python3
"""
Voice Cloning CLI
Command-line interface for the voice cloning system
"""

import os
import sys
import click
import json
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from voice_cloner import VoiceCloner, GenerationConfig
from metrics.audio_scorer import compare_audio as score_compare


@click.group()
@click.option('--data-dir', default='data', help='Data directory path')
@click.option('--output-dir', default='outputs', help='Output directory path')
@click.option('--device', default='auto', help='Device to use (cpu/cuda/auto)')
@click.pass_context
def cli(ctx, data_dir, output_dir, device):
    """Voice Cloning Command Line Interface"""
    ctx.ensure_object(dict)
    ctx.obj['data_dir'] = data_dir
    ctx.obj['output_dir'] = output_dir
    ctx.obj['device'] = device


@cli.command()
@click.pass_context
def list_voices(ctx):
    """List all available voice profiles"""
    cloner = VoiceCloner(
        data_dir=ctx.obj['data_dir'],
        output_dir=ctx.obj['output_dir'],
        device=ctx.obj['device']
    )
    cloner.list_voices()


@cli.command()
@click.argument('name')
@click.argument('audio_files', nargs=-1, required=True)
@click.option('--speaker-id', help='Custom speaker ID')
@click.pass_context
def add_voice(ctx, name, audio_files, speaker_id):
    """Add a new voice profile
    
    NAME: Speaker name
    AUDIO_FILES: Path(s) to audio files
    """
    cloner = VoiceCloner(
        data_dir=ctx.obj['data_dir'],
        output_dir=ctx.obj['output_dir'],
        device=ctx.obj['device']
    )
    
    # Convert audio_files tuple to list
    audio_files = list(audio_files)
    
    try:
        assigned_id = cloner.add_voice_profile(name, audio_files, speaker_id)
        click.echo(f"Added voice profile: {name} (ID: {assigned_id})")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('speaker_id')
@click.pass_context
def analyze(ctx, speaker_id):
    """Analyze voice characteristics of a speaker"""
    cloner = VoiceCloner(
        data_dir=ctx.obj['data_dir'],
        output_dir=ctx.obj['output_dir'],
        device=ctx.obj['device']
    )
    
    try:
        analysis = cloner.analyze_voice(speaker_id)
        
        click.echo(f"Voice Analysis for {analysis['name']}")
        click.echo("-" * 50)
        click.echo(f"Duration: {analysis['duration']:.2f} seconds")
        click.echo(f"Sample Rate: {analysis['sample_rate']} Hz")
        click.echo(f"Fundamental Frequency: {analysis['f0_mean']:.1f} ± {analysis['f0_std']:.1f} Hz")
        click.echo(f"Spectral Centroid: {analysis['spectral_centroid']:.0f} Hz")
        click.echo(f"Spectral Rolloff: {analysis['spectral_rolloff']:.0f} Hz")
        click.echo(f"Tempo: {analysis['tempo']:.1f} BPM")
        click.echo(f"Energy: {analysis['energy_mean']:.4f}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('text')
@click.argument('speaker_id')
@click.option('--output', '-o', help='Output file path')
@click.option('--language', default='en', help='Language code')
@click.option('--speed', default=1.0, help='Speech speed multiplier')
@click.pass_context
def generate(ctx, text, speaker_id, output, language, speed):
    """Generate speech from text
    
    TEXT: Text to synthesize
    SPEAKER_ID: Speaker ID to use
    """
    cloner = VoiceCloner(
        data_dir=ctx.obj['data_dir'],
        output_dir=ctx.obj['output_dir'],
        device=ctx.obj['device']
    )
    
    config = GenerationConfig(language=language, speed=speed)
    
    try:
        output_path = cloner.generate_speech(text, speaker_id, output, config)
        click.echo(f"Generated speech: {output_path}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('text_file')
@click.argument('speaker_id')
@click.option('--output-dir', help='Output directory for batch files')
@click.option('--language', default='en', help='Language code')
@click.option('--speed', default=1.0, help='Speech speed multiplier')
@click.pass_context
def batch(ctx, text_file, speaker_id, output_dir, language, speed):
    """Generate speech for multiple texts from file
    
    TEXT_FILE: File containing texts (one per line)
    SPEAKER_ID: Speaker ID to use
    """
    cloner = VoiceCloner(
        data_dir=ctx.obj['data_dir'],
        output_dir=ctx.obj['output_dir'],
        device=ctx.obj['device']
    )
    
    # Read texts from file
    try:
        with open(text_file, 'r') as f:
            texts = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        click.echo(f"Text file not found: {text_file}", err=True)
        sys.exit(1)
    
    config = GenerationConfig(language=language, speed=speed)
    
    try:
        generated_files = cloner.batch_generate(texts, speaker_id, output_dir, config)
        click.echo(f"Generated {len(generated_files)} files")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('text')
@click.option('--speakers', help='Comma-separated speaker IDs (default: all)')
@click.option('--output-dir', help='Output directory')
@click.pass_context
def compare(ctx, text, speakers, output_dir):
    """Compare the same text across multiple speakers
    
    TEXT: Text to synthesize
    """
    cloner = VoiceCloner(
        data_dir=ctx.obj['data_dir'],
        output_dir=ctx.obj['output_dir'],
        device=ctx.obj['device']
    )
    
    speaker_ids = None
    if speakers:
        speaker_ids = [s.strip() for s in speakers.split(',')]
    
    try:
        results = cloner.compare_speakers(text, speaker_ids, output_dir)
        click.echo(f"Generated comparison with {len(results)} speakers")
        for speaker_id, file_path in results.items():
            click.echo(f"  {speaker_id}: {file_path}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--output', '-o', help='Output config file')
@click.pass_context
def save_config(ctx, output):
    """Save current voice profiles configuration"""
    cloner = VoiceCloner(
        data_dir=ctx.obj['data_dir'],
        output_dir=ctx.obj['output_dir'],
        device=ctx.obj['device']
    )
    
    try:
        cloner.save_profile_config(output)
        click.echo("Configuration saved")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('config_file')
@click.pass_context
def load_config(ctx, config_file):
    """Load voice profiles configuration from file"""
    cloner = VoiceCloner(
        data_dir=ctx.obj['data_dir'],
        output_dir=ctx.obj['output_dir'],
        device=ctx.obj['device']
    )
    
    try:
        cloner.load_profile_config(config_file)
        click.echo("Configuration loaded")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.group()
def demo():
    """Demo commands"""
    pass


@demo.command()
@click.pass_context
def quick(ctx):
    """Run a quick demo"""
    cloner = VoiceCloner(
        data_dir=ctx.obj['data_dir'],
        output_dir=ctx.obj['output_dir'],
        device=ctx.obj['device']
    )
    
    click.echo("Quick Voice Cloning Demo")
    click.echo("-" * 30)
    
    # List available voices
    cloner.list_voices()
    
    # If no voices available, show instructions
    if not cloner.voice_profiles:
        click.echo("To get started:")
        click.echo("1. Place audio files in data/raw/speaker_name/ directories")
        click.echo("2. Or use: python scripts/cli.py add-voice 'Speaker Name' audio1.wav audio2.wav")
        return
    
    # Generate sample with first available speaker
    speaker_id = list(cloner.voice_profiles.keys())[0]
    sample_text = "Hello! This is a demonstration of voice cloning technology."
    
    try:
        output_path = cloner.generate_speech(sample_text, speaker_id)
        click.echo(f"Demo generated: {output_path}")
    except Exception as e:
        click.echo(f"Demo failed: {e}", err=True)


@cli.command()
@click.argument('reference_audio')
@click.argument('speaker_id')
@click.option('--output', '-o', default=None, help='Output file path for generated audio')
@click.option('--asr-model', default='openai/whisper-small', help='ASR model for transcription')
@click.option('--language', default=None, help='Language code for ASR/generation (e.g., en)')
@click.option('--text', default=None, help='Bypass ASR and use this exact text for generation')
@click.pass_context
def match(ctx, reference_audio, speaker_id, output, asr_model, language, text):
    """Transcribe REFERENCE_AUDIO and regenerate with SPEAKER_ID, then score similarity.

    REFERENCE_AUDIO: Path to reference WAV/MP3 file used for transcription.
    """    # Build TTS cloner
    cloner = VoiceCloner(
        data_dir=ctx.obj['data_dir'],
        output_dir=ctx.obj['output_dir'],
        device=ctx.obj['device']
    )

    # If user provided text, skip ASR entirely
    if text is None:
        # Lazy imports to avoid heavy startup
        try:
            import torch
            from transformers import pipeline
        except Exception as e:
            click.echo(f"Missing dependencies for ASR: {e}. Try: pip install transformers torch", err=True)
            sys.exit(1)

        # Choose device for ASR
        use_cuda = torch.cuda.is_available() and (ctx.obj['device'] in ('auto', 'cuda'))
        asr_device = 0 if use_cuda else -1

        try:
            asr = pipeline('automatic-speech-recognition', model=asr_model, device=asr_device)
            # Transcribe
            result = asr(reference_audio)
            text = result['text'] if isinstance(result, dict) else result[0]['text']
            click.echo(f"Transcript: {text}")
        except Exception as e:
            click.echo(f"ASR failed: {e}", err=True)
            sys.exit(1)

    # Generate from text (either provided or transcribed)
    try:
        gen_cfg = GenerationConfig(language=language or 'en', speed=1.0)
        out_path = cloner.generate_speech(text, speaker_id, output, gen_cfg)
        click.echo(f"Generated: {out_path}")

        # Score similarity
        s = score_compare(reference_audio, out_path, target_sr=16000, align=True)
        click.echo(f"PESQ: {s['pesq']:.3f} (norm {s['pesq_norm']:.3f}), STOI: {s['stoi']:.3f}, Combined: {s['combined']:.3f}")
    except Exception as e:
        click.echo(f"Error during match/generate/score: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
