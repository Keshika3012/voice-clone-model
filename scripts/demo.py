#!/usr/bin/env python3
"""
Voice Cloning Demo
Showcase the voice cloning system capabilities
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from voice_cloner import VoiceCloner, GenerationConfig
import time


class VoiceCloningDemo:
    """Comprehensive demo of voice cloning capabilities"""
    
    def __init__(self):
        print("🎭 Voice Cloning Demo")
        print("=" * 50)
        
        # Initialize voice cloner
        self.cloner = VoiceCloner()
        
        # Demo texts
        self.demo_texts = {
            'greeting': "Hello! Welcome to our advanced voice cloning demonstration.",
            'technical': "This artificial intelligence system can synthesize human speech with remarkable accuracy and naturalness.",
            'conversational': "I hope you're enjoying this demonstration. The technology behind voice cloning has advanced significantly.",
            'long_form': "Voice cloning represents one of the most exciting developments in artificial intelligence and machine learning. By analyzing the unique characteristics of a person's voice, we can create synthetic speech that captures their vocal patterns, intonation, and speaking style."
        }
    
    def run_basic_demo(self):
        """Run basic voice generation demo"""
        print("\n BASIC VOICE GENERATION")
        print("-" * 30)
        
        # List available voices
        self.cloner.list_voices()
        
        if not self.cloner.voice_profiles:
            print("No voice profiles available. Please add some voice data first.")
            print("Try: python scripts/extract_podcast_episodes.py")
            return
        
        # Generate samples for each available speaker
        for speaker_id in self.cloner.voice_profiles.keys():
            print(f"\n🎙️ Generating demo for {speaker_id.upper()}...")
            
            output_path = f"outputs/demos/{speaker_id}_greeting.wav"
            try:
                generated_file = self.cloner.generate_speech(
                    self.demo_texts['greeting'],
                    speaker_id,
                    output_path
                )
                print(f"✅ Generated: {generated_file}")
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def run_speed_demo(self):
        """Demonstrate speech speed control"""
        print("\n⚡ SPEECH SPEED CONTROL")
        print("-" * 30)
        
        if not self.cloner.voice_profiles:
            print("❌ No voice profiles available")
            return
        
        speaker_id = list(self.cloner.voice_profiles.keys())[0]
        speeds = [0.8, 1.0, 1.2, 1.5]
        
        print(f"🎭 Using speaker: {speaker_id}")
        
        for speed in speeds:
            print(f"🎚️ Generating at {speed}x speed...")
            
            config = GenerationConfig(speed=speed)
            output_path = f"outputs/demos/{speaker_id}_speed_{speed}x.wav"
            
            try:
                generated_file = self.cloner.generate_speech(
                    self.demo_texts['conversational'],
                    speaker_id,
                    output_path,
                    config
                )
                print(f"✅ Generated: {generated_file}")
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def run_comparison_demo(self):
        """Demonstrate speaker comparison"""
        print("\n🆚 SPEAKER COMPARISON")
        print("-" * 30)
        
        if len(self.cloner.voice_profiles) < 2:
            print("❌ Need at least 2 speakers for comparison")
            return
        
        text = self.demo_texts['technical']
        print(f"📝 Text: '{text}'")
        
        try:
            results = self.cloner.compare_speakers(text)
            print(f"✅ Generated comparison with {len(results)} speakers:")
            for speaker_id, file_path in results.items():
                print(f"  • {speaker_id}: {file_path}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def run_batch_demo(self):
        """Demonstrate batch generation"""
        print("\n📦 BATCH GENERATION")
        print("-" * 30)
        
        if not self.cloner.voice_profiles:
            print("❌ No voice profiles available")
            return
        
        speaker_id = list(self.cloner.voice_profiles.keys())[0]
        
        batch_texts = [
            "This is the first sample in our batch demonstration.",
            "Here's the second sample showing consistent voice quality.",
            "The third sample demonstrates batch processing efficiency.",
            "Finally, this fourth sample completes our batch demo."
        ]
        
        print(f"🎭 Generating batch for: {speaker_id}")
        
        try:
            generated_files = self.cloner.batch_generate(
                batch_texts, 
                speaker_id,
                "outputs/demos/batch_demo"
            )
            print(f"✅ Generated {len(generated_files)} batch files")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def run_analysis_demo(self):
        """Demonstrate voice analysis"""
        print("\n🔬 VOICE ANALYSIS")
        print("-" * 30)
        
        for speaker_id in self.cloner.voice_profiles.keys():
            print(f"\n🎭 Analyzing: {speaker_id.upper()}")
            
            try:
                analysis = self.cloner.analyze_voice(speaker_id)
                
                print(f"📊 Voice Characteristics:")
                print(f"  • Duration: {analysis['duration']:.2f} seconds")
                print(f"  • Pitch (F0): {analysis['f0_mean']:.1f} ± {analysis['f0_std']:.1f} Hz")
                print(f"  • Spectral Centroid: {analysis['spectral_centroid']:.0f} Hz")
                print(f"  • Energy: {analysis['energy_mean']:.4f}")
                
                # Voice type classification
                if analysis['f0_mean'] < 120:
                    voice_type = "Deep/Bass voice"
                elif analysis['f0_mean'] < 180:
                    voice_type = "Medium pitch voice"
                else:
                    voice_type = "High pitch voice"
                
                print(f"  • Voice Type: {voice_type}")
                
            except Exception as e:
                print(f"❌ Analysis failed: {e}")
    
    def run_interactive_demo(self):
        """Interactive demo where user can input custom text"""
        print("\n💬 INTERACTIVE DEMO")
        print("-" * 30)
        
        if not self.cloner.voice_profiles:
            print("❌ No voice profiles available")
            return
        
        # Show available speakers
        print("Available speakers:")
        for i, (speaker_id, profile) in enumerate(self.cloner.voice_profiles.items(), 1):
            print(f"  {i}. {profile.name} (ID: {speaker_id})")
        
        # Get user input
        try:
            choice = int(input("\nSelect speaker (number): ")) - 1
            speaker_ids = list(self.cloner.voice_profiles.keys())
            
            if 0 <= choice < len(speaker_ids):
                selected_speaker = speaker_ids[choice]
                print(f"Selected: {self.cloner.voice_profiles[selected_speaker].name}")
                
                # Get custom text
                custom_text = input("\nEnter text to synthesize: ").strip()
                
                if custom_text:
                    print(f"\n🎤 Generating speech...")
                    
                    output_path = f"outputs/demos/interactive_{selected_speaker}_{int(time.time())}.wav"
                    
                    generated_file = self.cloner.generate_speech(
                        custom_text,
                        selected_speaker,
                        output_path
                    )
                    
                    print(f"✅ Generated: {generated_file}")
                    print("🎵 You can now play the audio file!")
                else:
                    print("❌ No text provided")
            else:
                print("❌ Invalid selection")
                
        except (ValueError, KeyboardInterrupt):
            print("\n👋 Demo cancelled")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def run_full_demo(self):
        """Run complete demonstration"""
        print("🚀 Running full voice cloning demonstration...\n")
        
        try:
            self.run_basic_demo()
            
            if len(self.cloner.voice_profiles) >= 1:
                self.run_speed_demo()
                self.run_analysis_demo()
                self.run_batch_demo()
            
            if len(self.cloner.voice_profiles) >= 2:
                self.run_comparison_demo()
            
            print("\n" + "=" * 50)
            print("✅ DEMO COMPLETE!")
            print("=" * 50)
            print("📂 Check the outputs/demos/ directory for generated files")
            
        except Exception as e:
            print(f"❌ Demo failed: {e}")


def show_menu():
    """Show demo menu"""
    print("\n🎭 Voice Cloning Demo Menu")
    print("=" * 30)
    print("1. Basic Demo - Generate simple samples")
    print("2. Speed Demo - Test different speech speeds")
    print("3. Comparison Demo - Compare multiple speakers")
    print("4. Batch Demo - Generate multiple samples")
    print("5. Analysis Demo - Analyze voice characteristics")
    print("6. Interactive Demo - Custom text input")
    print("7. Full Demo - Run all demonstrations")
    print("8. Exit")
    print("=" * 30)


def main():
    """Main function"""
    demo = VoiceCloningDemo()
    
    while True:
        show_menu()
        
        try:
            choice = input("\nSelect option (1-8): ").strip()
            
            if choice == '1':
                demo.run_basic_demo()
            elif choice == '2':
                demo.run_speed_demo()
            elif choice == '3':
                demo.run_comparison_demo()
            elif choice == '4':
                demo.run_batch_demo()
            elif choice == '5':
                demo.run_analysis_demo()
            elif choice == '6':
                demo.run_interactive_demo()
            elif choice == '7':
                demo.run_full_demo()
            elif choice == '8':
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please select 1-8.")
                
            input("\nPress Enter to continue...")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()