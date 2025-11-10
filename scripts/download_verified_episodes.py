#!/usr/bin/env python3
"""
Download Verified Host-Only Episodes
Downloads specific episodes that have been manually verified to contain only Chris and Daniel
"""

import requests
import xml.etree.ElementTree as ET
import json
import re
import os
from pathlib import Path
from urllib.parse import urlparse
import logging
from typing import List, Dict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VerifiedEpisodeDownloader:
    """Download specific verified host-only episodes"""
    
    def __init__(self, feed_url: str, output_dir: str = "data/raw"):
        self.feed_url = feed_url
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Manually verified episode titles (only Chris and Daniel)
        self.verified_titles = {
            "Cracking the code of failed AI pilots",
            "Inside America's AI Action Plan", 
            "Workforce dynamics in an AI-assisted world",
            "Agentic AI for Drone & Robotic Swarming",
            "AI in the shadows: From hallucinations to blackmail",
            "AI hot takes and debates: Autonomy",
            "Behind-the-Scenes: VC Funding for AI Startups",
            "Model Context Protocol Deep Dive",
            "GenAI hot takes and bad use cases",
            "Tool calling and agents",
            "Deep-dive into DeepSeek",
            "Clones, commerce & campaigns",
            "Creating tested, reliable AI applications",
            "Pausing to think about scikit-learn & OpenAI o1",
            "Only as good as the data",
            "The first real-time voice assistant",
            "Apple Intelligence & Advanced RAG",
            "Rise of the AI PC & local LLMs",
            "First impressions of GPT-4o",
            "Autonomous fighter jets?!",
            "Udio & the age of multi-modal AI",
            "Should kids still learn to code?",
            "YOLOv9: Computer vision is alive and well",
            "Representation Engineering (Activation Hacking)",
            "Gemini vs OpenAI",
            "Large Action Models (LAMs) & Rabbits 🐇",
            "AI predictions for 2024",
            "The OpenAI debacle (a retrospective)",
            "Government regulation of AI has arrived",
            "Generative models: exploration to deployment"
        }
        
        logger.info(f"Initialized with {len(self.verified_titles)} verified episode titles")
    
    def fetch_feed(self) -> str:
        """Fetch RSS feed content"""
        logger.info(f"Fetching RSS feed: {self.feed_url}")
        
        try:
            response = requests.get(self.feed_url, timeout=30)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.error(f" Failed to fetch feed: {e}")
            raise
    
    def parse_feed(self, feed_content: str) -> Dict:
        """Parse RSS feed XML and extract episodes"""
        logger.info("Parsing RSS feed")
        
        try:
            root = ET.fromstring(feed_content)
            channel = root.find('channel')
            
            if channel is None:
                raise ValueError("No channel found in RSS feed")
            
            episodes = []
            for item in channel.findall('item'):
                episode = self._parse_episode(item)
                if episode:
                    episodes.append(episode)
            
            logger.info(f"Found {len(episodes)} total episodes in feed")
            return {
                'title': self._get_text(channel, 'title'),
                'episodes': episodes
            }
            
        except ET.ParseError as e:
            logger.error(f"Failed to parse XML: {e}")
            raise
    
    def _get_text(self, element, tag: str) -> str:
        """Safely get text from XML element"""
        child = element.find(tag)
        return child.text if child is not None and child.text else ""
    
    def _parse_episode(self, item) -> Dict:
        """Parse individual episode from XML item"""
        episode = {
            'title': self._get_text(item, 'title'),
            'description': self._get_text(item, 'description'),
            'pub_date': self._get_text(item, 'pubDate'),
            'duration': '',
            'audio_url': '',
            'file_size': 0
        }
        
        # Get enclosure (audio file)
        enclosure = item.find('enclosure')
        if enclosure is not None:
            episode['audio_url'] = enclosure.get('url', '')
            episode['file_size'] = int(enclosure.get('length', 0))
        
        # Get duration from iTunes tags
        itunes_duration = item.find('.//{http://www.itunes.com/dtds/podcast-1.0.dtd}duration')
        if itunes_duration is not None:
            episode['duration'] = itunes_duration.text
        
        return episode
    
    def find_verified_episodes(self, all_episodes: List[Dict]) -> List[Dict]:
        """Find episodes that match our verified titles"""
        logger.info("🔍 Searching for verified episodes")
        
        verified_episodes = []
        found_titles = set()
        
        for episode in all_episodes:
            title = episode['title']
            
            # Check for exact match or close match (handle emoji and punctuation differences)
            normalized_title = self._normalize_title(title)
            
            for verified_title in self.verified_titles:
                normalized_verified = self._normalize_title(verified_title)
                
                if normalized_title == normalized_verified or title == verified_title:
                    verified_episodes.append(episode)
                    found_titles.add(verified_title)
                    logger.info(f"Found: {title}")
                    break
        
        # Report missing episodes
        missing_titles = self.verified_titles - found_titles
        if missing_titles:
            logger.warning(f"Could not find {len(missing_titles)} episodes:")
            for title in sorted(missing_titles):
                logger.warning(f"   • {title}")
        
        logger.info(f"Found {len(verified_episodes)}/{len(self.verified_titles)} verified episodes")
        return verified_episodes
    
    def _normalize_title(self, title: str) -> str:
        """Normalize title for comparison (remove special chars, normalize spacing)"""
        # Remove emojis and special characters, normalize whitespace
        normalized = re.sub(r'[^\w\s-]', '', title)
        normalized = re.sub(r'\s+', ' ', normalized).strip().lower()
        return normalized
    
    def download_episodes(self, episodes: List[Dict], max_episodes: int = None) -> List[Dict]:
        """Download audio files for verified episodes"""
        if max_episodes:
            episodes = episodes[:max_episodes]
        
        logger.info(f"Downloading {len(episodes)} verified episodes")
        
        downloaded_files = []
        
        for i, episode in enumerate(episodes, 1):
            logger.info(f"[{i}/{len(episodes)}] Downloading: {episode['title']}")
            
            try:
                file_path = self._download_episode_audio(episode)
                if file_path:
                    downloaded_files.append({
                        'episode': episode,
                        'file_path': file_path
                    })
            except Exception as e:
                logger.error(f"Failed to download {episode['title']}: {e}")
                continue
        
        logger.info(f"Successfully downloaded {len(downloaded_files)} episodes")
        return downloaded_files
    
    def _download_episode_audio(self, episode: Dict) -> str:
        """Download individual episode audio file"""
        if not episode['audio_url']:
            logger.warning("No audio URL found")
            return None
        
        # Create safe filename from title
        safe_title = re.sub(r'[^\w\s-]', '', episode['title'])
        safe_title = re.sub(r'[-\s]+', '-', safe_title)
        filename = f"{safe_title[:60]}.mp3"
        
        # Create verified episodes directory
        episode_dir = self.output_dir / "verified_host_only_episodes"
        episode_dir.mkdir(exist_ok=True)
        
        file_path = episode_dir / filename
        
        # Skip if already downloaded
        if file_path.exists():
            logger.info(f"⚡ Already exists: {filename}")
            return str(file_path)
        
        # Download file
        try:
            response = requests.get(episode['audio_url'], stream=True, timeout=60)
            response.raise_for_status()
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            logger.info(f"💾 Downloaded: {filename}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Download failed: {e}")
            if file_path.exists():
                file_path.unlink()
            return None
    
    def save_verified_metadata(self, episodes: List[Dict], filename: str = "verified_host_only_episodes.json"):
        """Save verified episode metadata"""
        metadata_file = self.output_dir / filename
        
        with open(metadata_file, 'w') as f:
            json.dump(episodes, f, indent=2, default=str)
        
        logger.info(f"Saved metadata: {metadata_file}")
    
    def download_verified_episodes(self, max_episodes: int = None) -> Dict:
        """Main method to download all verified host-only episodes"""
        logger.info("Starting verified episode download")
        
        # Fetch and parse feed
        feed_content = self.fetch_feed()
        podcast_data = self.parse_feed(feed_content)
        
        # Find verified episodes
        verified_episodes = self.find_verified_episodes(podcast_data['episodes'])
        
        # Save metadata
        self.save_verified_metadata(verified_episodes)
        
        # Download episodes
        downloaded_files = self.download_episodes(verified_episodes, max_episodes)
        
        return {
            'podcast_info': {
                'title': podcast_data['title'],
                'total_episodes_in_feed': len(podcast_data['episodes']),
                'verified_episodes_found': len(verified_episodes),
                'verified_episodes_target': len(self.verified_titles)
            },
            'episodes': verified_episodes,
            'downloaded_files': downloaded_files
        }


def main():
    """Main function"""
    feed_url = "https://feeds.transistor.fm/practical-ai-machine-learning-data-science-llm"
    
    downloader = VerifiedEpisodeDownloader(feed_url)
    
    try:
        # Download all verified episodes (no limit)
        results = downloader.download_verified_episodes()
        
        print("\n" + "="*70)
        print("VERIFIED EPISODE DOWNLOAD SUMMARY")
        print("="*70)
        print(f"Podcast: {results['podcast_info']['title']}")
        print(f"Total Episodes in Feed: {results['podcast_info']['total_episodes_in_feed']}")
        print(f"Verified Episodes Target: {results['podcast_info']['verified_episodes_target']}")
        print(f"Verified Episodes Found: {results['podcast_info']['verified_episodes_found']}")
        print(f"Successfully Downloaded: {len(results['downloaded_files'])}")
        
        success_rate = len(results['downloaded_files']) / results['podcast_info']['verified_episodes_target'] * 100
        print(f"Download Success Rate: {success_rate:.1f}%")
        
        print("\nDownload complete!")
        
        if results['downloaded_files']:
            print("\n📂 Downloaded Episodes (first 10):")
            for item in results['downloaded_files'][:10]:
                episode_title = item['episode']['title']
                file_size = item['episode'].get('file_size', 0) / 1024 / 1024
                print(f"• {episode_title}")
                print(f"  File: {Path(item['file_path']).name} ({file_size:.1f} MB)")
        
        print(f"\nThese episodes contain only Chris Benson and Daniel Whitenack!")
        print("Perfect for voice cloning training data.")
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
