#!/usr/bin/env python3
"""
Podcast Episode Extractor
Extract episodes with only Chris Benson and Daniel Whitenack from Practical AI feed
"""

import requests
import xml.etree.ElementTree as ET
import json
import re
import os
from pathlib import Path
from urllib.parse import urlparse
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PodcastExtractor:
    """Extract and analyze podcast episodes from RSS feed"""
    
    def __init__(self, feed_url: str, output_dir: str = "data/raw"):
        self.feed_url = feed_url
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Target hosts (different variations they might appear as)
        self.target_hosts = {
            'chris': ['chris benson', 'chris', 'benson'],
            'daniel': ['daniel whitenack', 'daniel', 'whitenack', 'dan whitenack', 'dan']
        }
        
        # Common guest indicators to filter out
        self.guest_indicators = [
            'guest', 'interview', 'special guest', 'joining us', 'with us today',
            'founder', 'ceo', 'cto', 'director', 'professor', 'researcher',
            'phd', 'dr.', 'author of', 'creator of', 'lead', 'senior',
            'engineer', 'scientist', 'developer'
        ]
    
    def fetch_feed(self) -> str:
        """Fetch RSS feed content"""
        logger.info(f"📡 Fetching RSS feed: {self.feed_url}")
        
        try:
            response = requests.get(self.feed_url, timeout=30)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.error(f"❌ Failed to fetch feed: {e}")
            raise
    
    def parse_feed(self, feed_content: str) -> dict:
        """Parse RSS feed XML"""
        logger.info("📝 Parsing RSS feed")
        
        try:
            root = ET.fromstring(feed_content)
            
            # Find channel
            channel = root.find('channel')
            if channel is None:
                raise ValueError("No channel found in RSS feed")
            
            # Extract podcast info
            podcast_info = {
                'title': self._get_text(channel, 'title'),
                'description': self._get_text(channel, 'description'),
                'language': self._get_text(channel, 'language'),
                'episodes': []
            }
            
            # Extract episodes
            for item in channel.findall('item'):
                episode = self._parse_episode(item)
                if episode:
                    podcast_info['episodes'].append(episode)
            
            logger.info(f"📊 Found {len(podcast_info['episodes'])} episodes")
            return podcast_info
            
        except ET.ParseError as e:
            logger.error(f"❌ Failed to parse XML: {e}")
            raise
    
    def _get_text(self, element, tag: str) -> str:
        """Safely get text from XML element"""
        child = element.find(tag)
        return child.text if child is not None and child.text else ""
    
    def _parse_episode(self, item) -> dict:
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
    
    def filter_host_only_episodes(self, episodes: list) -> list:
        """Filter episodes with only Chris Benson and Daniel Whitenack"""
        logger.info("🔍 Filtering episodes for host-only content")
        
        host_only_episodes = []
        
        for episode in episodes:
            if self._is_host_only_episode(episode):
                host_only_episodes.append(episode)
                logger.info(f"✅ Host-only episode: {episode['title']}")
            else:
                logger.debug(f"❌ Has guests: {episode['title']}")
        
        logger.info(f"🎯 Found {len(host_only_episodes)} host-only episodes")
        return host_only_episodes
    
    def _is_host_only_episode(self, episode: dict) -> bool:
        """Check if episode contains only the target hosts"""
        text_to_check = f"{episode['title']} {episode['description']}".lower()
        
        # Check for obvious guest indicators
        for indicator in self.guest_indicators:
            if indicator in text_to_check:
                return False
        
        # Check for common guest patterns
        guest_patterns = [
            r'with\s+[A-Z][a-z]+\s+[A-Z][a-z]+',  # "with John Smith"
            r'interview\s+with',
            r'joined\s+by',
            r'special\s+guest',
            r'talks\s+with',
            r'featuring\s+[A-Z][a-z]+',
        ]
        
        for pattern in guest_patterns:
            if re.search(pattern, text_to_check):
                return False
        
        # Look for host names to ensure it's a regular episode
        has_chris = any(name in text_to_check for name in self.target_hosts['chris'])
        has_daniel = any(name in text_to_check for name in self.target_hosts['daniel'])
        
        # If neither host is mentioned, it might be a special episode
        if not (has_chris or has_daniel):
            # Check if it's a typical discussion episode
            discussion_indicators = [
                'discuss', 'talk about', 'dive into', 'explore', 'look at',
                'practical ai', 'machine learning', 'deep learning', 'ai news'
            ]
            
            if not any(indicator in text_to_check for indicator in discussion_indicators):
                return False
        
        return True
    
    def download_episodes(self, episodes: list, max_episodes: int = None) -> list:
        """Download audio files for selected episodes"""
        if max_episodes:
            episodes = episodes[:max_episodes]
        
        logger.info(f"⬇️ Downloading {len(episodes)} episodes")
        
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
                logger.error(f"❌ Failed to download {episode['title']}: {e}")
                continue
        
        logger.info(f"✅ Downloaded {len(downloaded_files)} episodes")
        return downloaded_files
    
    def _download_episode_audio(self, episode: dict) -> str:
        """Download individual episode audio file"""
        if not episode['audio_url']:
            logger.warning("No audio URL found")
            return None
        
        # Create filename from title
        safe_title = re.sub(r'[^\w\s-]', '', episode['title'])
        safe_title = re.sub(r'[-\s]+', '-', safe_title)
        filename = f"{safe_title[:50]}.mp3"  # Limit filename length
        
        # Create episode directory
        episode_dir = self.output_dir / "practical_ai_episodes"
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
            logger.error(f"❌ Download failed: {e}")
            if file_path.exists():
                file_path.unlink()  # Remove partial file
            return None
    
    def save_episode_metadata(self, episodes: list, filename: str = "host_only_episodes.json"):
        """Save episode metadata to JSON file"""
        metadata_file = self.output_dir / filename
        
        with open(metadata_file, 'w') as f:
            json.dump(episodes, f, indent=2, default=str)
        
        logger.info(f"💾 Saved metadata: {metadata_file}")
    
    def extract_host_only_episodes(self, max_episodes: int = 20) -> dict:
        """Main method to extract host-only episodes"""
        logger.info("🚀 Starting host-only episode extraction")
        
        # Fetch and parse feed
        feed_content = self.fetch_feed()
        podcast_data = self.parse_feed(feed_content)
        
        # Filter for host-only episodes
        host_only_episodes = self.filter_host_only_episodes(podcast_data['episodes'])
        
        # Save metadata
        self.save_episode_metadata(host_only_episodes)
        
        # Download episodes
        downloaded_files = self.download_episodes(host_only_episodes, max_episodes)
        
        return {
            'podcast_info': {
                'title': podcast_data['title'],
                'total_episodes': len(podcast_data['episodes']),
                'host_only_episodes': len(host_only_episodes)
            },
            'episodes': host_only_episodes,
            'downloaded_files': downloaded_files
        }


def main():
    """Main function"""
    feed_url = "https://feeds.transistor.fm/practical-ai-machine-learning-data-science-llm"
    
    extractor = PodcastExtractor(feed_url)
    
    try:
        results = extractor.extract_host_only_episodes(max_episodes=10)  # Start with 10 episodes
        
        print("\n" + "="*60)
        print("📊 EXTRACTION SUMMARY")
        print("="*60)
        print(f"Podcast: {results['podcast_info']['title']}")
        print(f"Total Episodes: {results['podcast_info']['total_episodes']}")
        print(f"Host-Only Episodes Found: {results['podcast_info']['host_only_episodes']}")
        print(f"Downloaded: {len(results['downloaded_files'])}")
        print("\n✅ Extraction complete!")
        
        if results['downloaded_files']:
            print("\n📂 Downloaded Episodes:")
            for item in results['downloaded_files'][:5]:  # Show first 5
                print(f"• {item['episode']['title']}")
                print(f"  File: {item['file_path']}")
        
    except Exception as e:
        logger.error(f"❌ Extraction failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())