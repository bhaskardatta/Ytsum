#!/usr/bin/env python3

import os
import sys
import subprocess
import logging
from typing import Optional, List, Dict, Tuple
from pathlib import Path
import re
import webvtt
from urllib.parse import quote
import time

import pysrt
from gtts import gTTS
from transformers import pipeline  # Import the pipeline for local summarization
import nltk  # Import NLTK for tokenization
from sentence_transformers import SentenceTransformer, util  # For semantic similarity

from config import (
    API_URLS, 
    OUTPUT_DIR_STR, 
    TEMP_DIR_STR,
    AUDIO_DIR_STR,
    MAX_CHUNK_SIZE
)
from utils import query_api, chunk_text, safe_file_write, APIError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(OUTPUT_DIR_STR, 'youtube_summarizer.log'))
    ]
)
logger = logging.getLogger(__name__)

# Ensure NLTK Punkt tokenizer is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logger.info("Downloading NLTK Punkt tokenizer...")
    nltk.download('punkt', quiet=True)
except Exception as e:
    logger.error(f"Failed to download NLTK Punkt tokenizer: {str(e)}")
    sys.exit(1)

# Load a local summarization model
local_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Load a sentence transformer model for semantic similarity
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_important_points(vtt_path: str, video_url: str) -> List[Dict[str, str]]:
    """
    Extract important points from the VTT file using a rule-based approach.
    
    Args:
        vtt_path: Path to the VTT file.
        video_url: The YouTube video URL.
    
    Returns:
        List of dictionaries containing important points, timestamps, and clickable links.
    """
    important_points = []
    
    # Keywords and phrases to prioritize in the captions
    KEY_PHRASES = [
        "key point", "important", "takeaway", "summary", "explain", "example", "concept",
        "note", "remember", "focus", "highlight", "critical", "essential", "tip", "trick",
        "insight", "lesson", "principle", "theory", "practice", "method", "strategy",
        "technique", "approach", "solution", "problem", "question", "answer", "definition",
        "formula", "equation", "rule", "law", "hypothesis", "experiment", "result",
        "analysis", "conclusion", "recommendation", "advice"
    ]
    
    # Parse the VTT file
    try:
        captions = list(webvtt.read(vtt_path))
        logger.info(f"Successfully parsed VTT file: {vtt_path}")
    except Exception as e:
        logger.error(f"Failed to parse VTT file: {str(e)}")
        return important_points
    
    # If no captions are found, return an empty list
    if not captions:
        logger.warning("No captions found in the VTT file.")
        return important_points
    
    # Rule 1: Identify long captions (more than 50 characters)
    long_captions = [caption for caption in captions if len(caption.text) > 50]
    
    # Rule 2: Identify captions that contain questions
    question_captions = [caption for caption in captions if "?" in caption.text]
    
    # Rule 3: Identify captions that contain key phrases
    key_phrase_captions = [
        caption for caption in captions
        if any(phrase.lower() in caption.text.lower() for phrase in KEY_PHRASES)
    ]
    
    # Combine all matches and remove duplicates based on caption text
    seen_texts = set()
    matches = []
    for caption in long_captions + question_captions + key_phrase_captions:
        if caption.text not in seen_texts:
            matches.append(caption)
            seen_texts.add(caption.text)
    
    # If no matches are found, fall back to the top 5 longest captions
    if not matches:
        logger.warning("No matches found using rules. Falling back to top 5 longest captions.")
        matches = sorted(captions, key=lambda x: len(x.text), reverse=True)[:5]
    
    # Convert matches to important points
    for caption in matches:
        start_time = caption.start
        end_time = caption.end
        
        # Convert start time to seconds for YouTube URL
        try:
            # Split the timestamp into hours, minutes, seconds, and milliseconds
            hh_mm_ss, milliseconds = start_time.split(".")
            hh, mm, ss = hh_mm_ss.split(":")
            
            # Convert to total seconds
            start_seconds = int(hh) * 3600 + int(mm) * 60 + int(ss)
            logger.info(f"Converted start time to seconds: {start_seconds}")
        except Exception as e:
            logger.error(f"Failed to convert start time to seconds: {str(e)}")
            continue
        
        # Generate a clickable link to the exact timestamp
        clickable_link = f"{video_url}&t={start_seconds}s"
        logger.info(f"Generated clickable link: {clickable_link}")
        
        # Add the important point to the list
        important_points.append({
            "text": caption.text,
            "start_time": start_time,
            "end_time": end_time,
            "link": clickable_link
        })
        logger.info(f"Added important point: {caption.text}")
    
    # If no important points are found, add a placeholder message
    if not important_points:
        logger.warning("No important points found. Adding placeholder content.")
        important_points.append({
            "text": "No important points were identified in this video.",
            "start_time": "00:00:00.000",
            "end_time": "00:00:00.000",
            "link": video_url
        })
    
    return important_points

class YouTubeSummarizer:
    def __init__(self, youtube_url: str):
        self.youtube_url = youtube_url
        self.video_id = self._extract_video_id()
    
    def _extract_video_id(self) -> str:
        """Extract YouTube video ID from URL"""
        import re
        patterns = [
            r"v=([A-Za-z0-9_\-]{11})",
            r"youtu\.be/([A-Za-z0-9_\-]{11})",
            r"embed/([A-Za-z0-9_\-]{11})",
        ]
        for pattern in patterns:
            if match := re.search(pattern, self.youtube_url):
                return match.group(1)
        raise ValueError("Could not extract video ID from URL")

    def download_subtitles(self) -> str:
        """Download subtitles using yt-dlp"""
        output_pattern = os.path.join(TEMP_DIR_STR, "%(id)s.%(ext)s")
        command = [
            "yt-dlp",
            "--write-auto-sub",
            "--sub-lang", "en",
            "--sub-format", "srt",
            "--skip-download",
            "--output", output_pattern,
            self.youtube_url
        ]
        
        try:
            result = subprocess.run(
                command, 
                check=True, 
                capture_output=True, 
                text=True
            )
            logger.info(f"yt-dlp output: {result.stdout}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to download subtitles: {e.stderr}")
            raise
            
        # Check for subtitle files
        srt_path = os.path.join(TEMP_DIR_STR, f"{self.video_id}.en.srt")
        vtt_path = os.path.join(TEMP_DIR_STR, f"{self.video_id}.en.vtt")
        
        if os.path.isfile(srt_path):
            return srt_path
        elif os.path.isfile(vtt_path):
            return vtt_path
        else:
            raise FileNotFoundError("No subtitle files found after download")

    def convert_subtitles_to_text(self, subtitle_path: str) -> str:
        """Convert subtitle file to plain text"""
        ext = os.path.splitext(subtitle_path)[1].lower()
        text_lines = []
        
        try:
            if ext == ".srt":
                subs = pysrt.open(subtitle_path)
                text_lines = [sub.text.replace("\n", " ").strip() for sub in subs]
            elif ext == ".vtt":
                text_lines = [caption.text.replace("\n", " ").strip() 
                            for caption in webvtt.read(subtitle_path)]
            else:
                raise ValueError(f"Unsupported subtitle format: {ext}")
        except Exception as e:
            logger.error(f"Failed to process subtitles: {str(e)}")
            raise
            
        text_path = os.path.join(TEMP_DIR_STR, f"{self.video_id}_transcript.txt")
        content = " ".join(text_lines)
        safe_file_write(text_path, content)
        return text_path

    def summarize_text(self, text_path: str, max_length: int = 300) -> str:
        """Generate summary from text using the local model"""
        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read()
            
        chunks = chunk_text(text, MAX_CHUNK_SIZE)
        summaries = []
        
        for idx, chunk in enumerate(chunks, 1):
            logger.info(f"Summarizing chunk {idx}/{len(chunks)}")
            
            try:
                # Use the local summarization model
                summary = local_summarizer(chunk, max_length=max_length, min_length=30, do_sample=False)
                if isinstance(summary, list) and len(summary) > 0:
                    summaries.append(summary[0].get('summary_text', ''))
            except Exception as e:
                logger.error(f"Failed to summarize chunk {idx}: {str(e)}")
                continue
                
        summary_path = os.path.join(OUTPUT_DIR_STR, f"{self.video_id}_summary.txt")
        content = " ".join(summaries) if summaries else "Summarization failed."
        safe_file_write(summary_path, content)
        return summary_path

    def generate_podcast_script(self, summary_path: str) -> str:
        """Generate podcast script from summary"""
        with open(summary_path, "r", encoding="utf-8") as f:
            summary_text = f.read()
            
        prompt = (
            "Create a natural podcast dialogue between Alex and Chris discussing this topic. "
            "Format: [emotion] Speaker: dialogue\n"
            "Example:\n"
            "[excited] Alex: Hey everyone, welcome to the show!\n"
            "[friendly] Chris: Today we're discussing some amazing tech.\n\n"
            f"Topic to discuss:\n{summary_text}"
        )
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 1000,
                "temperature": 0.7
            }
        }
        
        try:
            response = query_api(API_URLS['PODCAST'], payload)
            if isinstance(response, list) and len(response) > 0:
                full_text = response[0].get('generated_text', '')
                
                # Extract only the dialogue part
                dialogue_lines = []
                for line in full_text.split('\n'):
                    # Only keep lines that start with [ and contain dialogue
                    if line.strip().startswith('[') and (':' in line or ']' in line):
                        dialogue_lines.append(line.strip())
                
                script = '\n'.join(dialogue_lines) if dialogue_lines else "Failed to generate dialogue."
            else:
                script = "Failed to generate podcast script."
        except APIError as e:
            logger.error(f"Failed to generate podcast script: {str(e)}")
            script = "Failed to generate podcast script."
            
        script_path = os.path.join(OUTPUT_DIR_STR, f"{self.video_id}_podcast.txt")
        safe_file_write(script_path, script)
        return script_path

    def get_important_points(self, vtt_path: str) -> List[Dict[str, str]]:
        """
        Get important points from the VTT file using a local summarization model.
        
        Args:
            vtt_path: Path to the VTT file.
        
        Returns:
            List of dictionaries containing important points, timestamps, and clickable links.
        """
        return extract_important_points(vtt_path, self.youtube_url)

    def generate_audio_from_script(self, script_path, output_path):
        """Generate audio from podcast script using gTTS."""
        logging.info("Starting audio generation...")
        
        # Create temp directory for audio segments
        temp_dir = os.path.join(os.path.dirname(script_path), "temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Read the script
        with open(script_path, 'r') as f:
            lines = f.readlines()
        
        audio_files = []
        file_list = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # Extract emotion and text
            parts = line.split(': ', 1)
            if len(parts) != 2:
                continue
                
            emotion, text = parts
            speaker = "Alex" if "Alex" in emotion else "Chris"
            
            # Clean text and split into sentences
            text = text.replace('..', '.').replace('!.', '!').replace('?.', '?')
            sentences = re.split('([.!?])', text)
            
            # Process each sentence
            for j in range(0, len(sentences)-1, 2):
                sentence = sentences[j].strip()
                punct = sentences[j+1] if j+1 < len(sentences) else ""
                
                if not sentence:
                    continue
                    
                full_sentence = sentence + punct
                logging.info(f"Generating audio for: {full_sentence[:30]}...")
                
                try:
                    # Configure TTS based on speaker and emotion
                    tld = 'com' if speaker == 'Alex' else 'co.uk'
                    rate = False if 'excited' in emotion.lower() else True
                    
                    # Generate audio
                    tts = gTTS(text=full_sentence, lang='en', tld=tld, slow=rate)
                    audio_file = os.path.join(temp_dir, f'segment_{i}_{j}.mp3')
                    tts.save(audio_file)
                    audio_files.append(audio_file)
                    
                    # Add appropriate silence
                    silence_file = os.path.join(temp_dir, f'silence_{i}_{j}.mp3')
                    duration = 0.7 if j == 0 else 0.3
                    self._generate_silence(silence_file, duration)
                    audio_files.append(silence_file)
                    
                    # Add to file list
                    file_list.extend([
                        f"file '{audio_file}'\n",
                        f"file '{silence_file}'\n"
                    ])
                    
                except Exception as e:
                    logging.error(f"Failed to generate audio for sentence: {full_sentence}. Error: {str(e)}")
                    continue
        
        # Write file list for ffmpeg
        files_txt = os.path.join(temp_dir, "files.txt")
        with open(files_txt, 'w') as f:
            f.writelines(file_list)
        
        try:
            # Concatenate all audio files
            subprocess.run([
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', files_txt,
                '-c:a', 'libmp3lame',
                '-q:a', '2',
                '-ar', '44100',
                output_path
            ], check=True)
            logging.info(f"Successfully generated audio file: {output_path}")
            
        except subprocess.CalledProcessError as e:
            logging.error(f"Processing failed: {e}")
            
        finally:
            # Cleanup temp files
            for file in audio_files:
                try:
                    os.remove(file)
                except:
                    pass
            try:
                os.remove(files_txt)
            except:
                pass
                
    def _generate_silence(self, output_file, duration):
        """Generate silence audio file."""
        subprocess.run([
            'ffmpeg', '-y',
            '-f', 'lavfi',
            '-i', f'anullsrc=r=44100:cl=stereo',
            '-t', str(duration),
            output_file
        ], check=True)

def main():
    """Main function to run the YouTube summarizer."""
    if len(sys.argv) != 2:
        print("Usage: python youtube_summarizer.py <youtube_url>")
        sys.exit(1)

    youtube_url = sys.argv[1]
    summarizer = YouTubeSummarizer(youtube_url)
    
    try:
        # Download and process subtitles
        subtitle_path = summarizer.download_subtitles()
        text_path = summarizer.convert_subtitles_to_text(subtitle_path)
        
        # Generate summary
        summary_path = summarizer.summarize_text(text_path)
        
        # Generate podcast script
        podcast_script_path = summarizer.generate_podcast_script(summary_path)
        
        # Generate audio
        audio_output_path = os.path.join(AUDIO_DIR_STR, f"{summarizer.video_id}_podcast.mp3")
        os.makedirs(AUDIO_DIR_STR, exist_ok=True)
        summarizer.generate_audio_from_script(podcast_script_path, audio_output_path)
        
        # Extract important points from the VTT file
        if subtitle_path.endswith(".vtt"):
            important_points = summarizer.get_important_points(subtitle_path)
            
            # Save important points to a file
            important_points_path = os.path.join(OUTPUT_DIR_STR, f"{summarizer.video_id}_important_points.txt")
            if important_points:
                with open(important_points_path, "w", encoding="utf-8") as f:
                    for point in important_points:
                        f.write(f"Time: {point['start_time']} - {point['end_time']}\n")
                        f.write(f"Text: {point['text']}\n")
                        f.write(f"Link: {point['link']}\n")
                        f.write("-" * 40 + "\n")
                logger.info(f"Important points saved to: {important_points_path}")
            else:
                logger.warning("No important points found to save.")
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()