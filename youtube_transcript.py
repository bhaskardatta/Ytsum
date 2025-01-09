from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import SRTFormatter
import yt_dlp
import os
import re

def get_video_id(youtube_url):
    """
    Extracts the video ID from the given YouTube URL.
    """
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", youtube_url)
    if match:
        return match.group(1)
    else:
        raise ValueError("Invalid YouTube URL format. Unable to extract video ID.")

def fetch_video_title(youtube_url):
    """
    Fetch the video title using yt-dlp.
    """
    try:
        ydl_opts = {'quiet': True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            return info.get("title", "Unknown_Title").replace(" ", "_").replace("|", "_")
    except Exception as e:
        return f"Error_Fetching_Title_{e}"

def fetch_transcripts(youtube_url, target_language='en', save_path="output"):
    """
    Fetches auto-generated transcripts from a YouTube video and optionally translates them.
    """
    try:
        video_id = get_video_id(youtube_url)
        
        video_title = fetch_video_title(youtube_url)
        print(f"Video Title: {video_title}")

        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        
        if target_language != 'en':
            transcript = YouTubeTranscriptApi.translate_transcript(transcript, target_language)

        formatter = SRTFormatter()
        srt_transcript = formatter.format_transcript(transcript)

        os.makedirs(save_path, exist_ok=True)  
        save_file = os.path.join(save_path, f"{video_title}_{target_language}.srt")
        with open(save_file, "w", encoding="utf-8") as file:
            file.write(srt_transcript)
        
        return f"Transcript saved successfully at: {save_file}"
    
    except Exception as e:
        return f"An error occurred: {e}"