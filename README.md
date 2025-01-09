# Byte-Bandits

## Overview

Byte-Bandits is a collection of Python scripts designed for various tasks, including YouTube video processing, text summarization, and podcast integration. The project leverages advanced natural language processing models to extract, summarize, and generate audio content from YouTube videos.

## Features

- **YouTube Video Processing**: 
  - Search for YouTube videos and retrieve transcripts.
  - Extract important points from video subtitles.
  
- **Text Summarization**: 
  - Summarize video transcripts using Hugging Face models.
  - Generate concise summaries that capture key information.

- **Podcast Script Generation**: 
  - Convert summarized text into a structured podcast-style dialogue with emotional annotations.
  
- **Audio Generation**: 
  - Generate audio files from podcast scripts using Google Text-to-Speech (gTTS).
  - Support for different speakers and emotional tones.

- **Audio Processing**: 
  - Concatenate audio segments and manage temporary files efficiently.

## Contents

The repository includes the following scripts:

- **extract_pod.py**: Processes JSON files and generates audio.
- **integrate.sh**: Shell script for integrating different components.
- **parl_gen.py**: Generates audio tokens from the JSON file.
- **podcast_script.txt**: Text file containing podcast scripts.
- **pytube_search.py**: Searches YouTube videos using the PyTube library.
- **search.py**: Performs search operations.
- **text_summarization.py**: Summarizes text content generated from the video text.
- **youtube_summarizer.py**: Summarizes YouTube video content and generates audio.
- **youtube_transcript.py**: Retrieves transcripts from YouTube videos.
- **config.py**: Configuration file for API tokens and directory paths.
- **setup_check.sh**: Script to set up the environment and install dependencies.

## Requirements

Ensure you have Python installed. You may need to install additional packages using `pip`:
```bash
pip install -r requirements.txt
```
### Dependencies

The project requires the following Python packages:

- Core Dependencies:
  - torch>=2.0.0
  - transformers>=4.30.0
  - python-dotenv>=1.0.0
  - numpy>=1.24.3

- Audio Processing:
  - soundfile>=0.12.1
  - pydub>=0.25.1
  - gTTS>=2.5.1

- API and Data Processing:
  - requests>=2.31.0
  - pysrt>=1.1.2
  - webvtt-py>=0.4.6
  - langdetect>=1.0.9

- YouTube Integration:
  - youtube-search-python>=1.6.6
  - youtube-transcript-api>=0.6.1
  - yt-dlp>=2023.11.16

- Logging and Development:
  - logging>=0.4.9.6
  - pytest>=7.4.3
  - black>=23.11.0
  - pylint>=3.0.2

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/Byte-Bandits.git
   cd Byte-Bandits
   ```

2. **Run the Setup Script**:
   ```bash
   bash setup_check.sh
   ```

3. **Add Your Hugging Face API Token**:
   - Open the `.env` file created by the setup script and add your Hugging Face API token:
     ```
     HUGGINGFACE_API_TOKEN=your_token_here
     ```

4. **Create Necessary Directories**:
   The setup script will automatically create the required directories (`output`, `temp`, `audio`, `logs`).

## Usage

To summarize a YouTube video and generate a podcast script, run the following command:
```bash
python3 youtube_summarizer.py <youtube_url>
```
Replace `<youtube_url>` with the actual URL of the YouTube video you want to process.

## Logging

Logs are generated in the `logs` directory. You can check the logs for any errors or information about the processing steps.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for providing powerful NLP models.
- [gTTS](https://gtts.readthedocs.io/en/latest/) for text-to-speech conversion.
- [YouTube Transcript API](https://github.com/jdepoix/youtube-transcript-api) for retrieving video transcripts.
