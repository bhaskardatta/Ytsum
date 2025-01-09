from dotenv import load_dotenv
import os
from pathlib import Path

# Base directory (project root)
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv(BASE_DIR / '.env')

# API Configuration
HUGGINGFACE_API_TOKEN = os.getenv('HUGGINGFACE_API_TOKEN')
if not HUGGINGFACE_API_TOKEN:
    raise ValueError("HUGGINGFACE_API_TOKEN environment variable is not set")

# API URLs
API_URLS = {
    'SUMMARIZE': "https://api-inference.huggingface.co/models/facebook/bart-large-cnn",
    'TRANSLATE': "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-mul-en",
    'PODCAST': "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-Coder-32B-Instruct"
}

# API Headers
HEADERS = {
    "Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"
}

# Chunk sizes and retry settings
MAX_CHUNK_SIZE = 700
MAX_RETRIES = 5
RETRY_INTERVAL = 20

# Directory structure
OUTPUT_DIR = BASE_DIR / "output"
TEMP_DIR = BASE_DIR / "temp"
AUDIO_DIR = BASE_DIR / "audio"

# Create necessary directories
for directory in [OUTPUT_DIR, TEMP_DIR, AUDIO_DIR]:
    directory.mkdir(exist_ok=True)
    
# Export paths as strings for compatibility
OUTPUT_DIR_STR = str(OUTPUT_DIR)
TEMP_DIR_STR = str(TEMP_DIR)
AUDIO_DIR_STR = str(AUDIO_DIR) 