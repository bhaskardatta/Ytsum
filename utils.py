import requests
import time
import logging
from typing import Dict, Any, List
from config import HEADERS, MAX_RETRIES, RETRY_INTERVAL
from transformers import pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class APIError(Exception):
    """Custom exception for API-related errors"""
    pass

# Load local summarization model as a fallback
local_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def query_api(url: str, payload: Dict[str, Any], max_retries: int = MAX_RETRIES) -> Any:
    """
    Make an API request with retry logic and proper error handling
    
    Args:
        url: API endpoint URL
        payload: Request payload
        max_retries: Maximum number of retry attempts
    
    Returns:
        API response data
    
    Raises:
        APIError: If the API request fails after all retries
    """
    for attempt in range(max_retries):
        try:
            # Simplify the payload to just the text for summarization and text generation
            if 'inputs' in payload and isinstance(payload['inputs'], str):
                data = {"inputs": payload['inputs']}
            else:
                data = payload

            response = requests.post(
                url,
                headers=HEADERS,
                json=data,
                timeout=10  # Reduced timeout to 10 seconds
            )
            
            if response.status_code == 429:  # Rate limit
                wait_time = int(response.headers.get('Retry-After', 5))  # Reduced wait time to 5 seconds
                logger.warning(f"Rate limit hit. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            
            if response.status_code == 503:  # Model loading
                logger.warning(f"Model is loading. Waiting 5 seconds...")  # Reduced wait time to 5 seconds
                time.sleep(5)
                continue

            if response.status_code == 500:  # Internal Server Error
                logger.warning(f"Internal Server Error. Retrying in 5 seconds...")  # Reduced wait time to 5 seconds
                time.sleep(5)
                continue
                
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.Timeout:
            logger.warning(f"Request timed out (attempt {attempt + 1}/{max_retries})")
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:  # Don't wait on the last attempt
                time.sleep(5)  # Reduced wait time to 5 seconds
            logger.error(f"Request failed: {str(e)}")
            if attempt == max_retries - 1:
                raise APIError(f"API request failed after {max_retries} attempts: {str(e)}")
    
    raise APIError(f"API request failed after {max_retries} attempts")

def summarize_text_locally(text: str, max_length: int = 300) -> str:
    """
    Summarize text using a local model as a fallback.
    
    Args:
        text: Input text to summarize
        max_length: Maximum length of the summary
    
    Returns:
        Summary text
    """
    try:
        summary = local_summarizer(text, max_length=max_length, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        logger.error(f"Local summarization failed: {str(e)}")
        raise

def chunk_text(text: str, max_chunk_size: int) -> List[str]:
    """
    Split text into chunks of specified maximum size
    
    Args:
        text: Input text to chunk
        max_chunk_size: Maximum number of words per chunk
    
    Returns:
        List of text chunks
    """
    words = text.split()
    return [" ".join(words[i:i + max_chunk_size]) 
            for i in range(0, len(words), max_chunk_size)]

def safe_file_write(file_path: str, content: str) -> None:
    """
    Safely write content to a file with proper error handling
    
    Args:
        file_path: Path to the output file
        content: Content to write
    """
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"Successfully wrote to {file_path}")
    except IOError as e:
        logger.error(f"Failed to write to {file_path}: {str(e)}")
        raise