import subprocess
import sys
from pathlib import Path
from typing import List
import logging

logger = logging.getLogger(__name__)

def run_script(script_name: str, args: List[str] = None) -> bool:
    try:
        cmd = ["python3", script_name]
        if args:
            cmd.extend(args)
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"{script_name} executed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running {script_name}: {e.stderr}")
        return False

def main():
    if len(sys.argv) != 2:
        logger.error("Usage: python integrate.py <youtube_url>")
        sys.exit(1)

    youtube_url = sys.argv[1]
    scripts = ["youtube.py", "extract_pod.py", "audio_concatenator.py"]
    
    for script in scripts:
        args = [youtube_url] if script == "youtube.py" else None
        if not run_script(script, args):
            sys.exit(1)

if __name__ == "__main__":
    main() 