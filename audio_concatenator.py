import os
from pydub import AudioSegment
import re
import logging
from pathlib import Path
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join('logs', 'audio_concatenation.log'))
    ]
)
logger = logging.getLogger(__name__)

class AudioConcatenator:
    def __init__(self, input_folder: str):
        self.input_folder = Path(input_folder)
        if not self.input_folder.exists():
            raise ValueError(f"Input folder {input_folder} does not exist")

    def get_audio_files(self) -> List[Path]:
        """Get sorted list of WAV files from input folder"""
        wav_files = list(self.input_folder.glob("*.wav"))
        
        def get_index(filename: Path) -> int:
            numbers = re.findall(r'\d+', filename.stem)
            return int(numbers[0]) if numbers else 0
        
        return sorted(wav_files, key=get_index)

    def process_audio_file(self, file_path: Path) -> AudioSegment:
        """Process individual audio file with error handling"""
        try:
            audio = AudioSegment.from_wav(file_path)
            # Normalize audio to prevent volume variations
            return audio.normalize()
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            raise

    def add_transition(self, audio1: AudioSegment, audio2: AudioSegment, 
                      crossfade_duration: int = 50) -> AudioSegment:
        """Add smooth transition between audio segments"""
        return audio1.append(audio2, crossfade=crossfade_duration)

    def stitch_audio_files(self, output_filename: str = 'output.wav',
                          crossfade_duration: int = 50) -> None:
        """
        Concatenate all WAV files with smooth transitions
        
        Args:
            output_filename: Name of the output file
            crossfade_duration: Duration of crossfade in milliseconds
        """
        try:
            wav_files = self.get_audio_files()
            
            if not wav_files:
                logger.warning(f"No WAV files found in {self.input_folder}")
                return

            logger.info(f"Processing {len(wav_files)} WAV files")
            
            # Initialize with first audio file
            combined_audio = self.process_audio_file(wav_files[0])
            
            # Append remaining files with transitions
            for wav_file in wav_files[1:]:
                try:
                    audio_segment = self.process_audio_file(wav_file)
                    combined_audio = self.add_transition(
                        combined_audio, 
                        audio_segment,
                        crossfade_duration
                    )
                    logger.info(f"Added: {wav_file.name}")
                except Exception as e:
                    logger.error(f"Failed to process {wav_file.name}: {str(e)}")
                    continue

            # Apply final processing
            combined_audio = combined_audio.normalize()
            
            # Export with high quality settings
            output_path = self.input_folder / output_filename
            combined_audio.export(
                output_path,
                format="wav",
                parameters=[
                    "-q:a", "0",  # Highest quality
                    "-ar", "44100"  # Sample rate
                ]
            )
            
            duration_seconds = len(combined_audio) / 1000
            logger.info(f"Successfully created: {output_path}")
            logger.info(f"Final duration: {duration_seconds:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Audio concatenation failed: {str(e)}")
            raise

def main():
    """Main function to run the audio concatenator"""
    try:
        folder_path = "audio"
        concatenator = AudioConcatenator(folder_path)
        concatenator.stitch_audio_files("final_output.wav")
    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()