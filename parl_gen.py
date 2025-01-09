import torch
from transformers import AutoTokenizer
import soundfile as sf
from typing import Dict, List, Tuple
import logging
from pathlib import Path
import numpy as np
from pydub import AudioSegment
import io
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join('logs', 'audio_generation.log'))
    ]
)
logger = logging.getLogger(__name__)

class AudioGenerator:
    def __init__(self, model_name: str = "parler-tts/parler-tts-mini-multilingual-v1.1"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        self.model = self._load_model(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.description_tokenizer = AutoTokenizer.from_pretrained(
            self.model.config.text_encoder._name_or_path
        )
        
        # Cache for speaker embeddings
        self._speaker_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        
    def _load_model(self, model_name: str):
        try:
            from parler_tts import ParlerTTSForConditionalGeneration
            model = ParlerTTSForConditionalGeneration.from_pretrained(
                model_name, 
                attn_implementation="eager"
            ).to(self.device)
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def get_speaker_embedding(self, speaker: str, tone: str = "neutral") -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cached speaker embedding or generate new one"""
        cache_key = f"{speaker}_{tone}"
        if cache_key not in self._speaker_cache:
            self._speaker_cache[cache_key] = self._generate_speaker_embedding(speaker, tone)
        return self._speaker_cache[cache_key]

    def _generate_speaker_embedding(self, speaker: str, tone: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate speaker embedding with emotional tone"""
        description = (
            f"A {speaker} delivers speech in a {tone} tone with natural rhythm and clear articulation. "
            f"The voice has good dynamic range and maintains consistent quality throughout."
        )
        
        inputs = self.description_tokenizer(description, return_tensors="pt")
        return (
            inputs.input_ids.to(self.device),
            inputs.attention_mask.to(self.device)
        )

    def generate_audio(
        self, 
        text: str, 
        speaker: str,
        tone: str = "neutral",
        output_file: str = None,
        apply_effects: bool = True
    ) -> np.ndarray:
        """Generate audio with improved quality and effects"""
        try:
            # Get speaker embedding
            input_ids, attention_mask = self.get_speaker_embedding(speaker, tone)
            
            # Prepare text input
            prompt_inputs = self.tokenizer(text, return_tensors="pt")
            prompt_input_ids = prompt_inputs.input_ids.to(self.device)
            prompt_attention_mask = prompt_inputs.attention_mask.to(self.device)

            # Generate audio
            with torch.no_grad():
                generation = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    prompt_input_ids=prompt_input_ids,
                    prompt_attention_mask=prompt_attention_mask,
                    do_sample=True,
                    temperature=0.7,
                    max_length=500
                )

            # Convert to numpy array
            audio_arr = generation.cpu().numpy().squeeze()

            if apply_effects:
                audio_arr = self._apply_audio_effects(audio_arr)

            if output_file:
                sf.write(output_file, audio_arr, self.model.config.sampling_rate)

            return audio_arr

        except Exception as e:
            logger.error(f"Failed to generate audio: {str(e)}")
            raise

    def _apply_audio_effects(self, audio_arr: np.ndarray) -> np.ndarray:
        """Apply audio enhancement effects"""
        try:
            # Convert to AudioSegment for processing
            audio_segment = AudioSegment(
                audio_arr.tobytes(),
                frame_rate=self.model.config.sampling_rate,
                sample_width=2,
                channels=1
            )

            # Apply subtle compression
            audio_segment = audio_segment.compress_dynamic_range()
            
            # Add very slight reverb
            audio_segment = audio_segment.overlay(
                audio_segment.fade_out(50),
                position=10
            )

            # Convert back to numpy array
            buffer = io.BytesIO()
            audio_segment.export(buffer, format="wav")
            buffer.seek(0)
            audio_arr, _ = sf.read(buffer)
            
            return audio_arr
            
        except Exception as e:
            logger.error(f"Failed to apply audio effects: {str(e)}")
            return audio_arr  # Return original array if effects fail

def parl_loader():
    """Legacy function for backwards compatibility"""
    generator = AudioGenerator()
    return (
        generator.model,
        generator.tokenizer,
        generator.description_tokenizer
    )

def describe_speaker(description_tokenizer, speaker="host", tone="happy", device="cpu"):
    """Legacy function for backwards compatibility"""
    generator = AudioGenerator()
    input_ids, attention_mask = generator.get_speaker_embedding(speaker, tone)
    return input_ids, attention_mask

def audio_generator(
    model, 
    tokenizer, 
    input_ids, 
    description_attention_mask, 
    device="cpu", 
    prompt="Hey!!! Pass an input to generate audio", 
    file="default", 
    speaker="female",
    tone="happy"
):
    """Legacy function for backwards compatibility"""
    generator = AudioGenerator()
    output_file = f"audio/{file}.wav"
    generator.generate_audio(prompt, speaker, tone, output_file)

if __name__ == "__main__":
    # Example usage
    generator = AudioGenerator()
    generator.generate_audio(
        "Hello, this is a test!",
        "female",
        "excited",
        "audio/test.wav"
    )