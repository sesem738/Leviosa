import logging
import os
from typing import Optional

import whisper

from abstract_service import SpeechToTextService


class WhisperService(SpeechToTextService):
    """
    Implementation of the SpeechToTextService using the Whisper API.
    """

    def __init__(self, model_name: str = "base.en", weights_path: Optional[str] = None):
        """
        Initialize the WhisperService with a specific model and optional weights path.

        :param model_name: Name of the Whisper model to use.
        :param weights_path: Optional path to the directory where model weights are stored.
        """
        if weights_path and not os.path.exists(weights_path):
            os.makedirs(weights_path)
            logging.info(f"Created directory for weights: {weights_path}")

        if weights_path:
            self.model = whisper.load_model(model_name, download_root=weights_path)
        else:
            self.model = whisper.load_model(model_name)

    def transcribe(self, audio_file: str) -> str:
        """
        Transcribe the given audio file to text.

        :param audio_file: Path to the audio file to transcribe.
        :return: Transcribed text.
        """
        result = self.model.transcribe(audio_file)
        return result['text']

    def transcribe_stream(self, data: bytes) -> Optional[str]:
        """
        Transcribe a stream of audio data to text.

        :param data: Chunk of audio data.
        :return: Transcribed text or None. Whisper does not support direct streaming transcription
                 in this simplified example. This function would require a more complex implementation
                 or a different approach.
        """
        return None

    @staticmethod
    def display_model_options():
        """
        Display the available Whisper model options.
        """
        MODEL_INFO = [
            {"name": "tiny", "params": "39 M", "english_only": "tiny.en", "multilingual": "tiny", "vram": "~1 GB",
             "speed": "~32x"},
            {"name": "base", "params": "74 M", "english_only": "base.en", "multilingual": "base", "vram": "~1 GB",
             "speed": "~16x"},
            {"name": "small", "params": "244 M", "english_only": "small.en", "multilingual": "small", "vram": "~2 GB",
             "speed": "~6x"},
            {"name": "medium", "params": "769 M", "english_only": "medium.en", "multilingual": "medium",
             "vram": "~5 GB",
             "speed": "~2x"},
            {"name": "large", "params": "1550 M", "english_only": "N/A", "multilingual": "large", "vram": "~10 GB",
             "speed": "1x"}
        ]

        logging.info("Available Whisper models:")
        for model in MODEL_INFO:
            logging.info(
                f"Model: {model['name']}, Parameters: {model['params']}, English-only: {model['english_only']}, "
                f"Multilingual: {model['multilingual']}, VRAM: {model['vram']}, Speed: {model['speed']}")
