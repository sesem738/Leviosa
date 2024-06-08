import logging
import os
from typing import Optional

import numpy as np
import whisper
from scipy.signal import resample

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

        self.model_name = model_name

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
        :return: Transcribed text or None.
        """
        # Convert byte data to numpy array
        audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

        # Resample audio data to 16kHz
        audio_data_resampled = resample(audio_data, int(len(audio_data) * 16000 / 44100))

        # Create a buffer with a size expected by the Whisper model
        buffer_size = int(16000 * 30)  # 30 seconds buffer
        if len(audio_data_resampled) < buffer_size:
            audio_data_resampled = np.pad(audio_data_resampled, (0, buffer_size - len(audio_data_resampled)), 'constant')

        # Assuming data is 16-bit PCM, mono, 16kHz
        mel = whisper.log_mel_spectrogram(audio_data_resampled[:buffer_size])

        # Use Whisper model to transcribe the audio
        options = whisper.DecodingOptions(fp16=False, language="en" if ".en" in self.model_name else None)
        result = whisper.decode(self.model, mel, options)
        return result.text

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
