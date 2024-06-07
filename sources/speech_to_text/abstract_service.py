from abc import ABC, abstractmethod
from typing import Optional


class SpeechToTextService(ABC):
    """
    Abstract base class for speech_to_text services.
    """

    @abstractmethod
    def transcribe(self, audio_file: str) -> str:
        """
        Transcribe the given audio file to text.

        :param audio_file: Path to the audio file to transcribe.
        :return: Transcribed text.
        """
        pass

    @abstractmethod
    def transcribe_stream(self, data: bytes) -> Optional[str]:
        """
        Transcribe a stream of audio data to text.

        :param data: Chunk of audio data.
        :return: Transcribed text or None.
        """
        pass

    @staticmethod
    @abstractmethod
    def display_model_options():
        """
        Display the available model options for the service.
        """
        pass
