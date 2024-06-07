import logging
from abc import ABC, abstractmethod
from typing import Optional

import whisper
from microphone import MicrophoneRecorder

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SpeechToTextService(ABC):
    """
    Abstract base class for speech-to-text services.
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


class WhisperService(SpeechToTextService):
    """
    Implementation of the SpeechToTextService using the Whisper API.
    """
    def __init__(self):
        self.model = whisper.load_model("base")

    def transcribe(self, audio_file: str) -> str:
        result = self.model.transcribe(audio_file)
        return result['text']

    def transcribe_stream(self, data: bytes) -> Optional[str]:
        # For simplicity, Whisper does not support direct streaming transcription.
        # This function needs a more complex implementation or a different approach.
        return None


class SpeechToTextFactory:
    """
    Factory class to get instances of speech-to-text services.
    """
    @staticmethod
    def get_service(service_name: str) -> SpeechToTextService:
        if service_name == "whisper":
            return WhisperService()
        # Add more services here as needed
        else:
            raise ValueError(f"Service {service_name} is not supported.")


def transcribe_audio(service_name: str, audio_file: str) -> str:
    """
    Transcribe an audio file using the specified speech-to-text service.

    :param service_name: Name of the speech-to-text service to use.
    :param audio_file: Path to the audio file to transcribe.
    :return: Transcribed text.
    """
    service = SpeechToTextFactory.get_service(service_name)
    return service.transcribe(audio_file)


def transcribe_audio_stream(service: SpeechToTextService, data: bytes) -> None:
    """
    Transcribe audio stream data using the specified speech-to-text service.

    :param service: Instance of the speech-to-text service to use.
    :param data: Audio data chunk to transcribe.
    """
    result = service.transcribe_stream(data)
    if result:
        logging.info(f"Transcribed Text: {result}")


if __name__ == "__main__":
    # Transcribe a pre-recorded audio file
    audio_path = "data/audios/output.wav"
    transcribed_text = transcribe_audio("whisper", audio_path)
    logging.info(f"Transcribed Text from file: {transcribed_text}")

    # Set up real-time transcription
    choice_device = 2  # Specify the device index if needed
    recorder = MicrophoneRecorder(device_index=choice_device)
    stt_service = WhisperService()

    # Real-time transcription using the MicrophoneRecorder
    def stream_callback(data: bytes):
        transcribe_audio_stream(stt_service, data)

    try:
        recorder.start_stream(callback=stream_callback)
    except KeyboardInterrupt:
        recorder.stop_stream()
