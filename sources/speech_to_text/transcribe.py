import logging
from typing import Optional, Type

from microphone import MicrophoneRecorder
from whisper_service import WhisperService
from abstract_service import SpeechToTextService

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SpeechToText:
    """
    Encapsulates the Speech to Text functionality using a specified service.
    """

    def __init__(self, service_class: Type[SpeechToTextService], model_name: str = "base.en", weights_path: Optional[str] = None):
        """
        Initialize the SpeechToText with a specific service class, model, and weights path.

        :param service_class: The class of the speech-to-text service to use.
        :param model_name: Name of the model to use.
        :param weights_path: Optional path to the directory where model weights are stored.
        """
        self.service: SpeechToTextService = service_class(model_name=model_name, weights_path=weights_path)

    def transcribe_audio(self, audio_file: str) -> str:
        """
        Transcribe an audio file using the specified speech-to-text service.

        :param audio_file: Path to the audio file to transcribe.
        :return: Transcribed text.
        """
        return self.service.transcribe(audio_file)

    def transcribe_audio_stream(self, data: bytes) -> None:
        """
        Transcribe audio stream data using the specified speech-to-text service.

        :param data: Audio data chunk to transcribe.
        """
        result = self.service.transcribe_stream(data)
        if result:
            logging.info(f"Transcribed Text: {result}")

    @staticmethod
    def display_model_options(service_class: Type[SpeechToTextService]):
        """
        Display the available model options for the given service class.
        """
        service_class.display_model_options()


if __name__ == "__main__":
    # Display model options for Whisper service
    SpeechToText.display_model_options(WhisperService)

    # Example user selection
    selected_model = "base.en"  # User selects the model (e.g., "base.en", "small", etc.)
    weights_path = "data/weights/whisper/"  # Specify the directory for model weights

    # Create SpeechToText instance with Whisper service
    logging.info(f"Using Whisper service with model: {selected_model}")
    stt = SpeechToText(service_class=WhisperService, model_name=selected_model, weights_path=weights_path)
    logging.info("SpeechToText instance created.")

    # Transcribe a pre-recorded audio file
    audio_path = "data/audios/output.wav"
    logging.info(f"Transcribing audio file: {audio_path}")
    transcribed_text = stt.transcribe_audio(audio_path)
    logging.info(f"Transcribed Text from file: {transcribed_text}")

    # Set up real-time transcription
    # choice_device = 2  # Specify the device index if needed
    # recorder = MicrophoneRecorder(device_index=choice_device)

    # Real-time transcription using the MicrophoneRecorder
    # def stream_callback(data: bytes):
    #     stt.transcribe_audio_stream(data)
    #
    # try:
    #     recorder.start_stream(callback=stream_callback)
    # except KeyboardInterrupt:
    #     recorder.stop_stream()
