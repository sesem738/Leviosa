import logging
from typing import Optional, Type
from pydub import AudioSegment  # We will use pydub for getting audio file details

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
        logging.info(f"SpeechToText initialized with service: {service_class.__name__} and model: {model_name}")

    def transcribe_audio(self, audio_file: str) -> str:
        """
        Transcribe an audio file using the specified speech-to-text service.

        :param audio_file: Path to the audio file to transcribe.
        :return: Transcribed text.
        """
        # Log details about the audio file
        try:
            audio = AudioSegment.from_file(audio_file)
            file_length = len(audio) / 1000  # length in seconds
            logging.info(f"Transcribing audio file: {audio_file}")
            logging.info(f"Audio length: {file_length} seconds")
            logging.info(f"Channels: {audio.channels}")
            logging.info(f"Frame rate: {audio.frame_rate} Hz")
        except Exception as e:
            logging.error(f"Could not retrieve audio file details: {e}")

        # Transcribe the audio file
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
    # SpeechToText.display_model_options(WhisperService) # Uncomment to display model options for Whisper service

    # Example user selection
    selected_model = "base.en"  # User selects the model (e.g., "base.en", "small", etc.)
    weights_path = "data/weights/whisper/"  # Specify the directory for model weights

    # Create SpeechToText instance with Whisper service
    logging.info(f"Using Whisper service with model: {selected_model}")
    stt = SpeechToText(service_class=WhisperService, model_name=selected_model, weights_path=weights_path)
    logging.info("SpeechToText instance created.")

    # Transcribe a pre-recorded audio file
    audio_path = "data/audios/output.wav"
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
