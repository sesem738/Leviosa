import logging
from whisper_service import WhisperService
from transcribe import SpeechToText
from microphone import MicrophoneRecorder


def main():
    """
    Main function to demonstrate the SpeechToText functionality.
    """
    # Example user selection
    selected_model = "base.en"  # User selects the model (e.g., "base.en", "small", etc.)
    weights_path = "data/weights/whisper/"  # Specify the directory for model weights

    # Create SpeechToText instance with Whisper service
    logging.info(f"Using Whisper service with model: {selected_model}")
    stt = SpeechToText(service_class=WhisperService, model_name=selected_model, weights_path=weights_path)
    logging.info("SpeechToText instance created.")

    # Specify the device index if needed
    choice_device = 2
    recorder = MicrophoneRecorder(device_index=choice_device)

    # Uncomment to display microphone information for debugging
    # recorder.display_devices_info()

    # Define the path to save the recorded audio file
    output_path = "data/audios/output.wav"

    # Record audio to a file
    try:
        logging.info("Starting audio recording. Press Ctrl+C to stop.")
        recorder.start_stream()
    except KeyboardInterrupt:
        recorder.stop_stream()
        logging.info(f"Recording stopped. Audio saved to {output_path}")

        # Transcribe the recorded audio file
        transcribed_text = stt.transcribe_audio(output_path)
        logging.info(f"Transcribed Text from file: {transcribed_text}")


if __name__ == "__main__":
