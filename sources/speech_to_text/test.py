import logging
from datetime import datetime
from whisper_service import WhisperService
from transcribe import SpeechToText
from microphone import MicrophoneRecorder

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    """
    Main function to demonstrate the SpeechToText functionality.
    """
    # Example user selection
    selected_model = "base.en"  # User selects the model (e.g., "base.en", "small", etc.)
    weights_path = "data/weights/whisper/"  # Specify the directory for model weights
    # Define the path to save the recorded audio file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"data/audios/output_{timestamp}.wav"

    # Create SpeechToText instance with Whisper service
    logging.info(f"Using Whisper service with model: {selected_model}")
    stt = SpeechToText(service_class=WhisperService, model_name=selected_model, weights_path=weights_path)
    logging.info("SpeechToText instance created.")

    # Specify the device index if needed
    choice_device = 2  # specific to my system
    recorder = MicrophoneRecorder(device_index=choice_device)

    # Uncomment to display microphone information for debugging
    # recorder.display_devices_info()

    # Record audio to a file
    try:
        logging.info("Starting audio recording. Press Ctrl+C to stop.")
        recorder.start_stream(save_path=output_path)
    except KeyboardInterrupt:
        recorder.stop_stream()
        logging.info(f"Recording stopped. Audio saved to {output_path}")

    # Transcribe the recorded audio file
    transcribed_text = stt.transcribe_audio(output_path)
    logging.info(f"Transcribed Text from file: \n\n{transcribed_text}")


if __name__ == "__main__":
    main()
