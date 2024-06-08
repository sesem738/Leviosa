import logging
from datetime import datetime
from whisper_service import WhisperService
from transcribe import SpeechToText
from microphone import MicrophoneRecorder

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# TODO: DOES NOT WORK YET, WHISPER SERVICE TRANSCRIBE_STREAM METHOD NEEDS TO BE DEBUGGED

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
    choice_device = 2  # specific to my system
    recorder = MicrophoneRecorder(device_index=choice_device)

    # Uncomment to display microphone information for debugging
    # recorder.display_devices_info()

    def stt_stream_cb(data):
        """
        Callback function to process audio stream data and transcribe it.
        """
        transcribed_text = stt.service.transcribe_stream(data)
        if transcribed_text:
            logging.info(f"stt: {transcribed_text}")

    # Record audio and process in real-time with the callback
    try:
        logging.info("Starting audio recording. Press Ctrl+C to stop.")
        recorder.start_stream(callback=stt_stream_cb)
    except KeyboardInterrupt:
        recorder.stop_stream()
        logging.info("Recording stopped.")


if __name__ == "__main__":
    main()
