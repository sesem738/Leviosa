import logging
import os
import time
from datetime import datetime

import google.generativeai as genai
from dotenv import load_dotenv
from microphone import MicrophoneRecorder

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    """
    Main function to test the Google AI API.
    """
    # Load environment variables from .env file
    load_dotenv()
    GOOGLE_API_KEY = os.getenv('gemini_api_key')
    if not GOOGLE_API_KEY:
        logging.error("Google API key not found. Please set the GOOGLE_API_KEY environment variable.")
        return

    genai.configure(api_key=GOOGLE_API_KEY)

    # Define the path to save the recorded audio file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"data/audios/output_{timestamp}.wav"

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

    # Start Timer
    start_time = time.perf_counter()  # Use perf_counter for higher precision timing

    file_obj = genai.upload_file(path=output_path)
    prompt = "Listen carefully to the following audio file and tell me what was said."
    model = genai.GenerativeModel('models/gemini-1.5-flash')
    response = model.generate_content([prompt, file_obj])

    # Stop Timer and Calculate Elapsed Time
    end_time = time.perf_counter()
    elapsed_time_ms = (end_time - start_time) * 1000  # Convert to milliseconds

    logging.info(f"Transcribed Text: \n\n{response.text}")
    logging.info(f"Total time taken for transcription: {elapsed_time_ms:.2f} ms")


if __name__ == "__main__":
    main()
