import logging
import os
import time
from datetime import datetime

import google.generativeai as genai
from dotenv import load_dotenv

from speech_to_text.microphone import MicrophoneRecorder
from text_to_trajectory.trajectory import (
    extract_code_from_response,
    execute_waypoints_code,
    plot_3d_trajectory
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Load environment variables from .env file
load_dotenv()
GOOGLE_API_KEY = os.getenv('gemini_api_key')
if not GOOGLE_API_KEY:
    logging.error("Google API key not found. Please set the GOOGLE_API_KEY environment variable.")
    exit(1)

genai.configure(api_key=GOOGLE_API_KEY)


def fetch_waypoints_code_from_gemini(audio_file: str):
    """
    Fetches the Python code for generating waypoints using the Google AI API.
    :param audio_file: Path to the audio file.
    :return: the code response from the AI model
    """
    # upload the audio file
    audio = genai.upload_file(path=audio_file)
    # prompt the model with the audio file
    prompt = """
    You are an AI assistant that converts natural audio commands into Python code for generating a list of waypoints for drone trajectories.
    You will need to generate Python code that outputs a list of waypoints, each specified as [x, y, z].
    When you receive a user prompt, reason step-by-step.
    Assume the unit of measurement is meters.
    The waypoints should start from [0, 0, 1] and create a continuous trajectory.
    The code should generate waypoints in the following format and be enclosed within triple backticks:
    ```python
    
    ```
     The code you write should not define a function that gets called. It should directly generate the waypoints.
     Executing the code should output a list of waypoints. I should not need to call a function you write to get the waypoints.
     Make sure to import all the necessary libraries you use in the code.
     No need to use a return statement since we are not defining a function.
     
    Listen carefully to the following audio file, tell me back the command you understand I said, and 
    convert the audio command into Python code for generating waypoints.
    """
    # Start Timer
    start_time = time.perf_counter()  # Use perf_counter for higher precision timing
    model = genai.GenerativeModel('models/gemini-1.5-flash')
    response = model.generate_content([prompt, audio])
    code_text = response.text  # Extract the code from the response

    # Stop Timer and Calculate Elapsed Time
    end_time = time.perf_counter()
    elapsed_time_ms = (end_time - start_time) * 1000  # Convert to milliseconds

    logging.info(f"Generated response:\n\n {code_text}")
    logging.info(f"Total time taken for transcription: {elapsed_time_ms:.2f} ms")

    return code_text


def main():
    """
    Main function to test the Google AI API.
    """

    # Define the path to save the recorded audio file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"data/audios/output_{timestamp}.wav"

    # Specify the device index if needed
    choice_device = 2  # specific to my system
    recorder = MicrophoneRecorder(device_index=choice_device)

    # Record audio to a file
    try:
        logging.info("Starting audio recording. Press Ctrl+C to stop.")
        recorder.start_stream(save_path=output_path)
    except KeyboardInterrupt:
        recorder.stop_stream()
        logging.info(f"Recording stopped. Audio saved to {output_path}")

    # Fetch the Python code for generating waypoints using the OpenAI API
    code_response = fetch_waypoints_code_from_gemini(output_path)
    if code_response:
        # Extract the code from the response
        code = extract_code_from_response(code_response)
        if code:
            # Execute the extracted Python code to get the waypoints
            waypoints = execute_waypoints_code(code)
            if waypoints:
                print(f"Derived waypoints: {waypoints}")

                # Plot the 3D trajectory based on the derived waypoints
                plot_3d_trajectory(waypoints)
            else:
                print("Failed to derive waypoints.")
        else:
            print("Failed to extract Python code.")
    else:
        print("Failed to generate Python code.")


if __name__ == "__main__":
    main()
