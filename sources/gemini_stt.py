import logging
import os
import time
from datetime import datetime

import google.generativeai as genai
from dotenv import load_dotenv

from speech_to_text.microphone import MicrophoneRecorder
from text_to_trajectory.trajectory import process_waypoints

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Load environment variables from .env file
load_dotenv()
GOOGLE_API_KEY = os.getenv('gemini_api_key')
if not GOOGLE_API_KEY:
    logging.error("Google API key not found. Please set the GOOGLE_API_KEY environment variable.")
    exit(1)

genai.configure(api_key=GOOGLE_API_KEY)

def fetch_waypoints_code_from_gemini(audio_file: str, error: str = None):
    """
    Fetches the Python code for generating waypoints using the Google AI API.
    :param audio_file: Path to the audio file.
    :param error: Error message from previous code execution, if any.
    :return: the code response from the AI model
    """
    # Start Timer
    start_time = time.perf_counter()  # Use perf_counter for higher precision timing
    # upload the audio file
    audio = genai.upload_file(path=audio_file)
    # prompt the model with the audio file
    base_prompt = """
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
    if error:
        base_prompt += f"\n\nThe previous code generated the following error:\n{error}\nPlease correct the code based on this error. Again, ensure that the code generates a list of waypoints. and is enclosed within triple backticks: ```python\n\n```\n\n"

    model = genai.GenerativeModel('models/gemini-1.5-flash')
    response = model.generate_content([base_prompt, audio])
    code_text = response.text  # Extract the code from the response

    # Stop Timer and Calculate Elapsed Time
    end_time = time.perf_counter()
    elapsed_time_ms = (end_time - start_time) * 1000  # Convert to milliseconds

    logging.info(f"Generated response:\n\n {code_text}")
    logging.info(f"Total time taken for transcription: {elapsed_time_ms:.2f} ms")

    return code_text


def process_waypoints_with_retry(audio_file: str, max_retries: int = 3, save_path: str = None):
    """
    Process the waypoints with a retry mechanism.
    :param audio_file: Path to the audio file.
    :param max_retries: Maximum number of retries.
    :param save_path: Path to save the plot image.
    :return: List of waypoints or None if the process fails.
    """
    error = None
    for attempt in range(max_retries):
        code_response = fetch_waypoints_code_from_gemini(audio_file, error)
        try:
            waypoints = process_waypoints(code_response, save_path=save_path)
            return waypoints
        except Exception as e:
            logging.error(f"An error occurred while processing waypoints: {e}")
            error = str(e)
            logging.info(f"Retrying... ({attempt + 1}/{max_retries})")
    logging.error("Maximum number of retries reached. Failed to process waypoints.")
    return None



def main():
    """
    Main function to test the Google AI API.
    """

    # Define the path to save the recorded audio file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"data/audios/output_{timestamp}.wav"
    traj_plot_path = f"data/plots/waypoints_{timestamp}.png"

    # Create directories if they don't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(os.path.dirname(traj_plot_path), exist_ok=True)

    # Specify the device index if needed
    choice_device = 1 # specific to my system
    recorder = MicrophoneRecorder(device_index=choice_device)

    # Record audio to a file
    try:
        logging.info("Starting audio recording. Press Ctrl+C to stop.")
        recorder.start_stream(save_path=output_path)
    except KeyboardInterrupt:
        recorder.stop_stream()
        logging.info(f"Recording stopped. Audio saved to {output_path}")

    # Process the waypoints with retry mechanism
    waypoints = process_waypoints_with_retry(output_path, max_retries=3, save_path=traj_plot_path)
    if waypoints:
        logging.info(f"Successfully processed waypoints: {waypoints}")
        print(f"Trajectory plot saved to: {traj_plot_path}")
    else:
        logging.error("Failed to process waypoints after maximum retries.")



if __name__ == "__main__":
    main()
