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

few_shot_examples = ""
# """"
# Example 1:
# Audio command: "Execute a figure eight"
# ```python
# import numpy as np
# t = np.linspace(0, 2*np.pi, 100)
# x = np.sin(t)
# y = np.sin(2*t)
# z = np.ones_like(t) * 3
# waypoints = np.column_stack((x, y, z))
# ```
#
# Example 2:
# Audio command: "Perform a circle"
# ```python
# import numpy as np
# t = np.linspace(0, 2*np.pi, 100)
# x = 5 * np.cos(t)
# y = 5 * np.sin(t)
# z = np.ones_like(t) * 2
# waypoints = np.column_stack((x, y, z))
# ```
#
# Example 3:
# Audio command: "Do a square"
# ```python
# import numpy as np
# waypoints = np.array([
#     [0, 0, 1],
#     [4, 0, 1],
#     [4, 4, 1],
#     [0, 4, 1],
#     [0, 0, 1]
# ])
# ```
# """


def fetch_waypoints_code_from_gemini(audio_file: str, error: str = None):
    """
    Fetches the Python code for generating waypoints using the Google AI API.
    :param audio_file: Path to the audio file.
    :param error: Error message from previous code execution, if any.
    :return: the code response from the AI model

    TODO: add few shots examples to help, 3 for now
    """
    # Start Timer
    start_time = time.perf_counter()  # Use perf_counter for higher precision timing
    # upload the audio file
    audio = genai.upload_file(path=audio_file)
    # prompt the model with the audio file
    base_prompt = f"""
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
    
    {few_shot_examples}
    """
    if error:
        base_prompt += f"\n\nThe previous code generated the following error:\n{error}\nPlease correct the code based on this error. Again, ensure that the code generates a list of waypoints and is enclosed within triple backticks: ```python\n\n```\n\n"

    model = genai.GenerativeModel('models/gemini-1.5-flash')
    response = model.generate_content([base_prompt, audio])

    code_text = None
    try:
        code_text = response.text  # Extract the code from the response
    except ValueError as e:
        logging.error(f"An error occurred while extracting the code from the response: {e}\n Exiting...")

    # Stop Timer and Calculate Elapsed Time
    end_time = time.perf_counter()
    elapsed_time_ms = (end_time - start_time) * 1000  # Convert to milliseconds

    logging.info(f"Generated response:\n\n {code_text}")
    logging.info(f"Total time taken for transcription: {elapsed_time_ms:.2f} ms")

    return code_text


def analyze_plot_with_gemini(audio_file: str, image_path: str):
    """
    Analyze the plot image using Gemini and provide feedback.
    :param audio_file: Path to the audio file.
    :param image_path: Path to the plot image.
    :return: Feedback from the AI model.
    """
    # Start Timer
    start_time = time.perf_counter()  # Use perf_counter for higher precision timing
    # Upload the image file and audio file
    audio = genai.upload_file(path=audio_file)
    image = genai.upload_file(path=image_path)

    # Prompt the model with the image file and audio command
    base_prompt = """
    You are an AI assistant that analyzes drone trajectory plots. I have provided an audio file with a command and an image file containing the trajectory plot.
    Please analyze the plot and provide feedback on the trajectory. Specifically, look for continuity, completeness, and any anomalies based on the command from the audio file.
    Think step by step and be detailed in your analysis.
    If the trajectory is correct, please respond with the phrase "--VALID TRAJECTORY--" and comments why you think it is valid.
    If the trajectory is incorrect, provide suggestions on how to correct it.
    """
    model = genai.GenerativeModel('models/gemini-1.5-flash')
    response = model.generate_content([base_prompt, audio, image])

    feedback = None
    try:
        feedback = response.text  # Extract the feedback from the response
    except ValueError as e:
        logging.error(f"An error occurred while extracting the feedback from the response: {e}\n Exiting...")

    # Stop Timer and Calculate Elapsed Time
    end_time = time.perf_counter()
    elapsed_time_ms = (end_time - start_time) * 1000  # Convert to milliseconds

    logging.info(f"Feedback from plot analysis:\n\n {feedback}")
    logging.info(f"Total time taken for plot analysis: {elapsed_time_ms:.2f} ms")

    return feedback


def process_waypoints_with_retry(audio_file: str, max_retries: int = 3, save_path: str = None):
    """
    Process the waypoints with a retry mechanism.
    :param audio_file: Path to the audio file.
    :param max_retries: Maximum number of retries.
    :param save_path: Path to save the plot image.
    :return: List of waypoints or None if the process fails.
    """
    error = None
    feedback = None
    for attempt in range(max_retries):
        code_response = fetch_waypoints_code_from_gemini(audio_file, error or feedback)
        try:
            waypoints = process_waypoints(code_response, save_path=save_path)

            # Analyze the generated plot image with Gemini
            feedback = analyze_plot_with_gemini(audio_file, save_path)
            if "--VALID TRAJECTORY--" in feedback:
                return waypoints
            # logging.info(f"Feedback from plot analysis: {feedback}")
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
    choice_device = 2  # specific to my system
    recorder = MicrophoneRecorder(device_index=choice_device)

    # Record audio to a file
    try:
        logging.info("Starting audio recording. Press Ctrl+C to stop.")
        recorder.start_stream(save_path=output_path)
    except KeyboardInterrupt:
        recorder.stop_stream()
        logging.info(f"Recording stopped. Audio saved to {output_path}")

    # Time the process_waypoints_with_retry function
    start_time = time.perf_counter()
    waypoints = process_waypoints_with_retry(output_path, max_retries=50, save_path=traj_plot_path)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    if waypoints:
        logging.info(f"Successfully processed waypoints")
    else:
        logging.error("Failed to process waypoints after maximum retries.")

    logging.info(f"Total time taken for process_waypoints_with_retry: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
