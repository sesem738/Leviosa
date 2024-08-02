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
    Fetches the Python code for generating waypoints for three drones using the Google AI API.
    """
    start_time = time.perf_counter()
    audio = genai.upload_file(path=audio_file)

    base_prompt = f""" You are an AI assistant that converts natural audio commands into Python code for generating 
    lists of waypoints for N drone trajectories. You will need to generate Python code that outputs N lists 
    of waypoints, each specified as [x, y, z] depending on the number of drones. If no specific number of drones is 
    specified, use N=3. When you receive a user prompt, reason step-by-step. Assume the unit 
    of measurement is meters. The waypoints for each drone should start from different positions.
     Create continuous trajectories for each drone. The trajectory for the 
    drones can either combine or be independent based on the audio command. The code should generate 
    waypoints in the following format and be enclosed within triple backticks: 
    ```python 
    import numpy as np
    
    #define any preprocessing functions or steps necessary here

    # Drone 1 waypoints
    waypoints1 =...

    # Drone 2 waypoints
    waypoints2 = ...

    ... 
    # Drone N waypoints
    waypointsN = ...

    waypoints = [waypoints1, waypoints2, ... waypointsN]
    ```
    Make sure to import all necessary libraries you use in the code. Feel free to also use numpy functions to help you 
    generate the waypoint lists like np.sin, np.cos, np.linspace, etc.
    Listen carefully to the following audio file, tell me back the command you understand I said, and 
    convert the audio command into Python code for generating waypoints for three drones. Think step by step before
    generating the python code. Every time you generate based on feedback, remember you have to start the trajectory 
    from scratch, you can't extend the previous trajectory.
    """

    if error:
        base_prompt += (f"\n\nThe previous code generated the following error:\n{error}\nPlease correct the code based "
                        f"on this error.")

    model = genai.GenerativeModel('models/gemini-1.5-flash')
    response = model.generate_content([base_prompt, audio])

    code_text = None
    try:
        code_text = response.text
    except ValueError as e:
        logging.error(f"An error occurred while extracting the code from the response: {e}\n Exiting...")

    end_time = time.perf_counter()
    elapsed_time_ms = (end_time - start_time) * 1000

    logging.info(f"Generated response:\n\n {code_text}")
    logging.info(f"Total time taken for transcription: {elapsed_time_ms:.2f} ms")

    return code_text


def analyze_plot_with_multiple_critics(audio_file: str, image_path: str, num_critics: int = 3):
    """
    Analyze the plot image using multiple Gemini critics and provide aggregated feedback.
    """
    start_time = time.perf_counter()
    audio = genai.upload_file(path=audio_file)
    image = genai.upload_file(path=image_path)

    base_prompt = """
    You are an AI assistant that analyzes multi-drone trajectory plots. I have provided an audio file with a command and
     an image file containing the trajectory plot for drones.
    Please analyze the plot and provide feedback on the trajectories. Specifically, look for:
    1. Continuity of each drone's path
    2. Completeness of the trajectories based on the audio command
    3. Any anomalies or potential collisions between drones
    4. Appropriate starting positions for each drone
    5. Depending on the audio command, each drone does NOT have to come back to the starting point
    6. IMPORTANTLY, The overall shape formed by the combination of all the drones trajectories SHOULD match what the 
    audio command asks!
    Think step by step and be detailed in your analysis. 
    If all trajectories are correct, please respond with the phrase "--VALID TRAJECTORIES--" and comments on why you 
    think they are valid. If any trajectory is incorrect, say whether it is close or not and provide suggestions on how to 
    correct it. 
    """

    model = genai.GenerativeModel('models/gemini-1.5-flash')

    feedbacks = []
    for i in range(num_critics):
        response = model.generate_content([base_prompt, audio, image])
        try:
            feedback = response.text
            feedbacks.append(feedback)
        except ValueError as e:
            logging.error(f"An error occurred while extracting the feedback from critic {i + 1}: {e}")

    end_time = time.perf_counter()
    elapsed_time_ms = (end_time - start_time) * 1000

    logging.info(f"Total time taken for plot analysis with {num_critics} critics: {elapsed_time_ms:.2f} ms")

    agg_feedback = aggregate_feedback(feedbacks)
    logging.info(f"Aggregated feedback from multiple critics:\n\n {agg_feedback}")

    return agg_feedback


def aggregate_feedback(feedbacks: list, acceptance_rate: float = 0.75) -> str:
    """
    Aggregate feedback from multiple critics and summarize it using the Gemini model.
    """
    valid_count = sum("--VALID TRAJECTORIES--" in feedback for feedback in feedbacks)
    total_critics = len(feedbacks)
    majority_threshold = int(total_critics * acceptance_rate)

    if valid_count > majority_threshold:
        result = "MAJORITY VALID"
    else:
        result = "MAJORITY INVALID"

    # Summarize feedback using Gemini model
    base_prompt = f"""
    You are an AI assistant that summarizes feedback from multiple critics. I will provide you with the feedback from 
    {total_critics} critics. Your task is to summarize the feedback, identifying common points and the 
    overall consensus. 
    Here is the feedback from the critics:
    {" ".join(feedbacks)}
    """

    model = genai.GenerativeModel('models/gemini-1.5-flash')
    response = model.generate_content([base_prompt])

    summary = response.text if response else "Error in generating summary"

    return f"{result} ({valid_count}/{total_critics})\n Feedback Summary:\n{summary}"


def process_waypoints_with_retry(audio_file: str, max_retries: int = 3, save_path: str = None, num_critics: int = 5):
    """
    Process the waypoints with a retry mechanism.
    """
    error = None
    feedback = None
    for attempt in range(max_retries):
        code_response = fetch_waypoints_code_from_gemini(audio_file, error or feedback)
        try:
            waypoints = process_waypoints(code_response, save_path=save_path)

            # Analyze the generated plot image with multiple Gemini critics
            feedback = analyze_plot_with_multiple_critics(audio_file, save_path, num_critics)
            if "MAJORITY VALID" in feedback:
                return waypoints
        except Exception as e:
            logging.error(f"An error occurred while processing waypoints: {e}")
            error = str(e)
        logging.info(f"Retrying... ({attempt + 1}/{max_retries})")
    logging.error("Maximum number of retries reached. Failed to process waypoints.")
    return None


def analyze_plot_with_gemini(audio_file: str, image_path: str):
    """
    Analyze the plot image using Gemini and provide feedback.
    """
    start_time = time.perf_counter()
    audio = genai.upload_file(path=audio_file)
    image = genai.upload_file(path=image_path)

    base_prompt = """
    You are an AI assistant that analyzes multi-drone trajectory plots. I have provided an audio file with a command and
     an image file containing the trajectory plot for three drones.
    Please analyze the plot and provide feedback on the trajectories. Specifically, look for:
    1. Continuity of each drone's path
    2. Completeness of the trajectories based on the audio command
    3. Any anomalies or potential collisions between drones
    4. Appropriate starting positions for each drone
    5. Depending on the audio command, each drone does NOT have to come back to the starting point
    6. IMPORTANTLY, The overall shape formed by the combination of all the drones trajectories SHOULD match what the 
    audio command asks!
    Think step by step and be detailed in your analysis.
    If all trajectories are correct, please respond with the phrase "--VALID TRAJECTORIES--" and comments on why you 
    think they are valid. If any trajectory is incorrect, provide suggestions on how to correct it. 
    """

    model = genai.GenerativeModel('models/gemini-1.5-flash')
    response = model.generate_content([base_prompt, audio, image])

    feedback = None
    try:
        feedback = response.text
    except ValueError as e:
        logging.error(f"An error occurred while extracting the feedback from the response: {e}\n Exiting...")

    end_time = time.perf_counter()
    elapsed_time_ms = (end_time - start_time) * 1000

    logging.info(f"Feedback from plot analysis:\n\n {feedback}")
    logging.info(f"Total time taken for plot analysis: {elapsed_time_ms:.2f} ms")

    return feedback


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
    waypoints = process_waypoints_with_retry(output_path, max_retries=30, save_path=traj_plot_path, num_critics=3)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    if waypoints:
        logging.info(f"Successfully processed waypoints")
    else:
        logging.error("Failed to process waypoints after maximum retries.")

    logging.info(f"Total time taken for process_waypoints_with_retry: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
