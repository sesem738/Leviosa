import logging
import os
import random
import time
from typing import Optional

DEFAULT_GUI = True
import google.generativeai as genai
from dotenv import load_dotenv

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# from speech_to_text.microphone import MicrophoneRecorder
from sources.text_to_trajectory.trajectory import process_waypoints, plot_multi_drone_3d_trajectory


def setup_logging(log_file_path):
    """
    Set up logging to a specified file.
    :param log_file_path: The file path for the log output.
    """
    # Reset logging handlers if they exist
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=log_file_path,
        filemode='w'  # Use 'a' to append to the log file instead of overwriting it
    )


# Load environment variables from .env file
load_dotenv()
GOOGLE_API_KEY = os.getenv('gemini_api_key')
if not GOOGLE_API_KEY:
    logging.error("Google API key not found. Please set the GOOGLE_API_KEY environment variable.")
    exit(1)

genai.configure(api_key=GOOGLE_API_KEY)


# Continuous Experiment Types and Their Prompts


def interpret_text_request(prompt: Optional[str] = None, audio_file: Optional[str] = None) -> str:
    """
    Interpret the text prompt into a list of requirements using the Google AI API.

    :param audio_file:  The path to the audio file containing the text prompt.
    :param prompt: The text prompt to interpret.

    :return: The interpreted requirements.
    """
    base_prompt = f""" You are an AI assistant specializing in translating natural language text or audio commands into 
    structured requirements for drone waypoint trajectories. Your task is to analyze the given text or audio prompt and 
    extract the underlying intent, then formulate a detailed set of requirements that a drone control system could use.

        Please follow these guidelines:
        1. Interpret the overall goal or purpose of the command, not just the literal words.
        2. Infer any implicit requirements that aren't directly stated but are necessary for the task.
        3. Specify the number of drones required, even if not explicitly mentioned.
        4. Determine appropriate starting positions based on the nature of the task.
        5. Describe the overall shape, path, or formation the drones should follow.
        7. This is just a trajectory generation task, so you don't need to consider real-time constraints or obstacle avoidance.

        Format your response as a structured list of requirements, each prefaced with [REQ], like this:
        [REQ] Number of drones: X
        [REQ] Starting formation: Description
        [REQ] Flight path: Detailed description
        ...

        Reason through your interpretation, but do not restate the content directly. 
        Focus on translating the command into actionable, technical requirements.
        """

    # model = genai.GenerativeModel('models/gemini-1.5-flash')
    # response = model.generate_content([base_prompt, prompt])
    if audio_file:
        prompt = genai.upload_file(path=audio_file)
    else:
        prompt = prompt
    response = call_gemini_with_retry([base_prompt, prompt])

    requirements = None
    try:
        requirements = response.text
    except ValueError as e:
        logging.error(f"An error occurred while interpreting the text request: {e}\n Exiting...")

    logging.info(f"Interpreted requirements:\n\n {requirements}")

    return requirements


def fetch_waypoints_code_from_gemini(requirements: str, error: str = None):
    """
    Fetches the Python code for generating waypoints for three drones using the Google AI API.

    :param requirements: The requirements for the waypoints.
    :param error: The error message from the previous code generation.

    :return: The generated Python code for waypoints.
    """
    start_time = time.perf_counter()

    base_prompt = f""" You are an AI assistant that converts requirements into Python code for generating 
    lists of waypoints for N drone trajectories. You will need to generate Python code that outputs N lists 
    of waypoints, each specified as [x, y, z] depending on the number of drones. If no specific number of drones is 
    specified, use N=3. When you receive a requirements, reason step-by-step. Assume the unit 
    of measurement is meters. Create trajectories for each drone. The trajectory for the 
    drones can either combine or be independent. The code should generate waypoints in the following format and be 
    enclosed within triple backticks: 

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
    generate the waypoint lists like np.sin, np.cos, np.linspace, etc. Think step by step before
    generating the python code. Every time you generate based on feedback, remember you have to start the trajectory 
    from scratch, you can't extend the previous trajectory.
    Based on the following requirements:
    {requirements}
    """

    if error:
        base_prompt += (
            f"\n\nThe previous code generated the following error:\n"
            f"{error}\nPlease correct the code based on this error.")

    # model = genai.GenerativeModel('models/gemini-1.5-flash')
    # response = model.generate_content([base_prompt])
    response = call_gemini_with_retry([base_prompt])
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


def analyze_plot_with_multiple_critics(
        image_path: str,
        requirements: str,
        num_critics: int = 3,
        prev_feedback: str = None
) -> str:
    """
    Analyze the plot image using multiple Gemini critics and provide aggregated feedback.

    :param image_path: The path to the plot image.
    :param requirements: The requirements for the plot.
    :param num_critics: The number of critics to analyze the plot.
    :param prev_feedback: The previous feedback for comparison.

    :return: The aggregated feedback from multiple critics.

    """
    start_time = time.perf_counter()
    image = genai.upload_file(path=image_path)

    base_prompt = f"""
    You are an AI assistant that analyzes multi-drone trajectory plots. I have provided an image file containing the 
    trajectory plot for drones. Based on the following requirements:
    {requirements}
    Please analyze the plot and provide feedback on the trajectories. Specifically, look for:
    1. Continuity of each drone's path
    2. Completeness of the trajectories based on the requirements
    3. Any anomalies or potential collisions between drones
    4. Appropriate starting positions for each drone
    5. Depending on the requirements, each drone does NOT have to come back to the starting point
    6. IMPORTANTLY, The overall shape formed by the combination of all the drones' trajectories SHOULD match what the 
    requirements specify!
    7. Always Score the trajectories based on how well they meet the requirements from 0 to 100.
    Think step by step and be detailed in your analysis. 
    If all trajectories are roughly correct, please respond with the phrase "--VALID TRAJECTORIES--" and comments on why you 
    think they are valid. If trajectory is incorrect, say whether it is close or not and provide suggestions on how to 
    correct it. 
    """

    feedbacks = []
    for i in range(num_critics):
        # response = model.generate_content([base_prompt, image])
        response = call_gemini_with_retry([base_prompt, image])
        try:
            feedback = response.text
            feedbacks.append(feedback)
        except ValueError as e:
            logging.error(f"An error occurred while extracting the feedback from critic {i + 1}: {e}")

    end_time = time.perf_counter()
    elapsed_time_ms = (end_time - start_time) * 1000

    logging.info(
        f"Total time taken for plot analysis with {num_critics} critics: {elapsed_time_ms:.2f} ms")

    agg_feedback = aggregate_feedback(feedbacks, prev_feedback=prev_feedback)
    logging.info(f"Aggregated feedback from multiple critics:\n\n {agg_feedback}")

    return agg_feedback


def aggregate_feedback(
        feedbacks: list,
        acceptance_rate: float = 0.75,
        prev_feedback: str = ""
) -> str:
    """
    Aggregate feedback from multiple critics and summarize it using the Gemini model.

    :param feedbacks: The list of feedback from multiple critics.
    :param acceptance_rate: The acceptance rate threshold for majority consensus.
    :param prev_feedback: The previous feedback for comparison.

    :return: The aggregated feedback summary.
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
    {total_critics} critics without mentioning the specifics about the individual critics to avoid confusion. 
    Don't mention individual critic scores. Your task is to summarize the feedback, identifying common points and the 
    overall consensus. 
    Here is the feedback from the critics:
    {" ".join(feedbacks)}


    Finally tell based on the previous feedback, what the previous score was (the previous overall over 100 number) and 
    what the current score is. and how much the score has improved or decreased by a + or -. 
    If the score  Say "BETTER" if the score has improved, "WORSE" if the score has decreased,

    Here is the previous feedback:
    {prev_feedback}
    """

    # model = genai.GenerativeModel('models/gemini-1.5-flash')
    # response = model.generate_content([base_prompt])
    response = call_gemini_with_retry([base_prompt])

    summary = response.text if response else "Error in generating summary"

    return f"{result} ({valid_count}/{total_critics})\n Feedback Summary:\n{summary}"


def process_waypoints_with_retry(
        commands: str,
        save_path: str = None,
        num_critics: int = 5,
        max_retries: int = 3
):
    """
    Process the waypoints with a retry mechanism.

    :param commands: The text prompt or audio file path for the experiment.
    :param save_path: The path to save the plot image.
    :param num_critics: The number of critics to analyze the plot.
    :param max_retries: The maximum number of retries.

    :return: The processed waypoints if successful, otherwise None.

    """
    error = None
    feedback = None
    prev_feedback = ""
    best_waypoints = None

    # Interpret the text prompt
    requirements = interpret_text_request(audio_file=commands)
    print(requirements)

    for attempt in range(max_retries):
        code_response = fetch_waypoints_code_from_gemini(requirements, error or feedback)
        try:
            waypoints = process_waypoints(code_response, save_path=save_path)

            # Analyze the generated plot image with multiple Gemini critics
            feedback = analyze_plot_with_multiple_critics(save_path, requirements, num_critics, prev_feedback)

            prev_feedback = feedback
            if "BETTER" in feedback:
                # save these waypoints as the best waypoints
                best_waypoints = waypoints
                best_save_path = save_path.replace(".png", "_best.png")
                plot_multi_drone_3d_trajectory(best_waypoints, plot=False, save_path=best_save_path)

            if "MAJORITY VALID" in feedback:
                return waypoints

        except Exception as e:

            logging.error(f"An error occurred while processing waypoints: {e}")
            error = str(e)
        logging.info(f"Retrying... ({attempt + 1}/{max_retries})")
    logging.error("Maximum number of retries reached. Failed to process waypoints.")
    return None


def retry_with_backoff(attempt, max_attempts=5, base_delay=1):
    """
    Retry function with exponential backoff and jitter.

    :param attempt: The current attempt number.
    :param max_attempts: The maximum number of retry attempts.
    :param base_delay: The base delay in seconds for exponential backoff.

    :return: True if the retry should proceed, False if the maximum number of retries is reached.
    """
    if attempt >= max_attempts:
        logging.error("Max attempts reached, aborting...")
        return False  # Indicate that the maximum number of retries has been reached
    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)  # Exponential backoff with jitter
    # range of delay is
    logging.info(f"Retrying in {delay:.2f} seconds...")
    time.sleep(delay)
    return True  # Indicate that the retry should proceed


def call_gemini_with_retry(base_prompt, model_name='models/gemini-1.5-pro', max_attempts=25, base_delay=1):
    """
    Calls the Gemini model API with retry logic and exponential backoff.
    the more recent and powerful the model is, the more retries and longer delays are needed.

    :param base_prompt: The prompt to send to the model.
    :param model_name: The name of the Gemini model to use.
    :param max_attempts: Maximum number of retry attempts.
    :param base_delay: Base delay in seconds for exponential backoff.

    :return: The response text from the model, or None if the call fails.
    """
    model = genai.GenerativeModel(model_name)

    for attempt in range(max_attempts):
        try:
            response = model.generate_content(base_prompt)
            response_text = response
            if response_text:
                return response  # Return the response text if successful
        except Exception as e:
            logging.error(f"An error occurred during Gemini API call: {e}")
            if "429" in str(e):
                if not retry_with_backoff(attempt, max_attempts, base_delay):
                    break  # Exit the loop if maximum retries are reached
            else:
                break  # Break if the error is not retryable

    logging.error("Failed to get a valid response from the Gemini API.")
    return None  # Return None if all attempts fail
