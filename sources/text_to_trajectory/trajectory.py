import os
import re
import time
import logging
import matplotlib.pyplot as plt
import numpy as np
import openai
from dotenv import load_dotenv
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()


def read_latest_command(file_path):
    """
    Reads the latest command from a given text file.
    
    Args:
        file_path (str): Path to the text file containing commands.
    
    Returns:
        str: The last line in the file as the latest command.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
        if lines:
            return lines[-1].strip()
        return None


def fetch_waypoints_code(client, command):
    """
    Uses the OpenAI API in chat mode to convert a text command into Python code for generating waypoints.
    
    Args:
        client (OpenAI): OpenAI client initialized with the API key.
        command (str): Text command describing the desired trajectory.
    
    Returns:
        str: Python code for generating the waypoints.
    """
    try:
        # Start timing the API call
        start_time = time.time()

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": """
                You are an AI assistant that converts text commands into Python code for generating a list of waypoints for drone trajectories.
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
                """
                },
                {"role": "user",
                 "content": f"Convert the following command into Python code for generating waypoints: '{command}'"}
            ]
        )

        # End timing the API call
        end_time = time.time()
        duration = end_time - start_time
        print(f"Time taken for API call: {duration:.2f} seconds")

        # Extract and print the response
        code_text = response.choices[0].message.content
        print(f"Generated Python code: {code_text}")

        return code_text
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def extract_code_from_response(response):
    """
    Extracts the Python code enclosed in triple backticks from the response.
    
    Args:
        response (str): The response text containing the Python code.
    
    Returns:
        str: Extracted Python code.
    """
    code_pattern = re.compile(r'```python(.*?)```', re.DOTALL)
    match = code_pattern.search(response)
    if match:
        return match.group(1).strip()
    return None


def execute_waypoints_code(code):
    """
    Executes the generated Python code to produce waypoints.
    
    Args:
        code (str): Python code for generating the waypoints.
    
    Returns:
        list: List of waypoints.
    """
    print('Executing Python code...')
    print(code)
    local_vars = {}
    exec(code, {}, local_vars)
    waypoints = local_vars.get('waypoints', [])
    return waypoints


def plot_3d_trajectory(waypoints, plot: bool = True, save_path: str = None):
    """
    Plots a 3D trajectory based on the given waypoints.
    
    Args:
        waypoints (list): List of waypoints where each waypoint is a list of [x, y, z].
        plot (bool): Whether to display the plot.
        save_path (str): Path to save the plot image, if provided.
    """
    waypoints = np.array(waypoints)
    x = waypoints[:, 0]
    y = waypoints[:, 1]
    z = waypoints[:, 2]

    # Create the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, label='3D trajectory')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    if save_path:
        plt.savefig(save_path)
        logging.info(f"trajectory path saved at {save_path}")


    if plot:
        plt.show()


def process_waypoints(code_response, plot=False, save_path=None):
    """
    Processes the code response to derive waypoints and optionally plot them.

    Args:
        code_response: The response containing the Python code.
        plot (bool): Whether to display the plot.
        save_path (str): Path to save the plot image, if provided.

    Returns:
        list: Derived waypoints or None if the process fails.
    """
    if not code_response:
        print("Failed to generate Python code.")
        return None

    # Extract the code from the response
    code = extract_code_from_response(code_response)
    if not code:
        print("Failed to extract Python code.")
        return None

    # Execute the extracted Python code to get the waypoints
    waypoints = execute_waypoints_code(code)
    if not waypoints:
        print("Failed to derive waypoints.")
        return None

    print(f"Derived waypoints: {waypoints}")

    # Plot the 3D trajectory based on the derived waypoints
    plot_3d_trajectory(waypoints, plot=plot, save_path=save_path)

    return waypoints


def main(file_path):
    """
    Main function to read the latest command and convert it into waypoints.
    
    Args:
        file_path (str): Path to the file containing the voice-to-text commands.
    """
    # Load the API key from environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("API key is not set. Please check your .env file.")

    # Initialize the OpenAI client
    client = openai.OpenAI(api_key=api_key)

    # Read the latest command from the file
    command = read_latest_command(file_path)
    if command:
        print(f"Processing command: {command}")

        # Fetch the Python code for generating waypoints using the OpenAI API
        code_response = fetch_waypoints_code(client, command)
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
    else:
        print("No commands found in the file.")


# Example usage
if __name__ == "__main__":
    FILE_PATH = "command.txt"
    main(FILE_PATH)
