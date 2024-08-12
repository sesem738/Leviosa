import logging
import os
import re
import time

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
        list: List of waypoints if successful, otherwise None.
    """
    print('Executing Python code...')
    print(code)

    # Prepare the local variables and import necessary modules
    local_vars = {}

    try:
        # Execute the code in a safe environment with limited built-ins
        exec(code, {'np': np}, local_vars)

        # Retrieve the waypoints from the local variables
        waypoints = local_vars.get('waypoints', None)

        # Validate the waypoints
        if not isinstance(waypoints, list):
            raise ValueError("The code did not produce a list of waypoints.")

        return waypoints

    except Exception as e:
        # Print the exception for debugging purposes
        print(f"An error occurred during execution: {e}")
        return None


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


# def plot_multi_drone_3d_trajectory(waypoints_list, plot=True, save_path=None):
#     """
#     Plots 3D trajectories for multiple drones based on the given waypoints.
#
#     Args:
#         waypoints_list (list): List of waypoint lists for each drone.
#         plot (bool): Whether to display the plot.
#         save_path (str): Path to save the plot image, if provided.
#     """
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d')
#
#     # Generate a color map for the number of drones
#     colors = plt.cm.rainbow(np.linspace(0, 1, len(waypoints_list)))
#
#     for i, waypoints in enumerate(waypoints_list):
#         waypoints = np.array(waypoints)
#         x = waypoints[:, 0]
#         y = waypoints[:, 1]
#         z = waypoints[:, 2]
#
#         ax.plot(x, y, z, color=colors[i], label=f'Drone {i + 1}')
#         ax.scatter(x[0], y[0], z[0], color=colors[i], s=100, marker='o')
#         ax.text(x[0], y[0], z[0], f'Start {i + 1}')
#
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     ax.legend()
#     plt.title('Multi-Drone Trajectories')
#
#     if save_path:
#         plt.savefig(save_path)
#         logging.info(f"Trajectory plot saved at {save_path}")
#
#     if plot:
#         plt.show()
#
#     plt.close()

def plot_multi_drone_3d_trajectory(waypoints_list, plot=True, save_path=None):
    """
    Plots 3D trajectories for multiple drones based on the given waypoints.

    Args:
        waypoints_list (list): List of waypoint lists for each drone.
        plot (bool): Whether to display the plot.
        save_path (str): Path to save the plot image, if provided.
    """
    fig = plt.figure(figsize=(20, 8))

    # Subplot 1: Current perspective
    ax1 = fig.add_subplot(131, projection='3d')
    # Subplot 2: Top-down view
    ax2 = fig.add_subplot(132, projection='3d')
    # Subplot 3: Side view
    ax3 = fig.add_subplot(133, projection='3d')

    # Generate a color map for the number of drones
    colors = plt.cm.rainbow(np.linspace(0, 1, len(waypoints_list)))

    for i, waypoints in enumerate(waypoints_list):
        waypoints = np.array(waypoints)
        x = waypoints[:, 0]
        y = waypoints[:, 1]
        z = waypoints[:, 2]

        # Plotting on the first subplot (current perspective)
        ax1.plot(x, y, z, color=colors[i], label=f'Drone {i + 1}')
        ax1.scatter(x[0], y[0], z[0], color=colors[i], s=100, marker='o')
        ax1.text(x[0], y[0], z[0], f'Start {i + 1}')

        # Plotting on the second subplot (top-down view)
        ax2.plot(x, y, z, color=colors[i], label=f'Drone {i + 1}')
        ax2.scatter(x[0], y[0], z[0], color=colors[i], s=100, marker='o')
        ax2.text(x[0], y[0], z[0], f'Start {i + 1}')
        ax2.view_init(elev=90, azim=-90)  # Top-down view

        # Plotting on the third subplot (side view)
        ax3.plot(x, y, z, color=colors[i], label=f'Drone {i + 1}')
        ax3.scatter(x[0], y[0], z[0], color=colors[i], s=100, marker='o')
        ax3.text(x[0], y[0], z[0], f'Start {i + 1}')
        ax3.view_init(elev=0, azim=0)  # Side view

    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()

    ax1.set_title('Perspective View')
    ax2.set_title('Top-Down View')
    ax3.set_title('Side View')

    plt.suptitle('Multi-Drone Trajectories from Different Perspectives')

    if save_path:
        plt.savefig(save_path)
        logging.info(f"Trajectory plot saved at {save_path}")

    if plot:
        plt.show()

    plt.close()


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
    # print(f"Extracted Python code: {code}") # extracted code is correct
    if not code:
        print("Failed to extract Python code.")
        return None

    # Execute the extracted Python code to get the waypoints
    waypoints_list = execute_waypoints_code(code)
    if not waypoints_list:
        print("Failed to derive waypoints.")
        return None

    # print(f"Derived waypoints: {waypoints_list}")

    # # Plot the 3D trajectory based on the derived waypoints
    # plot_3d_trajectory(waypoints, plot=plot, save_path=save_path)

    print(f"Derived waypoints for {len(waypoints_list)} drones:")
    for i, waypoints in enumerate(waypoints_list):
        print(f"Drone {i + 1}: {waypoints}")

    # Plot the 3D trajectories based on the derived waypoints
    plot_multi_drone_3d_trajectory(waypoints_list, plot=plot, save_path=save_path)

    return waypoints_list


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
