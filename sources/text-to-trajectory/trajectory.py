import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import openai
from dotenv import load_dotenv
import os
import time

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

def fetch_polynomial_coefficients(client, command):
    """
    Uses the OpenAI API in chat mode to convert a text command into polynomial coefficients.
    
    Args:
        client (OpenAI): OpenAI client initialized with the API key.
        command (str): Text command describing the desired trajectory.
    
    Returns:
        dict: Coefficients of the polynomials if successful, None otherwise.
    """
    try:
        # Start timing the API call
        start_time = time.time()

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": """
                 You are an AI assistant that converts text commands into 8th order polynomial coefficients used for drone trajectories.
                 You will need to generate 3 sets of coefficients, one for each axis.
                 When you receive a user prompt reason step-by-step.
                 Assume the unit of measurement is meters.
                 The coefficients you provide should be for 8th order polynomials that can be plotted in 3D to form a single trajectory desired by the user. 
                 To make your output easy to parse for further processing, please provide the coefficients as a comma-separated list of floats, each list is is for one axis and surrounded by square brackets [] like this: 

                    x_coeff : [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
                    y_coeff : [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
                    z_coeff : [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
                 
                 Make sure that the coefficients are in the correct order and that you provide exactly 8 coefficients for each axis, no less and no more.
                 Also, never use a variable name that is not explicitly mentioned in the prompt. For exmaple, the coefficients below are invalid:

                 z_coeff : [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, current_altitude]

                 Always always always! use floating point numbers for the coefficients. If the coefficients represents a fixed altitude, always assume the altitude is 2.
                 """},
                {"role": "user", "content": f"Convert the following command into an 8th order polynomial coefficients: '{command}'"}
            ]
        )

        # End timing the API call
        end_time = time.time()
        duration = end_time - start_time
        print(f"Time taken for API call: {duration:.2f} seconds")

        # Extract and print the response
        coefficients_text = response.choices[0].message.content
        print(f"Polynomial coefficients text: {coefficients_text}")

        # Parse the coefficients text
        x_coeff = coefficients_text.split('x_coeff : ')[1].split(']')[0][1:]
        y_coeff = coefficients_text.split('y_coeff : ')[1].split(']')[0][1:]
        z_coeff = coefficients_text.split('z_coeff : ')[1].split(']')[0][1:]

        x_coeff = list(map(float, x_coeff.split(',')))
        y_coeff = list(map(float, y_coeff.split(',')))
        z_coeff = list(map(float, z_coeff.split(',')))

        return {'x': x_coeff, 'y': y_coeff, 'z': z_coeff}
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def plot_3d_trajectory(coefficients):
    """
    Plots a 3D trajectory based on the given polynomial coefficients.
    
    Args:
        coefficients (dict): Dictionary containing lists of polynomial coefficients for x, y, and z.
    """
    # Define the parameter t
    t = np.linspace(-10, 10, 1000)

    # Extract coefficients for x, y, and z
    x_coeff = coefficients['x']
    y_coeff = coefficients['y']
    z_coeff = coefficients['z']

    # Calculate x, y, z values based on the polynomial coefficients
    x = sum(c * t**i for i, c in enumerate(x_coeff))
    y = sum(c * t**i for i, c in enumerate(y_coeff))
    z = sum(c * t**i for i, c in enumerate(z_coeff))

    # Create the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, label='3D trajectory')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

def main(file_path):
    """
    Main function to read the latest command and convert it into polynomial coefficients.
    
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

        # Fetch the polynomial coefficients using the OpenAI API
        coefficients = fetch_polynomial_coefficients(client, command)
        if coefficients:
            print(f"Derived coefficients: {coefficients}")

            # Plot the 3D trajectory based on the derived coefficients
            plot_3d_trajectory(coefficients)
        else:
            print("Failed to derive coefficients.")
    else:
        print("No commands found in the file.")

# Example usage
if __name__ == "__main__":
    FILE_PATH = "command.txt"
    main(FILE_PATH)

