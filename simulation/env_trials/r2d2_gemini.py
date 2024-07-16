import pybullet as p
import pybullet_data
import time
import numpy as np
import random
import os
import logging
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the API key from the environment variable
load_dotenv()
GOOGLE_API_KEY = os.getenv('gemini_api_key')
if not GOOGLE_API_KEY:
    logging.error("Google API key not found. Please set the GOOGLE_API_KEY environment variable.")
    exit(1)

genai.configure(api_key=GOOGLE_API_KEY)


def get_llm_feedback(observations: str, rgb_image_path: str, depth_image_path: str) -> str:
    """
    Sends observations, an RGB image, and a depth image to the Gemini model and retrieves feedback.

    Args:
        observations (str): A string representation of the observations.
        rgb_image_path (str): Path to the RGB image file.
        depth_image_path (str): Path to the depth image file.

    Returns:
        str: The feedback from the Gemini model.
    """
    # Upload the image files
    rgb_image = genai.upload_file(path=rgb_image_path)
    depth_image = genai.upload_file(path=depth_image_path)

    prompt = f"""
    You are an AI assistant that analyzes observations from a robot in a forest simulation environment.
    Below are the observations from the environment along with an RGB image and a depth image. Provide comments about these observations.

    Observations:
    {observations}

    Attached Images:
    - RGB Image: {rgb_image_path}
    - Depth Image: {depth_image_path}

    Please provide your feedback based on the attached images.
    """

    model = genai.GenerativeModel('models/gemini-1.5-flash')
    response = model.generate_content([prompt, rgb_image, depth_image])
    feedback = response.text if response else "No response from the model."

    return feedback


def save_image(image: np.ndarray, path: str) -> None:
    """
    Save the image as a PNG file.

    Args:
        image (np.ndarray): The image data.
        path (str): The path where the image will be saved.
    """
    img = Image.fromarray(image)
    img.save(path)


# Connect to PyBullet simulator
p.connect(p.GUI)

# Set the path to the PyBullet data
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load plane URDF
plane_id = p.loadURDF("plane.urdf")

# Load robot URDF
robot_start_pos = [0, 0, 0.1]
robot_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
robot_id = p.loadURDF("r2d2.urdf", robot_start_pos, robot_start_orientation)

# Set gravity
p.setGravity(0, 0, -9.8)

# Simulation step parameters
time_step = 1. / 240.

# Keyboard control parameters
move_speed = 0.2
rotate_speed = 0.1


def get_keyboard_events():
    keys = p.getKeyboardEvents()
    move_x, move_y = 0, 0
    yaw_change = 0

    if p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN:
        move_x = -move_speed
    if p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN:
        move_x = move_speed
    if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN:
        move_y = move_speed
    if p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN:
        move_y = -move_speed
    if ord('q') in keys and keys[ord('q')] & p.KEY_IS_DOWN:
        yaw_change = rotate_speed
    if ord('e') in keys and keys[ord('e')] & p.KEY_IS_DOWN:
        yaw_change = -rotate_speed

    return move_x, move_y, yaw_change


def create_forest_environment(num_trees: int) -> None:
    """
    Creates a PyBullet simulation environment with a plane and randomly placed cuboid trees.

    Args:
        num_trees (int): The number of trees to generate in the forest.
    """
    tree_width = 0.5
    tree_height_min = 2
    tree_height_max = 5

    for _ in range(num_trees):
        x = random.uniform(-10, 10)
        y = random.uniform(-10, 10)
        tree_height = random.uniform(tree_height_min, tree_height_max)

        tree_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[tree_width / 2, tree_width / 2, tree_height / 2],
                                          rgbaColor=[0.4, 0.2, 0, 1])
        tree_collision = p.createCollisionShape(p.GEOM_BOX,
                                                halfExtents=[tree_width / 2, tree_width / 2, tree_height / 2])
        p.createMultiBody(baseMass=0,  # Make trees static (or adjust mass)
                          baseCollisionShapeIndex=tree_collision,
                          baseVisualShapeIndex=tree_visual,
                          basePosition=[x, y, tree_height / 2])


# Create the forest environment
create_forest_environment(50)

# Directory to save images
image_dir = "r2d2_images"
os.makedirs(image_dir, exist_ok=True)

# Run the simulation for 1000 steps and get feedback periodically
for i in range(1000):
    # Get keyboard events
    move_x, move_y, yaw_change = get_keyboard_events()

    # Get robot's current orientation
    pos, orn = p.getBasePositionAndOrientation(robot_id)
    euler = p.getEulerFromQuaternion(orn)

    new_yaw = euler[2] + yaw_change
    new_orientation = p.getQuaternionFromEuler([0, 0, new_yaw])

    # Calculate new position
    pos = [pos[0] + move_x * np.cos(new_yaw) - move_y * np.sin(new_yaw),
           pos[1] + move_x * np.sin(new_yaw) + move_y * np.cos(new_yaw),
           pos[2]]

    # Move robot
    p.resetBasePositionAndOrientation(robot_id, pos, new_orientation)

    # Camera view matrix (third-person view)
    cam_target_pos = pos
    cam_distance = 5.0  # Increase distance for a wider view
    cam_yaw = new_yaw * 180.0 / np.pi
    cam_pitch = -45  # Increase pitch for a higher view
    cam_roll = 0
    cam_up_axis_index = 2
    cam_fov = 60
    cam_aspect = 1
    cam_near = 0.1
    cam_far = 10.1

    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=cam_target_pos,
        distance=cam_distance,
        yaw=cam_yaw,
        pitch=cam_pitch,
        roll=cam_roll,
        upAxisIndex=cam_up_axis_index)

    projection_matrix = p.computeProjectionMatrixFOV(
        fov=cam_fov,
        aspect=cam_aspect,
        nearVal=cam_near,
        farVal=cam_far)

    # Get images from the simulation
    width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
        width=640,
        height=480,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix)

    # Convert image data to numpy arrays
    rgb_img = np.reshape(rgb_img, (height, width, 4))[:, :, :3]  # Remove alpha channel
    depth_img = np.reshape(depth_img, (height, width))

    # Save images every 100 steps
    if i % 100 == 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rgb_image_path = os.path.join(image_dir, f"r2d2_rgb_step_{i}_{timestamp}.png")
        depth_image_path = os.path.join(image_dir, f"r2d2_depth_step_{i}_{timestamp}.png")
        save_image(rgb_img, rgb_image_path)
        save_image((depth_img * 255).astype(np.uint8), depth_image_path)  # Save depth as grayscale image

        # Format observations
        observations = f"Position: {pos}, Orientation: {euler}"
        print(f"Observations at step {i}:\n{observations}")

        # Get LLM feedback
        feedback = get_llm_feedback(observations, rgb_image_path, depth_image_path)
        print(f"LLM Feedback at step {i}:\n{feedback}")

    # Step simulation
    p.stepSimulation()

    # Sleep to match the real-time simulation speed
    time.sleep(time_step)

# Disconnect from PyBullet
p.disconnect()
