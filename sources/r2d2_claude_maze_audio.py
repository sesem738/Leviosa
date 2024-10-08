import logging
import os
import time
from datetime import datetime
import numpy as np
import pybullet as p
import pybullet_data
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv
from langchain.chat_models import ChatAnthropic
from langchain.schema import HumanMessage, SystemMessage
import base64

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the API key from the environment variable
load_dotenv()
os.environ["ANTHROPIC_API_KEY"] = os.getenv('ANTHROPIC_API_KEY')

class ClaudeInterface:
    def __init__(self):
        self.model = ChatAnthropic(model_name="claude-3")

    def process_input(self, system_prompt: str, user_prompt: str, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=[
                {"type": "text", "text": user_prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encoded_image}"
                    }
                }
            ])
        ]

        response = self.model.invoke(messages)
        return response.content

    def get_llm_feedback(self, observations: str, rgb_image_path: str, depth_image_path: str, user_prompt: str) -> str:
        """ Sends observations and images to the Claude model and retrieves feedback. """
        start_time = time.perf_counter()
        system_prompt = f"""
        You are an AI assistant that analyzes observations from a robot in a maze simulation environment. Below are the observations from the environment along with an RGB image and a depth image. Provide comments about these observations.

        Observations: {observations}
        Attached Images: - RGB Image: {rgb_image_path} - Depth Image: {depth_image_path}
        """
        feedback = self.process_input(system_prompt, user_prompt, rgb_image_path)
        end_time = time.perf_counter()
        elapsed_time_ms = (end_time - start_time) * 1000
        logging.info(f"Time taken by Claude: {elapsed_time_ms:.2f} ms")
        return feedback

def save_image_with_grid(image: np.ndarray, path: str, maze_layout: np.ndarray) -> None:
    """ Save the image as a PNG file with a grid of numbers overlayed. """
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    cell_width = img.width // len(maze_layout[0])
    cell_height = img.height // len(maze_layout)
    for i, row in enumerate(maze_layout):
        for j, cell in enumerate(row):
            text = str(cell)
            text_x = j * cell_width + cell_width // 2
            text_y = i * cell_height + cell_height // 2
            draw.text((text_x, text_y), text, font=font, fill=(255, 0, 0))
    img.save(path)

def save_image(image: np.ndarray, path: str) -> None:
    """ Save the image as a PNG file. """
    img = Image.fromarray(image)
    img.save(path)

def save_depth_image(depth_buffer: np.ndarray, path: str) -> None:
    """ Save the depth image as a PNG file. """
    depth_normalized = (depth_buffer - np.min(depth_buffer)) / (np.max(depth_buffer) - np.min(depth_buffer))
    depth_image = (depth_normalized * 255).astype(np.uint8)
    img = Image.fromarray(depth_image)
    img.save(path)

# Connect to PyBullet simulator
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load plane URDF
plane_id = p.loadURDF("plane.urdf")

# Load robot URDF
robot_start_pos = [0.49812622931785777, -1.516623122138399, 0.470842044396454]
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

def create_maze_environment() -> np.ndarray:
    """ Creates a PyBullet simulation environment with a plane and a maze. """
    wall_thickness = 0.2
    wall_height = 1.0
    maze_layout = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1, 0, 1, 1, 0, 1],
        [1, 0, 1, 0, 0, 0, 0, 1, 0, 1],
        [1, 0, 1, 1, 1, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 1, 0, 1, 0, 1],
        [1, 1, 1, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ]
    numbered_maze_layout = np.zeros_like(maze_layout)
    counter = 1
    for i in range(len(maze_layout)):
        for j in range(len(maze_layout[0])):
            numbered_maze_layout[i, j] = counter
            counter += 1
    
    for i, row in enumerate(maze_layout):
        for j, cell in enumerate(row):
            if cell == 1:
                wall_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, wall_thickness / 2, wall_height / 2], rgbaColor=[0.7, 0.7, 0.7, 1])
                wall_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, wall_thickness / 2, wall_height / 2])
                p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_collision, baseVisualShapeIndex=wall_visual, basePosition=[j - len(row) / 2, i - len(maze_layout) / 2, wall_height / 2])
                
                wall_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[wall_thickness / 2, 0.5, wall_height / 2], rgbaColor=[0.7, 0.7, 0.7, 1])
                wall_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[wall_thickness / 2, 0.5, wall_height / 2])
                p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_collision, baseVisualShapeIndex=wall_visual, basePosition=[j - len(row) / 2, i - len(maze_layout) / 2, wall_height / 2])
    
    return numbered_maze_layout

# Create the maze environment
numbered_maze_layout = create_maze_environment()

# Directory to save images
image_dir = "r2d2_images"
os.makedirs(image_dir, exist_ok=True)

# Run the simulation for 1000 steps and get feedback periodically
overlay_grid = True  # Set to False to disable the grid overlay

# Initialize the Claude interface
claude = ClaudeInterface()

for i in range(1000):
    # Get keyboard events
    move_x, move_y, yaw_change = get_keyboard_events()

    # Get robot's current orientation
    pos, orn = p.getBasePositionAndOrientation(robot_id)
    euler = p.getEulerFromQuaternion(orn)
    new_yaw = euler[2] + yaw_change
    new_orientation = p.getQuaternionFromEuler([0, 0, new_yaw])

    # Calculate new position
    pos = [
        pos[0] + move_x * np.cos(new_yaw) - move_y * np.sin(new_yaw),
        pos[1] + move_x * np.sin(new_yaw) + move_y * np.cos(new_yaw),
        pos[2]
    ]

    # Move robot
    p.resetBasePositionAndOrientation(robot_id, pos, new_orientation)

    # Camera view matrix (third-person view)
    cam_target_pos = pos
    cam_distance = 10.0
    cam_yaw = new_yaw * 180.0 / np.pi
    cam_pitch = -60
    cam_roll = 0
    cam_up_axis_index = 2
    cam_fov = 60
    cam_aspect = 1
    cam_near = 0.1
    cam_far = 20.1

    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=cam_target_pos,
        distance=cam_distance,
        yaw=cam_yaw,
        pitch=cam_pitch,
        roll=cam_roll,
        upAxisIndex=cam_up_axis_index
    )
    projection_matrix = p.computeProjectionMatrixFOV(
        fov=cam_fov,
        aspect=cam_aspect,
        nearVal=cam_near,
        farVal=cam_far
    )

    # Get images from the simulation
    width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
        width=640,
        height=480,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix
    )

    # Convert image data to numpy arrays
    rgb_img = np.reshape(rgb_img, (height, width, 4))[:, :, :3]  # Remove alpha channel
    depth_img = np.reshape(depth_img, (height, width))

    # Save images every 100 steps
    if i % 100 == 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rgb_image_path = os.path.join(image_dir, f"r2d2_rgb_step_{i}_{timestamp}.png")
        depth_image_path = os.path.join(image_dir, f"r2d2_depth_step_{i}_{timestamp}.png")

        if overlay_grid:
            save_image_with_grid(rgb_img, rgb_image_path, numbered_maze_layout)
        else:
            save_image(rgb_img, rgb_image_path)
        save_depth_image(depth_img, depth_image_path)

        # Format observations
        observations = f"Position: {pos}, Orientation: {euler}"
        print(f"Observations at step {i}:\n{observations}")

        # Get user input for command
        user_prompt = input("Enter a command for the robot: ")

        # Get Claude feedback
        feedback = claude.get_llm_feedback(observations, rgb_image_path, depth_image_path, user_prompt)
        print(f"LLM Feedback at step {i}:\n{feedback}")

    # Step simulation
    p.stepSimulation()

    # Sleep to match the real-time simulation speed
    time.sleep(time_step)

# Disconnect from PyBullet
p.disconnect()
