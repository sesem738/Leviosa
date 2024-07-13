import pybullet as p
import pybullet_data
import time
import numpy as np

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
time_step = 1./240.

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

# Run the simulation for 1000 steps
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
    cam_distance = 2.0
    cam_yaw = new_yaw * 180.0 / np.pi
    cam_pitch = -30
    cam_roll = 0
    cam_up_axis_index = 2
    cam_fov = 60
    cam_aspect = 1
    cam_near = 0.1
    cam_far = 3.1

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

    # Display the images (not necessary but useful for debugging)
    # cv2.imshow('RGB Image', np.reshape(rgb_img, (height, width, -1)))

    # Step simulation
    p.stepSimulation()

    # Sleep to match the real-time simulation speed
    time.sleep(time_step)

# Disconnect from PyBullet
p.disconnect()
