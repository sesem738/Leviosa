import pybullet as p
import pybullet_data
import time

# Connect to PyBullet
client_id = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load environment
plane_id = p.loadURDF("plane.urdf")
robot_id = p.loadURDF("r2d2.urdf", [0, 0, 0.5])

# Load some arbitrary objects
cube_id = p.loadURDF("cube.urdf", [1, 0, 0.5])

# Step simulation to initialize
p.stepSimulation()

time.sleep(1/240)

# Function to check for collisions
def check_collisions(agent_id):
    contact_points = p.getContactPoints(bodyA=agent_id)
    if len(contact_points) > 0:
        print("Collision detected!")
        for point in contact_points:
            print(f"Contact with body {point[2]} at link {point[4]}")
    else:
        print("No collision detected.")

# Check for collisions
check_collisions(robot_id)

# Disconnect from PyBullet
p.disconnect()
