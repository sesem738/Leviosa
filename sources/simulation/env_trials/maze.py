import pybullet as p
import pybullet_data
import time

# Start the PyBullet physics engine in GUI mode
p.connect(p.GUI)

# Set the search path to find standard models in PyBullet
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load the SDF file
sdf_path = '/sources/simulation/env_trials/maze.sdf'  # Update this path to your actual path
p.loadSDF(sdf_path)

# Optionally, set the gravity in the simulation
p.setGravity(0, 0, -9.8)

# Let the simulation run for a while
for _ in range(240):
    p.stepSimulation()
    time.sleep(1./240.)

# Disconnect from the PyBullet server
p.disconnect()
