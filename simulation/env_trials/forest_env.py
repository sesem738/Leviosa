import pybullet as p
import time
import pybullet_data
import random

def create_forest_environment(num_trees):
    """
    Creates a PyBullet simulation environment with a plane and randomly placed cuboid trees.

    Args:
        num_trees (int): The number of trees to generate in the forest.

    Returns:
        int: The physics client ID for the created simulation.
    """

    # Connect to the physics server
    physicsClient = p.connect(p.GUI)  # Use p.DIRECT for non-graphical version

    # Set up basic environment properties
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")

    # Define tree properties (adjust as needed)
    tree_width = 0.5
    tree_height_min = 2
    tree_height_max = 5

    # Create trees randomly within a specified area
    for _ in range(num_trees):
        # Randomize tree position (adjust area as needed)
        x = random.uniform(-10, 10) 
        y = random.uniform(-10, 10)
        tree_height = random.uniform(tree_height_min, tree_height_max)

        # Create a cuboid as a tree
        tree_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[tree_width/2, tree_width/2, tree_height/2], rgbaColor=[0.4, 0.2, 0, 1]) # Brown color
        tree_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[tree_width/2, tree_width/2, tree_height/2]) 
        tree_id = p.createMultiBody(baseMass=0,  # Make trees static (or adjust mass)
                                   baseCollisionShapeIndex=tree_collision,
                                   baseVisualShapeIndex=tree_visual,
                                   basePosition=[x, y, tree_height/2]) 

    return physicsClient

# Example usage: Create a forest with 50 trees
physics_client_id = create_forest_environment(50)

# Keep the simulation running (or add other actions)
while True:
    p.stepSimulation(physics_client_id) 
    time.sleep(1/240.) 