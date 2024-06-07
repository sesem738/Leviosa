import numpy as np
import matplotlib.pyplot as plt

"""
Prompt:

Create an 8th order polynomial of a spiral with 3 loops, in 3D. 
This is the trajectory of a drone. So the drone will follow this curve as a flight trajectory. Write the python code for plotting this trajectory
"""

# Parameters
loops = 3
points_per_loop = 1000

# Generate theta values
theta = np.linspace(0, 2 * np.pi * loops, loops * points_per_loop)

# Spiral equation: r = a + b*theta
# Here, `a` is the initial radius, `b` controls how tight the spiral is
a = 0
b = 0.1

# Generate r values
r = a + b * theta

# Convert polar coordinates (r, theta) to Cartesian coordinates (x, y)
x = r * np.cos(theta)
y = r * np.sin(theta)

# Generate z values (linearly increasing)
z = np.linspace(0, 10, loops * points_per_loop)


# The X Y Z produce algebraic expressions for waypoints, not 8th order
waypoints = np.asarray([x, y, z])
waypoints = waypoints.T

# 3D Plot
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z)
ax.set_title('3D Spiral Trajectory with 3 Loops')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
