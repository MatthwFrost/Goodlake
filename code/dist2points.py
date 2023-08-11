
import numpy as np
import matplotlib.pyplot as plt


sample_points = 10000
matrix = np.random.rand(sample_points, 3)
ref_point = np.array([0, 0, 0])

# Compute the distances
differences = matrix - ref_point
distances = np.linalg.norm(differences, axis=1)

# Identify indices of the maximum and minimum distances
max_index = np.argmax(distances)
min_index = np.argmin(distances)

sorted_indices = np.argsort(distances)
N = 100
closest_points = matrix[sorted_indices[:N]]

# Set up the figure and axis for 3D plotting
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the points from the matrix
ax.scatter(matrix[:, 0], matrix[:, 1], matrix[:, 2], c='blue', s=1, label='Points in Matrix')

# Highlight the reference point in red
ax.scatter(ref_point[0], ref_point[1], ref_point[2], c='red', s=50, label='Reference Point')

# Highlight the point with the maximum distance in magenta
ax.scatter(matrix[max_index, 0], matrix[max_index, 1], matrix[max_index, 2], c='magenta', s=50, label='Farthest Point')

# Highlight the point with the minimum distance in yellow
ax.scatter(matrix[min_index, 0], matrix[min_index, 1], matrix[min_index, 2], c='yellow', s=50, label='Closest Point')

ax.scatter(closest_points[:, 0], closest_points[:,1], closest_points[:, 2], c='green', s=50, label='Closest points')

# Set labels for axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

plt.show()

