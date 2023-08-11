import numpy as np

sample_points = 100000
matrix = np.random.rand(sample_points,3)
ref_point = np.array([0, 0, 0])

# Compute the distances
differences = matrix - ref_point
distances = np.linalg.norm(differences, axis=1)

# Get the top N closest points (e.g., top 10)
sorted_indices = np.argsort(distances)
N = 100
closest_points = matrix[sorted_indices[:N]]
print(closest_points)

# Identify indices of the maximum and minimum distances
max_index = np.argmax(distances)
min_index = np.argmin(distances)


print(max_index)
print(min_index)

# 0.519s to run on 14,000 points
