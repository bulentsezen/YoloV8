import numpy as np

points = np.array([[1, 2], [4, 5]])
print("points :", points)

reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
print("3-d reshaped points:", reshaped_points)

transformed_points = reshaped_points.reshape(-1, 2)
print("2-d points:", transformed_points)