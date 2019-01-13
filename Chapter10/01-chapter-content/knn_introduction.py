"""
Simple introduction to k-Nearest Neighbour (k-NN) algorithm with OpenCV
"""

# Import required packages:
import cv2
import numpy as np
import matplotlib.pyplot as plt

# The data is composed of 16 points:
data = np.random.randint(0, 100, (16, 2)).astype(np.float32)

# We create the labels (0: red, 1: blue) for each of the 16 points:
labels = np.random.randint(0, 2, (16, 1)).astype(np.float32)

# Create the sample point to be classified:
sample = np.random.randint(0, 100, (1, 2)).astype(np.float32)

# k-NN creation:
knn = cv2.ml.KNearest_create()
# k-NN training:
knn.train(data, cv2.ml.ROW_SAMPLE, labels)
# k-NN find nearest:
k = 3
ret, results, neighbours, dist = knn.findNearest(sample, k)

# Plot all the data and print the results:
# Create the dimensions of the figure:
fig = plt.figure(figsize=(8, 6))
fig.patch.set_facecolor('silver')
# Take points with label 0 (will be the red triangles) and plot them:
red_triangles = data[labels.ravel() == 0]
plt.scatter(red_triangles[:, 0], red_triangles[:, 1], 200, 'r', '^')

# Take points with label 1 (will be the blue squares) and plot them:
blue_squares = data[labels.ravel() == 1]
plt.scatter(blue_squares[:, 0], blue_squares[:, 1], 200, 'b', 's')

# Plot the sample point:
plt.scatter(sample[:, 0], sample[:, 1], 200, 'g', 'o')

# Print results:
print("result: {}".format(results))
print("neighbours: {}".format(neighbours))
print("distance: {}".format(dist))

# Set the title:
if results[0][0] > 0:
    plt.suptitle("k-NN algorithm: sample green point is classified as blue (k = " + str(k) + ")", fontsize=14,
                 fontweight='bold')
else:
    plt.suptitle("k-NN algorithm: sample green point is classified as red (k = " + str(k) + ")", fontsize=14,
                 fontweight='bold')

# Show the results:
plt.show()
