"""
K-means clustering algorithm applied to three different 'clusters' of points (k=4)
"""

# Import required packages:
import numpy as np
import cv2
from matplotlib import pyplot as plt

# Create data (three different 'clusters' of points (it should be of np.float32 data type):
data = np.float32(np.vstack(
    (np.random.randint(0, 40, (50, 2)), np.random.randint(30, 70, (50, 2)), np.random.randint(60, 100, (50, 2)))))

# Define the algorithm termination criteria (the maximum number of iterations and/or the desired accuracy):
# In this case the maximum number of iterations is set to 20 and epsilon = 1.0
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)

# Apply kmeans algorithm
# K = 3, which is the number of clusters to split the set by
# attempts = 10, which specifies the number of times the algorithm is executed using different initial labellings (the
# algorithm returns the labels that yield the best compactness)
# Flag cv2.KMEANS_RANDOM_CENTERS selects random initial centers in each attempt. You can also use cv2.KMEANS_PP_CENTERS
ret, label, center = cv2.kmeans(data, 4, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Now separate the data using label output (stores the cluster indices for every sample)
# Therefore, we split the data to different clusters depending on their labels:
A = data[label.ravel() == 0]
B = data[label.ravel() == 1]
C = data[label.ravel() == 2]
D = data[label.ravel() == 3]

# Create the dimensions of the figure and set title:
fig = plt.figure(figsize=(12, 6))
plt.suptitle("K-means clustering algorithm", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

# Plot the 'original' data:
ax = plt.subplot(1, 2, 1)
plt.scatter(data[:, 0], data[:, 1], c='c')
plt.title("data")

# Plot the 'clustered' data and the centroids
ax = plt.subplot(1, 2, 2)
plt.scatter(A[:, 0], A[:, 1], c='b')
plt.scatter(B[:, 0], B[:, 1], c='g')
plt.scatter(C[:, 0], C[:, 1], c='r')
plt.scatter(D[:, 0], D[:, 1], c='y')
plt.scatter(center[:, 0], center[:, 1], s=100, c='m', marker='s')
plt.title("clustered data and centroids (K = 4)")

# Show the Figure:
plt.show()
