"""
K-means clustering data visualization (introduction to k-means clustering algorithm)
"""

# Import required packages:
import numpy as np
import cv2
from matplotlib import pyplot as plt

# Create data (three different 'clusters' of points (it should be of np.float32 data type):
data = np.float32(np.vstack(
    (np.random.randint(0, 40, (50, 2)), np.random.randint(30, 70, (50, 2)), np.random.randint(60, 100, (50, 2)))))

# Create the dimensions of the figure and set title:
fig = plt.figure(figsize=(6, 6))
plt.suptitle("K-means clustering algorithm", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

# Plot the 'original' data:
ax = plt.subplot(1, 1, 1)
plt.scatter(data[:, 0], data[:, 1], c='c')
plt.title("data to be clustered")

# Show the Figure:
plt.show()
