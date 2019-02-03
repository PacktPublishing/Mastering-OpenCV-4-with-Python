"""
Basic Operations example using TensorFlow library.
"""

# Import required packages:
import tensorflow as tf

# path to the folder that we want to save the logs for Tensorboard
logs_path = "./logs"

# TensorFlow version: 1.12.0
print("tensorflow version: {}".format(tf.__version__))

# Define placeholders:
X_1 = tf.placeholder(tf.int16, name="X_1")
X_2 = tf.placeholder(tf.int16, name="X_2")

# Define a multiplication operation:
multiply = tf.multiply(X_1, X_2, name="my_multiplication")

# Start the session and run the operation with different inputs:
with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter(logs_path, sess.graph)

    print("2 x 3 = {}".format(sess.run(multiply, feed_dict={X_1: 2, X_2: 3})))
    print("[2, 3] x [3, 4] = {}".format(sess.run(multiply, feed_dict={X_1: [2, 3], X_2: [3, 4]})))
