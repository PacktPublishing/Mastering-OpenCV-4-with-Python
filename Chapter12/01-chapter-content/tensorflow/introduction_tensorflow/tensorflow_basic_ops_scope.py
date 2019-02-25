"""
Basic Operations example using TensorFlow library creating scopes
"""

# Import required packages:
import tensorflow as tf

# Path to the folder that we want to save the logs for Tensorboard:
logs_path = "./logs"

# Define placeholders:
X_1 = tf.placeholder(tf.int16, name="X_1")
X_2 = tf.placeholder(tf.int16, name="X_2")

# Define two operations encapsulating the operations into a scope making
# the Tensorboard's Graph visualization more convenient:
with tf.name_scope('Operations'):
    addition = tf.add(X_1, X_2, name="my_addition")
    multiply = tf.multiply(X_1, X_2, name="my_multiplication")

# Start the session and run the operations with different inputs:
with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter(logs_path, sess.graph)

    # Perform some multiplications:
    print("2 x 3 = {}".format(sess.run(multiply, feed_dict={X_1: 2, X_2: 3})))
    print("[2, 3] x [3, 4] = {}".format(sess.run(multiply, feed_dict={X_1: [2, 3], X_2: [3, 4]})))

    # Perform some additions:
    print("2 + 3 = {}".format(sess.run(addition, feed_dict={X_1: 2, X_2: 3})))
    print("[2, 3] + [3, 4] = {}".format(sess.run(addition, feed_dict={X_1: [2, 3], X_2: [3, 4]})))
