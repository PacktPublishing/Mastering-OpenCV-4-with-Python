"""
Training a linear regression model using TensorFlow
"""

# Import required packages:
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Path to the folder that we want to save the logs for Tensorboard:
logs_path = "./logs"
# Number of points:
N = 50

# Make random numbers predictable:
np.random.seed(101)
tf.set_random_seed(101)

# Generate random data composed by 50 (N = 50) points:
x = np.linspace(0, N, N)
y = 3 * np.linspace(0, N, N) + np.random.uniform(-10, 10, N)

# You can check the shape of the created training data:
print(x.shape)
print(y.shape)

# Create the placeholders in order to feed our training examples into the optimizer while training:
X = tf.placeholder("float", name='X')
Y = tf.placeholder("float", name='Y')

# Declare two trainable TensorFlow Variables for the Weights and Bias
# We are going to initialize them randomly. Another way can be to set '0.0':
W = tf.Variable(np.random.randn(), name="W")
b = tf.Variable(np.random.randn(), name="b")

# Define the hyperparameters of the model:
learning_rate = 0.01
training_epochs = 1000

# This will be used to show results after every 25 epochs:
disp_step = 100

# Construct a linear model:
y_model = tf.add(tf.multiply(X, W), b, name="y_model")

# Define cost function, in this case, the Mean squared error
# (Note that other cost functions can be defined)
cost = tf.reduce_sum(tf.pow(y_model - Y, 2)) / (2 * N)

# Create the gradient descent optimizer that is going to minimize the cost function modifying the
# values of the variables W and b:
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize all variables:
init = tf.global_variables_initializer()

# Create a Saver object:
saver = tf.train.Saver()

# Start the training procedure inside a TensorFlow Session:
with tf.Session() as sess:
    # Run the initializer:
    sess.run(init)

    # Uncomment if you want to see the created graph
    # summary_writer = tf.summary.FileWriter(logs_path, sess.graph)

    # Iterate over all defined epochs:
    for epoch in range(training_epochs):

        # Feed each training data point into the optimizer:
        for (_x, _y) in zip(x, y):
            sess.run(optimizer, feed_dict={X: _x, Y: _y})

        # Display the results every 'display_step' epochs:
        if (epoch + 1) % disp_step == 0:
            # Calculate the actual cost, W and b:
            c = sess.run(cost, feed_dict={X: x, Y: y})
            w_est = sess.run(W)
            b_est = sess.run(b)
            print("epoch {}: cost = {} W = {}  b = {}".format(epoch + 1, c, w_est, b_est))

    # Save the final model
    saver.save(sess, './linear_regression')

    # Storing necessary values to be used outside the session
    training_cost = sess.run(cost, feed_dict={X: x, Y: y})
    weight = sess.run(W)
    bias = sess.run(b)

print("Training finished!")

# Calculate the predictions:
predictions = weight * x + bias

# Create the dimensions of the figure and set title:
fig = plt.figure(figsize=(8, 5))
plt.suptitle("Linear regression using TensorFlow", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

# Plot training data:
plt.subplot(1, 2, 1)
plt.plot(x, y, 'ro', label='Original data')
plt.xlabel('x')
plt.ylabel('y')
plt.title("Training Data")

# Plot results:
plt.subplot(1, 2, 2)
plt.plot(x, y, 'ro', label='Original data')
plt.plot(x, predictions, label='Fitted line')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression Result')
plt.legend()

# Show the Figure:
plt.show()
