"""
Training Softmax regression model using MNIST Dataset and saving the created model in TensorFlow
"""

# Load required packages:
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# Delete the current graph
tf.reset_default_graph()

# Load data:
data = input_data.read_data_sets("MNIST/", one_hot=True)

print("Size of Training set: {}".format(len(data.train.labels)))
print("Size of Test set: {}".format(len(data.test.labels)))
print("Size of Validation set: {}".format(len(data.validation.labels)))

# Set hyperparameter values for learning rate, batch size and the total number of training steps:
learning_rate = 0.001
batch_size = 100
num_steps = 1000

# Define placeholders:
x = tf.placeholder(tf.float32, shape=[None, 784], name='myInput')
y = tf.placeholder(tf.float32, shape=[None, 10], name='Y')

# Define W and b variables:
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Calculate the output logits and convert logits to probabilities:
output_logits = tf.matmul(x, W) + b
y_pred = tf.nn.softmax(output_logits, name='myOutput')

# Define the loss function, optimizer, and accuracy
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=output_logits), name='loss')
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='Adam-op').minimize(loss)
correct_prediction = tf.equal(tf.argmax(output_logits, 1), tf.argmax(y, 1), name='correct_pred')
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

# Create saver object:
saver = tf.train.Saver()

# Run the session:
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(num_steps):
        # Get a batch of training examples and their corresponding labels.
        x_batch, y_true_batch = data.train.next_batch(batch_size)

        # Put the batch into a dict to be fed into the placeholders
        feed_dict_train = {x: x_batch, y: y_true_batch}
        sess.run(optimizer, feed_dict=feed_dict_train)

    # Validation:
    feed_dict_validation = {x: data.validation.images, y: data.validation.labels}
    loss_test, acc_test = sess.run([loss, accuracy], feed_dict=feed_dict_validation)
    print("Validation loss: {}, Validation accuracy: {}".format(loss_test, acc_test))

    # Save model:
    saved_path_model = saver.save(sess, './softmax_regression_model_mnist')
    print('Model has been saved in {}'.format(saved_path_model))
