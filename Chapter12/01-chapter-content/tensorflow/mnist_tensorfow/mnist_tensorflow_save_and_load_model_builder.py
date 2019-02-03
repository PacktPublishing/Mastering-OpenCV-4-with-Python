"""
Testing Softmax regression model using 'OpenCV images' in TensorFlow
"""

# Import required packages:
import cv2
import tensorflow as tf
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
import numpy as np


# Note: Images should have black background:
def load_digit(image_name):
    """Loads a digit and pre-process in order to have the proper format"""

    gray = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    gray = cv2.resize(gray, (28, 28))
    flatten = gray.flatten() / 255.0
    return flatten


def export_model():
    """Exports the model"""

    trained_checkpoint_prefix = 'softmax_regression_model_mnist'

    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        sess.run(tf.global_variables_initializer())

        # Restore from checkpoint
        loader = tf.train.import_meta_graph(trained_checkpoint_prefix + '.meta')
        loader.restore(sess, trained_checkpoint_prefix)

        # Add signature:
        graph = tf.get_default_graph()
        inputs = tf.saved_model.utils.build_tensor_info(graph.get_tensor_by_name('myInput:0'))
        outputs = tf.saved_model.utils.build_tensor_info(graph.get_tensor_by_name('myOutput:0'))

        signature = signature_def_utils.build_signature_def(inputs={'myInput': inputs},
                                                            outputs={'myOutput': outputs},
                                                            method_name=signature_constants.PREDICT_METHOD_NAME)

        signature_map = {signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature}

        # Export model:
        builder = tf.saved_model.builder.SavedModelBuilder('./my_model')
        builder.add_meta_graph_and_variables(sess, signature_def_map=signature_map,
                                             tags=[tf.saved_model.tag_constants.SERVING])
        builder.save()


# Export the model:
export_model()

# Load some test images:
test_digit_0 = load_digit("digit_0.png")
test_digit_1 = load_digit("digit_1.png")
test_digit_2 = load_digit("digit_2.png")
test_digit_3 = load_digit("digit_3.png")

with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], './my_model')
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name('myInput:0')
    model = graph.get_tensor_by_name('myOutput:0')
    output = sess.run(model, {x: [test_digit_0, test_digit_1, test_digit_2, test_digit_3]})
    print("predicted labels: {}".format(np.argmax(output, axis=1)))
