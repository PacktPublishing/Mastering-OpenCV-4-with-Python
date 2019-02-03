"""
Saving a linear regression model using SavedModelBuilder in TensorFlow
"""

# Import required packages:
import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils


def export_model():
    """Exports the model"""

    trained_checkpoint_prefix = 'linear_regression'

    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        sess.run(tf.global_variables_initializer())

        # Restore from checkpoint:
        loader = tf.train.import_meta_graph(trained_checkpoint_prefix + '.meta')
        loader.restore(sess, trained_checkpoint_prefix)

        # Add signature:
        graph = tf.get_default_graph()
        inputs = tf.saved_model.utils.build_tensor_info(graph.get_tensor_by_name('X:0'))
        outputs = tf.saved_model.utils.build_tensor_info(graph.get_tensor_by_name('y_model:0'))

        signature = signature_def_utils.build_signature_def(inputs={'X': inputs},
                                                            outputs={'y_model': outputs},
                                                            method_name=signature_constants.PREDICT_METHOD_NAME)

        signature_map = {signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature}

        # Export model:
        builder = tf.saved_model.builder.SavedModelBuilder('./my_model')
        builder.add_meta_graph_and_variables(sess, signature_def_map=signature_map,
                                             tags=[tf.saved_model.tag_constants.SERVING])
        builder.save()


# Export the model:
export_model()

# Define 'M' more points to get the predictions using the trained model:
new_x = np.linspace(50 + 1, 50 + 10, 3)

with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], './my_model')
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name('X:0')
    model = graph.get_tensor_by_name('y_model:0')
    print(sess.run(model, {x: new_x}))
