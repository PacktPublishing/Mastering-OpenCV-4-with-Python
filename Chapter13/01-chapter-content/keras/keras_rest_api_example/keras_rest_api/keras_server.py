"""
Simple Keras Deep Learning REST API using pre-trained Deep Learning models
"""

# Import required packages:
from keras.applications import nasnet, NASNetMobile
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import flask
import io
import tensorflow as tf

# Initialize Flask app, Keras model and graph:
app = flask.Flask(__name__)
graph = None
model = None


def load_model():
    # Get default graph:
    global graph
    graph = tf.get_default_graph()
    # Load the pre-trained Keras model(pre-trained on ImageNet):
    global model
    model = NASNetMobile(weights="imagenet")


def preprocessing_image(image, target):
    # Make sure the image mode is RGB:
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Resize the input image:
    image = image.resize(target)
    # Convert PIL format to numpy array:
    image = img_to_array(image)
    # Convert the image/images into batch format:
    image = np.expand_dims(image, axis=0)
    # Pre-process (prepare) the image using the specific architecture:
    image = nasnet.preprocess_input(image)
    # Return the image:
    return image


@app.route("/predict", methods=["POST"])
def predict():
    # Initialize result:
    result = {"success": False}

    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # Read input image in PIL format:
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            # Pre-process the image to be classified:
            image = preprocessing_image(image, target=(224, 224))

            # Classify the input image:
            with graph.as_default():
                predictions = model.predict(image)
            results = imagenet_utils.decode_predictions(predictions)
            result["predictions"] = []

            # Add the predictions to the result:
            for (imagenet_id, label, prob) in results[0]:
                r = {"label": label, "probability": float(prob)}
                result["predictions"].append(r)

            # At this point we can say that the request was dispatched successfully:
            result["success"] = True

    # Return result as a JSON response:
    return flask.jsonify(result)


@app.route("/")
def home():
    # Initialize result:
    result = {"success": True}
    # Return result as a JSON response:
    return flask.jsonify(result)


if __name__ == "__main__":
    print("Loading Keras pre-trained model")
    load_model()
    print("Starting")
    app.run()
