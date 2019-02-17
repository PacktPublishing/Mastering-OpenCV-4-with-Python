"""
 Request example to perform a POST request using the Keras Deep Learning REST API
"""

# Import required packages:
import requests

KERAS_REST_API_URL = "http://localhost:5000/predict"
IMAGE_PATH = "car.jpg"

# Load the image and construct the payload:
image = open(IMAGE_PATH, "rb").read()
payload = {"image": image}

# Submit the POST request:
r = requests.post(KERAS_REST_API_URL, files=payload).json()

# Print the results:
if r["success"]:
    # Iterate over the predictions and print them:
    for (i, result) in enumerate(r["predictions"]):
        print("{}. {}: {:.4f}".format(i + 1, result["label"], result["probability"]))
else:
    print("Request failed")
