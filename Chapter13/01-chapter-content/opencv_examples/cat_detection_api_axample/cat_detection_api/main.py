"""
Deep Learning cat detection API
"""

# Import required packages:
from flask import Flask, request, jsonify
import urllib.request
from image_processing import ImageProcessing

app = Flask(__name__)
ip = ImageProcessing()


@app.errorhandler(400)
def bad_request(e):
    # return also the code error
    return jsonify({"status": "not ok", "message": "this server could not understand your request"}), 400


@app.errorhandler(404)
def not_found(e):
    # return also the code error
    return jsonify({"status": "not found", "message": "route not found"}), 404


@app.errorhandler(500)
def not_found(e):
    # return also the code error
    return jsonify({"status": "internal error", "message": "internal error occurred in server"}), 500


@app.route('/catfacedetection', methods=['GET', 'POST', 'PUT'])
def detect_cat_faces():
    if request.method == 'GET':
        if request.args.get('url'):
            with urllib.request.urlopen(request.args.get('url')) as url:
                return jsonify({"status": "ok", "result": ip.cat_face_detection(url.read())}), 200
        else:
            return jsonify({"status": "bad request", "message": "Parameter url is not present"}), 400
    elif request.method == 'POST':
        if request.files.get("image"):
            return jsonify({"status": "ok", "result": ip.cat_face_detection(request.files["image"].read())}), 200
        else:
            return jsonify({"status": "bad request", "message": "Parameter image is not present"}), 400
    else:
        return jsonify({"status": "failure", "message": "PUT method not supported for API"}), 405


@app.route('/catdetection', methods=['GET', 'POST', 'PUT'])
def detect_cats():
    if request.method == 'GET':
        if request.args.get('url'):
            with urllib.request.urlopen(request.args.get('url')) as url:
                return jsonify({"status": "ok", "result": ip.cat_detection(url.read())}), 200
        else:
            return jsonify({"status": "bad request", "message": "Parameter url is not present"}), 400
    elif request.method == 'POST':
        if request.files.get("image"):
            return jsonify({"status": "ok", "result": ip.cat_detection(request.files["image"].read())}), 200
        else:
            return jsonify({"status": "bad request", "message": "Parameter image is not present"}), 400
    else:
        return jsonify({"status": "failure", "message": "PUT method not supported for API"}), 405


if __name__ == "__main__":
    # Add parameter host='0.0.0.0' to run on your machines IP address:
    app.run(host='0.0.0.0')
