'''
REST API for semantic segmentation inference
'''

from flask import Response
import matplotlib.pyplot as plt
from flask import Flask, request, Response, jsonify, send_from_directory
from flask_cors import CORS

import base64
from PIL import Image
import numpy as np
import io
from Utils import InferenceWorker
import os
import cv2

app = Flask(__name__)
CORS(app)

current_video_path = None
video_feed_active = True

MODEL_WEIGHTS_PATH = './weights/model-bnet-one-day.pth'
inferenceWorker = InferenceWorker(model_weights_path=MODEL_WEIGHTS_PATH)


@app.route("/api/segment", methods=["POST"])
def segment():
    global video_feed_active
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Check if the file is an image or a video.
    content_type = request.files['file'].content_type
    if content_type.startswith('image/'):
        return segmentImage(file)
    elif content_type.startswith('video/'):
        video_feed_active = True
        temp_video_path = os.path.join('temp', file.filename)
        ensure_dir(temp_video_path)
        file.save(temp_video_path)
        global current_video_path
        current_video_path = temp_video_path
        return segmentVideo()
    else:
        return jsonify({'error': 'Unsupported file type'}), 400


@app.route("/api/clear", methods=["POST"])
def clear():
    global video_feed_active, current_video_path
    video_feed_active = False
    current_video_path = None
    # Clear the temp directory
    for file in os.listdir('temp'):
        file_path = os.path.join('temp', file)
        os.remove(file_path)
    return jsonify({'message': 'Temp directory cleared'}), 200


def segmentImage(file):
    temp_image_path = os.path.join('temp', file.filename)
    ensure_dir(temp_image_path)
    file.save(temp_image_path)

    # Process the image
    image = Image.open(temp_image_path)
    image_np = np.array(image)
    image_np = image_np.astype(np.float32) / 255.0
    mask_pred = inferenceWorker.segment(image_np)

    # Convert the mask to a base64 string
    mask_pred = (mask_pred * 255).astype('uint8')
    mask_image = Image.fromarray(mask_pred)

    # Convert the mask image to a base64 string
    buffered = io.BytesIO()
    mask_image.save(buffered, format="PNG")
    encoded_mask_image = base64.b64encode(buffered.getvalue()).decode()

    # Send the mask image back to the client
    maskPredB64str = f"data:image/png;base64,{encoded_mask_image}"
    return jsonify({'mask': maskPredB64str}), 200


def segmentVideo():
    global current_video_path, video_feed_active
    cap = cv2.VideoCapture(current_video_path)

    while cap.isOpened():

        if not video_feed_active:
            break

        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Normalize the frame.
        frame_normalized = frame_rgb.astype(np.float32) / 255.0

        # Segment the frame.
        segmented_frame = inferenceWorker.segment(frame_normalized)

        segmented_frame_bgr = (segmented_frame * 255).astype(np.uint8)
        ret, buffer = cv2.imencode('.jpg', segmented_frame_bgr)
        frame_bytes = buffer.tobytes()

        # Yield the frame bytes
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
    video_feed_active = False


@app.route('/video_feed')
def video_feed():
    if video_feed_active is False:
        return jsonify({'message': 'Video cleared'}), 200
    return Response(segmentVideo(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


if __name__ == '__main__':
    app.run(debug=True)
