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

video_feed_sessions = {}

MODEL_WEIGHTS_PATH = './weights/model-bnet-one-day.pth'
inferenceWorker = InferenceWorker(model_weights_path=MODEL_WEIGHTS_PATH)


@app.route("/api/segment", methods=["POST"])
def segment():
    global video_feed_sessions

    # get the session_id generated from frontend
    session_id = request.args.get('session_id')
    session_tmp_dir = os.path.join('temp', session_id)
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Check if the file is an image or a video.
    content_type = request.files['file'].content_type
    if content_type.startswith('image/'):
        print(f'segmenting single image {file.filename} with session_id {session_id}')
        video_feed_sessions[session_id] = {'path': None, 'active': False}
        return segmentImage(file)
    elif content_type.startswith('video/'):
        temp_video_path = os.path.join(session_tmp_dir, file.filename)
        ensure_dir(temp_video_path)
        file.save(temp_video_path)
        video_feed_sessions[session_id] = {'path': temp_video_path, 'active': True}
        return segmentVideo(session_id)
    else:
        return jsonify({'error': 'Unsupported file type'}), 400


@app.route("/api/clear", methods=["POST"])
def clear():
    global video_feed_sessions

    # Get the session_id generated from frontend
    session_id = request.args.get('session_id')

    # clear the video feed session
    video_feed_sessions[session_id]['active'] = False

    # Clear the video file if it exists
    session_tmp_dir = os.path.join('temp', session_id)
    if os.path.exists(session_tmp_dir):
        if video_feed_sessions[session_id]['path'] is not None:
            os.remove(video_feed_sessions[session_id]['path'])

    return jsonify({'message': 'Temp directory cleared'}), 200


def segmentImage(file):
    '''segmenting a single image won't require the image file to be saved into disk'''
    # # Process the image
    image = Image.open(io.BytesIO(file.read()))
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


def segmentVideo(session_id):
    global video_feed_sessions, inferenceWorker

    current_video_path = video_feed_sessions[session_id]['path']
    cap = cv2.VideoCapture(current_video_path)

    while cap.isOpened():

        # Stop the video feed if user clicks the clear button
        if not video_feed_sessions[session_id]['active']:
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

@app.route('/video_feed')
def video_feed():
    global video_feed_sessions

    session_id = request.args.get('session_id')
    # if session_id not in video_feed_sessions or not video_feed_sessions[session_id]['active']:
    #     return jsonify({'message': 'Video cleared or session not found'}), 404
    return Response(segmentVideo(session_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


if __name__ == '__main__':
    app.run(debug=True)
