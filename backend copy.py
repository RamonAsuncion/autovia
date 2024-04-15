'''
REST API for semantic segmentation inference
'''

import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, send_file
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

MODEL_WEIGHTS_PATH = './weights/model-bnet-one-day.pth'
inferenceWorker = InferenceWorker(model_weights_path=MODEL_WEIGHTS_PATH)


@app.route("/api/segment", methods=["POST"])
def segment():
    print(request.files)

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
        return segmentVideo(file)
    else:
        return jsonify({'error': 'Unsupported file type'}), 400


@app.route("/api/clear", methods=["POST"])
def clear():
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


def segmentVideo(file):
    # Save the uploaded video file temporarily
    temp_video_path = os.path.join('temp', file.filename)
    ensure_dir(temp_video_path)
    file.save(temp_video_path)

    # Process the video
    output_video_path = os.path.join('temp', 'output_' + file.filename)
    process_video(temp_video_path, output_video_path)

    # Send the processed video file back to the client
    return send_file(output_video_path, mimetype='video/mp4', as_attachment=True)


def process_video(video_path: str, output_path: str):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not opne video file.")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (512, 256))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Normalize the frame.
        frame_normalized = frame_rgb.astype(np.float32) / 255.0

        # Segment the frame.
        segmented_frame = inferenceWorker.segment(frame_normalized)
        img_float32 = np.float32(segmented_frame)
        segmented_frame_bgr = cv2.cvtColor(img_float32, cv2.COLOR_RGB2BGR)

        cv2.imshow('Segmented Frame', segmented_frame_bgr)
        out.write(segmented_frame_bgr)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


if __name__ == '__main__':
    app.run(debug=True)
