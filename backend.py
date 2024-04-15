'''
REST API for semantic segmentation inference
'''

from flask import Flask, request, jsonify
from flask_cors import CORS

import base64
from PIL import Image
import numpy as np
import io
from Utils import InferenceWorker

app = Flask(__name__)
CORS(app)

MODEL_WEIGHTS_PATH='./weights/model-bnet-one-day.pth'
inferenceWorker = InferenceWorker(model_weights_path=MODEL_WEIGHTS_PATH)


import matplotlib.pyplot as plt
plt.figure()

@app.route("/api/segment-image", methods=["POST"])
def segmentImage():
    bodyJson = request.json
    imgb64str = bodyJson['image']
    imgb64str = imgb64str.split(',', 1)[1] # remove the prefix data:image/jpeg;base64,
    decodedImg = base64.b64decode(imgb64str)
    image = Image.open(io.BytesIO(decodedImg))
    image_np = np.array(image)

    # print(image_np.min(), image_np.max())

    # normalize to 0 1
    image_np = image_np.astype(np.float32) / 255.0

    mask_pred = inferenceWorker.segmentImage(image_np)

    # scale from model output range [0,1] to image rgb range [0,255]
    mask_pred = (mask_pred * 255).astype('uint8')
    mask_image = Image.fromarray(mask_pred)

    # Convert PIL Image to byte array
    buffered = io.BytesIO()
    mask_image.save(buffered, format="PNG")  # You can choose PNG or JPEG depending on your needs
    encoded_mask_image = base64.b64encode(buffered.getvalue()).decode()

    # Prepend the appropriate prefix
    maskPredB64str = f"data:image/png;base64,{encoded_mask_image}"

    # return 
    return jsonify({'mask':maskPredB64str}), 200

if __name__ == '__main__':
    app.run(debug=True) 