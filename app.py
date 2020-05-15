from flask import Flask, request
import torch
from PIL import Image
import numpy as np
from flask_cors import CORS
import cv2
app = Flask(__name__)
CORS(app)

net = torch.jit.load('ResNet_18.zip')


@app.route('/')
def hello():
    return "Hello!"


@app.route("/predict", methods=['POST'])
def predict():

    # load image
    img = Image.open(request.files['file'].stream) .convert(mode = 'L').resize((224, 224)).unsqueeze(0)
    img = np.array(img)
    img = torch.FloatTensor(img / 255)

    # get predictions
    pred = net(img)
    pred_probas = torch.softmax(pred, axis=0)

    return {
        'pneumonia': pred_probas[1].item(),
        'normal': pred_probas[0].item()
    }


if __name__ == "__main__":
    app.run(debug=True)
