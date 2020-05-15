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
    img = Image.open(request.files['file'].stream) #.convert('RGB').resize((224, 224))
    img = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY).resize(224, 224)
    img = np.array(img)
    img = torch.FloatTensor(img.transpose((0, 1)) / 255).unsqueeze(0)

    # get predictions
    pred = net(img.unsqueeze(0)).squeeze()
    pred_probas = torch.softmax(pred, axis=0)

    return {
        'pneumonia': pred_probas[1].item(),
        'normal': pred_probas[0].item()
    }


if __name__ == "__main__":
    app.run(debug=True)
