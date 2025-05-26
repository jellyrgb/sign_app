# ViT 모델만 사용하는 Flask 서버
from flask import Flask, render_template, request, jsonify
import os
import numpy as np
from PIL import Image
import base64
import io
import tensorflow as tf
from transformers import AutoImageProcessor, TFViTForImageClassification
from sklearn.preprocessing import LabelBinarizer

app = Flask(__name__)

# 라벨 리스트 (J, Z 제외된 A~Z)
label_list = [chr(i) for i in range(ord('A'), ord('Z') + 1) if chr(i) not in ['J', 'Z']]
lb = LabelBinarizer()
lb.fit(label_list)

# ViT 모델 불러오기
vit_model = TFViTForImageClassification.from_pretrained("PanHwa/vit-sign-model")
vit_processor = AutoImageProcessor.from_pretrained("PanHwa/vit-sign-model")

def preprocess_image(image):
    img = image.convert("RGB").resize((224, 224))
    inputs = vit_processor(images=img, return_tensors="tf")
    return inputs['pixel_values']

def predict(image):
    inputs = preprocess_image(image)
    outputs = vit_model(inputs, training=False)
    logits = outputs.logits.numpy()
    label = lb.classes_[np.argmax(logits)]
    confidence = float(np.max(logits))
    return label, confidence

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    data = request.json
    image_data = data['image'].split(',')[1]
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes))

    label, confidence = predict(image)
    return jsonify({ "label": label, "confidence": round(confidence, 3) })

if __name__ == '__main__':
    app.run(debug=True)
