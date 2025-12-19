# app.py
from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
from PIL import Image
import base64
import io

app = Flask(__name__)

# Load the trained model (you need to save it as 'model.h5' from the notebook)
model = tf.keras.models.load_model('model.h5')

# Load the product dataframe
df = pd.read_excel('skin_products-1.xlsx')

# Skin type mapping
skintypes = ['Oily', 'Dry', 'Normal', 'Sensitive']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the base64 image from the request
        data = request.json['image']
        # Decode base64 image
        image_data = base64.b64decode(data.split(',')[1])
        image = Image.open(io.BytesIO(image_data))
        
        # Preprocess image
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (64, 64))
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Predict skin type
        prediction = model.predict(img)
        skin_index = np.argmax(prediction)
        detected_skin = skintypes[skin_index]
        
        # Get recommendations
        recommended = df[df['Skin Type'].str.contains(detected_skin, case=False, na=False)].head(5)
        
        return jsonify({
            'skin_type': detected_skin,
            'recommendations': recommended[['Product Name', 'Brand', 'Price', 'Reviews', 'Low-Cost Website']].to_dict('records')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

