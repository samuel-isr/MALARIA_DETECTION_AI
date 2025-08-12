# app.py (Final, Corrected Version)

import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import io
import traceback

# Initialize the Flask application
app = Flask(__name__)
CORS(app)

# Load our trained AI model
try:
    # This will now work correctly with the newly saved model
    model = tf.keras.models.load_model('malaria_model.h5')
    print("--- Model loaded successfully! ---")
except Exception as e:
    print(f"--- FATAL: Error loading model: {e} ---")
    traceback.print_exc()
    model = None

# Define the image size the model expects
IMG_SIZE = (128, 128)

def preprocess_image(image_bytes):
    """
    Takes image bytes, opens the image, resizes it, and prepares it
    for the model.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    # The preprocessing is now done inside the model, so we don't need it here.
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    return img_array

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model is not loaded!'}), 500
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    try:
        image_bytes = file.read()
        preprocessed_image = preprocess_image(image_bytes)
        prediction = model.predict(preprocessed_image)
        confidence = prediction[0][0]

        if confidence < 0.5:
            label = 'Parasitized'
            confidence_percent = (1 - confidence) * 100
        else:
            label = 'Uninfected'
            confidence_percent = confidence * 100
        
        # --- THIS IS THE FIX ---
        # Convert the numpy float32 to a standard Python float
        # before sending it as JSON.
        return jsonify({
            'prediction': label,
            'confidence': float(confidence_percent) 
        })

    except Exception as e:
        print("\n--- AN ERROR OCCURRED DURING PREDICTION ---")
        traceback.print_exc()
        return jsonify({'error': f'An error occurred on the server: {str(e)}'}), 500

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
