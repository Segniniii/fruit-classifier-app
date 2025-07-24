import os
import tensorflow as tf
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template

# --- SETUP ---

# Create the Flask web application
app = Flask(__name__)

# Suppress TensorFlow logging messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- LOAD THE TRAINED MODEL ---
print("--- Loading the simple CNN model... ---")
# Load the .keras file for the simple CNN
MODEL_PATH = 'simple_cnn_model.keras'
model = tf.keras.models.load_model(MODEL_PATH)
print("--- Model Loaded! ---")


# --- GET CLASS NAMES ---

import json
with open('class_names.json', 'r') as f:
    class_names = json.load(f)
print(f"--- Class names loaded successfully from class_names.json ({len(class_names)} classes) ---")

# --- HELPER FUNCTION (Corrected for Simple CNN) ---
def preprocess_image(image_file):
    """
    Takes an image file and prepares it for the SIMPLE CNN model.
    """
    img = Image.open(image_file.stream).convert('RGB')
    img = img.resize((100, 100))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    # Add the batch dimension. No other preprocessing is needed because
    # the simple CNN model has a Rescaling layer built-in.
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# --- FLASK ROUTES ---

@app.route('/', methods=['GET'])
def index():
    """Renders the main page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives an image upload and returns a prediction in JSON format.
    """
    # Check if a file was uploaded in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    # Check if a file was selected
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file:
        # Preprocess the image
        processed_image = preprocess_image(file)
        
        # Get the model's prediction
        prediction = model.predict(processed_image)
        
        # --- DEBUG LINES ---
        score = float(np.max(prediction))
        predicted_index = np.argmax(prediction)
        print(f"DEBUG: Predicted Index: {predicted_index}, Confidence: {score:.4f}")
        # --- END OF DEBUG LINES ---
        
        # Get the corresponding class name
        predicted_class_name = class_names[predicted_index]

        # Return the result as JSON, now with confidence score
        return jsonify({'prediction': predicted_class_name, 'confidence': f"{score*100:.2f}%"})

# This is required to run the app
if __name__ == '__main__':
    app.run(debug=True)
