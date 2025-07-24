import os
import tensorflow as tf
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template

# --- SETUP ---

# Initialize the Flask web application.
app = Flask(__name__)

# Hide TensorFlow's info-level log messages for a cleaner console output.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- IMPORTANT NOTE ON THIS MODEL'S LIMITATIONS ---
# This simple CNN model was trained on a very "clean" dataset where all images
# have a plain white background and perfect lighting.
# As a result, it is very good at recognizing images from that dataset,
# but it performs poorly on real-world images with different backgrounds, angles, or lighting.
# This demonstrates the challenge of "generalization" in machine learning.



# --- LOAD THE TRAINED MODEL ---
print("--- Loading the simple CNN model... ---")
# Load the saved Keras model from disk.
MODEL_PATH = 'simple_cnn_model.keras'
model = tf.keras.models.load_model(MODEL_PATH)
print("--- Model Loaded! ---")


# --- GET CLASS NAMES ---
# Load the class names from the JSON file to map predictions to labels.
import json
with open('class_names.json', 'r') as f:
    class_names = json.load(f)
print(f"--- Class names loaded successfully from class_names.json ({len(class_names)} classes) ---")

# --- IMAGE PREPROCESSING FUNCTION ---
def preprocess_image(image_file):
    """
    Preprocesses an uploaded image file for model prediction.
    """
    # Open the image from the file stream and ensure it's in RGB format.
    img = Image.open(image_file.stream).convert('RGB')
    # Resize the image to match the model's expected input dimensions.
    img = img.resize((100, 100))
    # Convert the PIL image to a NumPy array.
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    # Add a batch dimension, as the model expects inputs of shape (batch_size, height, width, channels).
    # No other preprocessing is needed because the model has a Rescaling layer.
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# --- FLASK ROUTES ---

@app.route('/', methods=['GET'])
def index():
    """Renders the main HTML page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives an image upload and returns a prediction in JSON format.
    """
    # Validate that a file was included in the request.
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    # Validate that a file was actually selected by the user.
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file:
        # Prepare the image for the model.
        processed_image = preprocess_image(file)
        
        # Get the model's prediction probabilities for each class.
        prediction = model.predict(processed_image)
        
        # --- DEBUG LINES ---
        # Get the highest probability score from the prediction array.
        score = float(np.max(prediction))
        # Get the index of the class with the highest probability.
        predicted_index = np.argmax(prediction)
        print(f"DEBUG: Predicted Index: {predicted_index}, Confidence: {score:.4f}")
        # --- END OF DEBUG LINES ---
        
        # Look up the class name using the predicted index.
        predicted_class_name = class_names[predicted_index]

        # Return the final prediction and confidence score as a JSON response.
        return jsonify({'prediction': predicted_class_name, 'confidence': f"{score*100:.2f}%"})

# Entry point for running the Flask application.
if __name__ == '__main__':
    app.run(debug=True)
