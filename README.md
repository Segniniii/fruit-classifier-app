🍎 Fruit Image Classifier 🍌
This project is a deep learning-based image classifier built with Python and TensorFlow/Keras. It demonstrates a complete machine learning pipeline, from data loading and model training to deployment via a simple web application using Flask.

The primary model is a simple Convolutional Neural Network (CNN) built from scratch.

Model Limitations: A Note on Generalization
This simple CNN model was trained on the "Fruits 360" dataset, which is very "clean"—all images have a plain white background and consistent lighting.

As a result, the model is highly accurate at recognizing images from that specific dataset. However, it performs poorly on real-world images that have different backgrounds, angles, or lighting conditions. This project serves as an excellent demonstration of the challenge of generalization in machine learning and highlights why techniques like data augmentation or transfer learning are necessary for building robust, real-world applications.

How to Run This Project
Follow these steps to get the project running on your local machine.

1. Clone the Repository

git clone https://github.com/Segniniii/fruit-classifier-app.git
cd fruit-classifier-app

2. Set Up the Environment
It is highly recommended to use a virtual environment.

# Create a virtual environment
python -m venv cnn_venv

# Activate it (on Windows)
.\cnn_venv\Scripts\activate

# Install the required libraries
pip install -r requirements.txt

3. Get the Dataset
This project uses the Fruits 360 dataset from Kaggle. The model was trained on the version with 206 fruit classes.

Download the dataset from this link.

Unzip the file and find the folder named fruits-360_100x100.

Place this folder inside the project directory and rename it to dataset.

4. Train the Model (Optional)
The trained model (simple_cnn_model.keras) is included in this repository. However, if you wish to train it yourself, you can run the training script:

python train_simple_cnn.py

5. Run the Web Application

python app.py

After running the command, open your web browser and navigate to http://127.0.0.1:5000. You can now upload an image to get a prediction.

Technologies Used
Python

TensorFlow / Keras for building and training the deep learning model.

Flask for the web application backend.

Pillow & NumPy for image manipulation and numerical operations.

Matplotlib for visualizing training results.