# 🍎 Fruit Image Classifier 🍌

This project is an end-to-end machine learning application designed to classify images of fruits. It uses a Convolutional Neural Network (CNN) built with Python and TensorFlow/Keras to analyze images and provides a prediction through a simple web interface.

The project demonstrates the entire machine learning pipeline, from data preparation and model training to deployment via a local web server.

---

## Features

* **Fruit Classification:** Identifies over 200 different types of fruits from an uploaded image.
* **Simple Web Interface:** A clean, user-friendly UI built with Flask to test the model in real-time.
* **CNN from Scratch:** The model is a Convolutional Neural Network built and trained from the ground up to demonstrate foundational deep learning concepts.
* **Performance Visualization:** After training, the script automatically generates plots showing the model's accuracy and loss over time.

---

## How It Works

The project is broken down into several key stages:

1.  **Data Preparation:** The model is trained on the **Fruits 360** dataset from Kaggle. This dataset contains over 90,000 images across 206 classes, with each image being 100x100 pixels on a clean white background. The script loads this data and splits it into training (80%) and validation (20%) sets.

2.  **Model Training:** A simple `Sequential` CNN model is built using Keras. It consists of several convolutional and max-pooling layers to extract features from the images, and dense layers for the final classification. The model is trained on the dataset to learn the distinguishing features of each fruit.

3.  **API Development:** A Flask server (`app.py`) provides a `/predict` endpoint. This endpoint receives an image, preprocesses it to match the training format, feeds it to the trained model, and returns a JSON response with the predicted fruit name and a confidence score.

### Model Performance & Limitations

The model achieves a high validation accuracy of over 99% on the test dataset.

<img width="1185" height="499" alt="image" src="https://github.com/user-attachments/assets/3e847780-935f-480c-a310-a272841c0303" />


However, because the model was trained exclusively on "clean" images with white backgrounds, it struggles to generalize to real-world photos with varied backgrounds, lighting, and angles. This project serves as an excellent demonstration of the challenges in creating robust, real-world ML systems.

---

## Tech Stack

* **Backend:** Python, Flask
* **Machine Learning:** TensorFlow / Keras, NumPy
* **Image Processing:** Pillow
* **Data Visualization:** Matplotlib
* **Frontend:** HTML, JavaScript

---

## How to Run Locally

**Prerequisites:**
* Python 3.9+
* `pip` and `venv`

**Instructions:**

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Segniniii/fruit-classifier-app.git](https://github.com/Segniniii/fruit-classifier-app.git)
    cd fruit-classifier-app
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For Windows
    python -m venv cnn_venv
    .\cnn_venv\Scripts\activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Get the Dataset:**
    * Download the dataset from the [Fruits 360 page on Kaggle](https://www.kaggle.com/datasets/moltean/fruits).
    * Unzip the file and find the folder named `fruits-360_100x100`.
    * Place this folder inside the project directory and rename it to `dataset`.

5.  **Train the Model (Optional):**
    The trained model (`simple_cnn_model.keras`) isn't included in this repository. To train it yourself, run:
    ```bash
    python train_simple_cnn.py
    ```

6.  **Run the Flask application:**
    ```bash
    python app.py
    ```

7.  Open your web browser and navigate to `http://127.0.0.1:5000`.

---

## Project Structure

```
.
├── dataset/                    # (Not in repo) Contains the fruit images
├── templates/
│   └── index.html              # Frontend HTML and JavaScript
├── .gitignore                  # Files and folders to ignore by Git
├── app.py                      # Flask application with API endpoints
├── requirements.txt            # Project dependencies
├── train_simple_cnn.py         # Script to train the model
└── simple_cnn_model.keras      # (Not in repo) The trained model file
