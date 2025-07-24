import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image

# Suppress TensorFlow logging messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- 1. DATA LOADING ---
IMG_HEIGHT = 100
IMG_WIDTH = 100
BATCH_SIZE = 32
DATA_DIR = os.path.join('dataset', 'fruits-360', 'Training')

train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="training", seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE
)
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="validation", seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE
)
class_names = train_dataset.class_names
print(f"Found {len(class_names)} classes.")
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# --- 2. BUILD THE MODEL ---
inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), include_top=False, weights='imagenet'
)
base_model.trainable = False
x = base_model(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(len(class_names), activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)

# --- 3. COMPILE AND TRAIN ---
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
EPOCHS = 20
history = model.fit(
    train_dataset, validation_data=validation_dataset, epochs=EPOCHS
)
print("--- Training Finished ---")

# --- 4. IMMEDIATE POST-TRAINING TEST ---
print("\n--- Running immediate post-training test ---")

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((100, 100))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

# !!! IMPORTANT: Update this path to your test image !!!
TEST_IMAGE_PATH = r'REPLACE_WITH_FULL_PATH_TO_YOUR_IMAGE'

if not os.path.exists(TEST_IMAGE_PATH):
    print(f"Test image not found at {TEST_IMAGE_PATH}, skipping post-training test.")
else:
    # Test 1: Predict using the model in memory (before saving)
    print("\n--- Testing model currently in memory ---")
    in_memory_image = preprocess_image(TEST_IMAGE_PATH)
    in_memory_prediction = model.predict(in_memory_image)
    in_memory_index = np.argmax(in_memory_prediction)
    print(f"IN-MEMORY PREDICTION: Index={in_memory_index}, Class={class_names[in_memory_index]}, Confidence={np.max(in_memory_prediction)*100:.2f}%")

    # Save the model
    print("\n--- Saving model... ---")
    model.save('fruit_classifier.keras')
    print("--- Model saved as fruit_classifier.keras ---")

    # Test 2: Reload the model and predict again
    print("\n--- Reloading model from disk and testing again ---")
    reloaded_model = tf.keras.models.load_model('fruit_classifier.keras')
    reloaded_image = preprocess_image(TEST_IMAGE_PATH)
    reloaded_prediction = reloaded_model.predict(reloaded_image)
    reloaded_index = np.argmax(reloaded_prediction)
    print(f"RELOADED MODEL PREDICTION: Index={reloaded_index}, Class={class_names[reloaded_index]}, Confidence={np.max(reloaded_prediction)*100:.2f}%")

# --- 5. VISUALIZE RESULTS ---
# The script will only show the plot after the tests are complete.
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(EPOCHS), history.history['accuracy'], label='Training Accuracy')
plt.plot(range(EPOCHS), history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.subplot(1, 2, 2)
plt.plot(range(EPOCHS), history.history['loss'], label='Training Loss')
plt.plot(range(EPOCHS), history.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()