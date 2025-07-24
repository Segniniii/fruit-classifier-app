import tensorflow as tf
import matplotlib.pyplot as plt
import os

# Hide TensorFlow's info-level log messages for a cleaner console output.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- 1. SETUP AND DATA LOADING ---
# Define constants and prepare the data pipeline.

# Define image dimensions for consistency.
IMG_HEIGHT = 100
IMG_WIDTH = 100
# Set batch size for training efficiency.
BATCH_SIZE = 32
# Path to the training data directory.
DATA_DIR = os.path.join('dataset', 'fruits-360', 'Training')

# Load images from the directory, splitting them into training (80%) and validation (20%) sets.
# Keras infers class labels from the subdirectory names.
print("--- Loading the data... ---")
train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123, # Use a seed for a reproducible train/validation split.
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

# This uses the same directory but grabs the validation subset.
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

# Get the class names from the directory structure.
class_names = train_dataset.class_names
print(f"Found {len(class_names)} classes. Here are a few: {class_names[:5]}...")

# Optimize dataset performance by caching data in memory and prefetching batches.
# This prevents the GPU from waiting for data I/O.
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# --- 2. BUILD THE CNN MODEL ---
# Define the model architecture by stacking layers sequentially.
model = tf.keras.Sequential([
    # Normalize pixel values from [0, 255] to [0, 1] for better model performance.
    tf.keras.layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),

    # First convolutional block. Scans for basic patterns like edges and textures.
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    # MaxPooling reduces image dimensions, focusing on key features to improve efficiency.
    tf.keras.layers.MaxPooling2D(),

    # Second convolutional block with more filters to learn more complex patterns.
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    # Third convolutional block for even more abstract features.
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    # Flatten the 2D feature maps into a 1D vector for the Dense layers.
    tf.keras.layers.Flatten(),

    # A standard fully-connected layer for high-level reasoning.
    tf.keras.layers.Dense(128, activation='relu'),

    # Output layer with one neuron per class. Softmax gives class probabilities.
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])

# --- 3. COMPILE THE MODEL ---
# Configure the training process.
model.compile(
    # The Adam optimizer is an efficient, popular choice for adjusting model weights.
    optimizer='adam',
    # Use SparseCategoricalCrossentropy for multi-class classification with integer labels.
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    # Monitor accuracy as the primary performance metric during training.
    metrics=['accuracy']
)

# Print a summary of the model's architecture and parameters.
model.summary()

# --- 4. TRAIN THE MODEL ---
print("\n--- Starting Training ---")
# An epoch is one full pass through the entire training dataset.
EPOCHS = 10

# Start the training process. The model learns by iterating over the dataset.
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS
)
print("--- Training Finished ---")

# --- 5. VISUALIZE TRAINING RESULTS ---
# The 'history' object stores metrics from each epoch, which we can plot.
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

# Visualize the training and validation metrics over epochs.
plt.figure(figsize=(12, 5))

# Plot for Accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

# Plot for Loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

# Display the plots.
plt.show()

# --- 6. SAVE THE FINAL MODEL ---
# Save the trained model's architecture, weights, and configuration to a single file.
model.save('simple_cnn_model.keras')
print("\n--- Simple CNN model saved as simple_cnn_model.keras ---")


#This trained model is still pretty stupid, if you show a real-world image it will not be able to guess it, however, it will be able to guess every single image in the dataset.