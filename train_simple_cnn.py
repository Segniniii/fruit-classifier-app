import tensorflow as tf
import matplotlib.pyplot as plt
import os

# Suppress TensorFlow logging messages for a cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- 1. SETUP AND DATA LOADING ---

# Define key parameters for our model and data
IMG_HEIGHT = 100
IMG_WIDTH = 100
BATCH_SIZE = 32
# Correctly join the path components, including the 'fruits-360' folder
DATA_DIR = os.path.join('dataset', 'fruits-360', 'Training')

# Load the training data from the directory
# We use 80% of the images for training and 20% for validation
train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, 
    validation_split=0.2,
    subset="training",
    seed=123, # Using a seed ensures the split is the same every time
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

# Load the validation data from the directory
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, # joined path again
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

# Get the class names (the names of the fruit folders)
class_names = train_dataset.class_names
print(f"Found {len(class_names)} classes: {class_names[:5]}...") # Print first 5 classes

# Configure dataset for performance by caching and prefetching
# This helps prevent I/O bottlenecks during training
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)



# --- 2. BUILD THE CNN MODEL ---

# Create the Sequential model
model = tf.keras.Sequential([
    # Input Layer: Rescale pixel values from [0, 255] to [0, 1]
    tf.keras.layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),

    # First Convolutional Block
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    # Second Convolutional Block
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    # Third Convolutional Block
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    # Flatten the 3D output to 1D to feed into the Dense layers
    tf.keras.layers.Flatten(),

    # A standard fully-connected layer
    tf.keras.layers.Dense(128, activation='relu'),

    # Output Layer: Has one neuron for each fruit class.
    # Softmax activation gives the probability for each class.
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])

# --- 3. COMPILE THE MODEL ---

# Configure the model for training
model.compile(
    optimizer='adam', # Adam is a good default optimizer
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# Print a summary of the model's architecture
model.summary()

# --- 4. TRAIN THE MODEL ---

print("\n--- Starting Training ---")
# Set the number of epochs (how many times to go through the entire dataset)
EPOCHS = 10

# Train the model using the .fit() method
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS
)
print("--- Training Finished ---")

# --- 5. VISUALIZE TRAINING RESULTS ---

# Extract accuracy and loss from the history object
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

# Create a plot to visualize the training and validation accuracy
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

# Create a plot to visualize the training and validation loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

# Show the plot
plt.show()

model.save('simple_cnn_model.keras')
print("\n--- Simple CNN model saved as simple_cnn_model.keras ---")