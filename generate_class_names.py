import os
import json
import tensorflow as tf

# THIS PATH IS CRITICAL. It must point to the folder with 131 fruit sub-folders.
DATA_DIR = os.path.join('dataset', 'fruits-360', 'Training')

# Load the dataset just to get the class names
train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    image_size=(100, 100),
    batch_size=1,
    shuffle=False
)

class_names = train_dataset.class_names

# Save the list to a JSON file
with open('class_names.json', 'w') as f:
    json.dump(class_names, f)

# This check is the most important part!
print(f"CRITICAL CHECK: Successfully saved {len(class_names)} class names to class_names.json")