import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.models import load_model # type: ignore
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.applications import EfficientNetB0 # type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Dropout# type: ignore
from tensorflow.keras.models import Model # type: ignore
import shutil
import random
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Define directories and parameters
TRAIN_DIR = '/Users/michelangelozampieri/Desktop/plastics_CNN/seven_plastics'
TEST_DIR = '/Users/michelangelozampieri/Desktop/plastics_CNN/seven_plastics_test'
IMG_SIZE = (244, 244)
BATCH_SIZE = 32
NUM_CLASSES = 8
EPOCHS_INITIAL = 5
EPOCHS_FINE_TUNE = 10

# Create a data generator for training images
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values to [0, 1]
    rotation_range=40,  # Randomly rotate images by up to 40 degrees
    width_shift_range=0.2,  # Randomly shift images horizontally by 20%
    height_shift_range=0.2,  # Randomly shift images vertically by 20%
    shear_range=0.2,  # Randomly apply shearing transformations
    zoom_range=0.2,  # Randomly zoom in/out on images
    horizontal_flip=True,  # Randomly flip images horizontally
    fill_mode='nearest'  # Fill in missing pixels after transformations
)

# Create a data generator for validation/test images
val_datagen = ImageDataGenerator(rescale=1./255)

# Load and iterate the training dataset
train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",  # Ensure labels are inferred from subdirectories
    label_mode="categorical",  # Use categorical labels for multi-class classification
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Load and iterate the test dataset
test_datagen = ImageDataGenerator(rescale=1.0/255.0)  # Normalize pixel values
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',  # Ensure labels are categorical
    shuffle=False  # Do not shuffle to maintain order for predictions
)

# Load and iterate the test dataset
test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",  # Ensure labels are inferred from subdirectories
    label_mode="categorical",  # Use categorical labels for validation
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

def one_hot_encode(dataset, num_classes):
    def process(image, label):
        label = tf.one_hot(label, depth=num_classes)
        return image, label
    return dataset.map(process)

train_dataset = one_hot_encode(train_dataset, NUM_CLASSES)
test_dataset = one_hot_encode(test_dataset, NUM_CLASSES)

# Load the EfficientNetB0 model
base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(244, 244, 3))
base_model.trainable = False  # Initially freeze the base model

# Custom classification head
x = GlobalAveragePooling2D()(base_model.output)  # Reduce feature maps to a single vector
x = Dense(64, activation="relu")(x)  # Fewer neurons for simplicity
x = Dropout(0.5)(x)  # Add dropout to prevent overfitting
x = BatchNormalization()(x)  # Optional: Stabilize training
output_layer = Dense(NUM_CLASSES, activation="softmax")(x)  # Output layer

# Create model
model = Model(inputs=base_model.input, outputs=output_layer)

# Compile the model
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)
# Train for a few epochs with frozen base model
print("Training model with frozen base model...")
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=EPOCHS_INITIAL,

)

# Evaluate the model on the test dataset
val_loss, val_accuracy = model.evaluate(test_dataset)
print(f"Validation accuracy: {val_accuracy}")

model.save("plastics_model_v1.h5")

# Fine-tune the model
model.trainable = True
for layer in model.layers[:-20]:  # Freeze all layers except the last 20
    layer.trainable = False

# Use a lower learning rate for fine-tuning
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)

# Recompile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Train again with fine-tuning
print("Fine-tuning model...")
history_fine = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=EPOCHS_FINE_TUNE
)

# Evaluate the model after fine-tuning
val_loss, val_accuracy = model.evaluate(test_dataset)
print(f"Validation accuracy after fine-tuning: {val_accuracy}")

# Save the fine-tuned model
model.save("plastics_model_v2.keras")

# Generate predictions
print("Generating predictions...")
predictions = model.predict(test_generator)  # Use test_generator or test_dataset

# Convert predictions to class labels
predicted_classes = np.argmax(predictions, axis=1)

# Map class indices to class names
class_indices = test_generator.class_indices  # Get class indices from the generator
class_names = {v: k for k, v in class_indices.items()}  # Reverse the mapping
predicted_labels = [class_names[idx] for idx in predicted_classes]

# Print predictions
print(predicted_labels)

# Get the file paths of the test images
file_paths = test_generator.filepaths

# Plot a few test images with their predicted labels
plt.figure(figsize=(12, 12))
for i in range(9):  # Display 9 images
    plt.subplot(3, 3, i + 1)
    img = plt.imread(file_paths[i])
    plt.imshow(img)
    plt.title(f"Predicted: {predicted_labels[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
