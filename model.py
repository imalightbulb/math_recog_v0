import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import keras
import random

from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras_preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight

# Preprocess Function
def preprocess_img(image_path):
    image = Image.open(image_path).convert('L')
    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    image_array = np.array(image) / 255.0
    image_array = image_array.reshape(28, 28, 1)  # match with MNIST since CNN works with 3D
    return image_array

# Process images in folder
data_dir = r'C:\Users\imalightbulb\PycharmProjects\math_recog_v0\data\ddx'
print(f"Data dir: {data_dir}")

if not os.path.exists(data_dir):
    print("Data directory does not exist.")
else:
    print("Data directory exists.")

files_in_dir = os.listdir(data_dir)
print(f"Files in data directory: {files_in_dir}")

# Store processed items
custom_images = []
custom_labels = []

label_ddx = 10

print("Processing custom images...")

for filename in os.listdir(data_dir):
    print(f"File {filename} found")
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        print(f"Processing image: {filename}")
        image_path = os.path.join(data_dir, filename)
        try:
            image_array = preprocess_img(image_path)
            custom_images.append(image_array)
            custom_labels.append(label_ddx)
        except Exception as e:
            print(f"Error processing image: {filename}: {e}")
    else:
        print(f"Skipped file: {filename}")

print(f"Total custom images processed: {len(custom_images)}")

# Load MNIST dataset
(mnist_images, mnist_labels), _ = keras.datasets.mnist.load_data()

mnist_images = mnist_images / 255.0
mnist_images = mnist_images.reshape(mnist_images.shape[0], 28, 28, 1)

custom_images = np.array(custom_images)
print(f"Shape of custom_images after np.array: {custom_images.shape}")

custom_labels = np.array(custom_labels)

combined_images = np.concatenate((custom_images, mnist_images), axis=0)
combined_labels = np.concatenate((custom_labels, mnist_labels), axis=0)
print(f"Shape of combined_images: {combined_images.shape}")

# Ensure labels are integers
combined_labels = combined_labels.astype(int)

# Class distribution before augmentation
unique, counts = np.unique(combined_labels, return_counts=True)
print("Class distribution before augmentation:", dict(zip(unique, counts)))

# Augment d/dx images
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)

ddx_images = custom_images  # All custom images have label 10
ddx_labels = custom_labels

augmented_images = []
augmented_labels = []

# Reduced the number of augmented images per original
num_augmented_images_per_original = 30  # Reduced from 100

for i in range(len(ddx_images)):
    image = ddx_images[i].reshape(1, 28, 28, 1)
    label = ddx_labels[i]
    aug_iter = datagen.flow(image, batch_size=1)
    for _ in range(num_augmented_images_per_original):
        aug_image = next(aug_iter)[0]
        augmented_images.append(aug_image)
        augmented_labels.append(label)

# DEBUG: Check augmentation code
print(f"Number of augmented images: {len(augmented_images)}")
print(f"Number of augmented labels: {len(augmented_labels)}")

# Convert to NumPy arrays
augmented_images = np.array(augmented_images)
augmented_labels = np.array(augmented_labels).astype(int)

# Combine augmented images with the original dataset
combined_images = np.concatenate((combined_images, augmented_images), axis=0)
combined_labels = np.concatenate((combined_labels, augmented_labels), axis=0)

# Class distribution after augmentation
unique, counts = np.unique(combined_labels, return_counts=True)
print("Label counts after augmentation:", dict(zip(unique, counts)))

# Shuffle the data to remove any ordering bias
shuffle_indices = np.random.permutation(len(combined_images))
combined_images = combined_images[shuffle_indices]
combined_labels = combined_labels[shuffle_indices]

# First split: 60% training, 40% for testing and validation
train_images, temp_images, train_labels, temp_labels = train_test_split(
    combined_images, combined_labels, test_size=0.4, random_state=42, stratify=combined_labels
)

# Second split: Split the 40% into 20% validation and 20% testing
val_images, test_images, val_labels, test_labels = train_test_split(
    temp_images, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
)

# Function to print class distribution
def print_class_distribution(labels, set_name):
    unique, counts = np.unique(labels, return_counts=True)
    print(f"Class distribution in {set_name} set:", dict(zip(unique, counts)))

print_class_distribution(train_labels, "training")
print_class_distribution(val_labels, "validation")
print_class_distribution(test_labels, "test")

# Build the CNN model with an extra convolutional layer and dropout layers
model = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(28, 28, 1)),

    # First convolutional layer
    keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),

    # Second convolutional layer
    keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),

    # Third convolutional layer (new)
    keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
    keras.layers.Flatten(),

    # Fully connected layer with dropout
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),

    # Output layer
    keras.layers.Dense(11, activation='softmax')  # 10 digits + 1 custom symbol
])

# Use a lower learning rate
optimizer = keras.optimizers.Adam(learning_rate=0.0001)

model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Compute class weights and adjust the weight for class 10
classes = np.unique(train_labels)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=train_labels
)
class_weight_dict = dict(zip(classes, class_weights))

# Adjust class weight for 'd/dx' (label 10)
original_weight = class_weight_dict[10]
adjusted_weight = 2.0  # Reduced from original weight
class_weight_dict[10] = adjusted_weight
print(f"Adjusted class weights: {class_weight_dict}")

# Train model
callbacks = [
    keras.callbacks.EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, monitor='val_loss')
]

history = model.fit(
    train_images, train_labels,
    epochs=50,
    batch_size=32,
    validation_data=(val_images, val_labels),
    callbacks=callbacks,
    class_weight=class_weight_dict
)

# VERIFICATION

print("Combined images shape:", combined_images.shape)
print("Combined labels shape:", combined_labels.shape)

# Sample image
plt.imshow(combined_images[0].reshape(28, 28), cmap='gray')
plt.title(f"Label: {combined_labels[0]}")
plt.show()

test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f"Test loss: {test_loss}, Test accuracy: {test_accuracy}")

y_pred = model.predict(test_images)
y_pred_classes = np.argmax(y_pred, axis=1)

# Classification report
report = classification_report(test_labels, y_pred_classes, output_dict=True)
print(classification_report(test_labels, y_pred_classes))

# Confusion Matrix
cm = confusion_matrix(test_labels, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Visualize predictions on custom class
indices = [i for i, label in enumerate(test_labels) if label == 10]
random_indices = random.sample(indices, min(len(indices), 5))

for idx in random_indices:
    plt.imshow(test_images[idx].reshape(28, 28), cmap='gray')
    plt.title(f"True: {test_labels[idx]}, Predicted: {y_pred_classes[idx]}")
    plt.show()

# Loss and accuracy curves
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy Curves')
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss Curves')
plt.show()

# Performance on custom class (label 10)
key = '10' if '10' in report else 10
print("Performance on custom class (label 10):")
print(f"Precision: {report[str(key)]['precision']}")
print(f"Recall: {report[str(key)]['recall']}")
print(f"F1-Score: {report[str(key)]['f1-score']}")

model.save('math_recog_v0.h5')
print("Model saved to math_recog_v0.h5")
