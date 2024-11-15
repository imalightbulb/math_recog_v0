
import os
import numpy as np
import keras
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image, ImageOps

# Preprocess Function (consistent with training)
def preprocess_img(image_path):
    image = Image.open(image_path).convert('L')
    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    image = ImageOps.invert(image)
    image_array = np.array(image) / 255.0
    image_array = image_array.reshape(28, 28, 1)
    return image_array

# Load the trained model
model = keras.models.load_model('math_recog_v0.h5')
print("Model loaded.")

# Directory containing new images
new_img_dir = 'new_img/'

# Lists to store preprocessed images and filenames
images_list = []
filenames_list = []

# Iterate over the files in the directory
for filename in os.listdir(new_img_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(new_img_dir, filename)
        try:
            image_array = preprocess_img(image_path)
            images_list.append(image_array)
            filenames_list.append(filename)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    else:
        print(f"Skipped file (unsupported extension): {filename}")

# Convert list to NumPy array
images_array = np.array(images_list)
print(f"Total images to predict: {len(images_array)}")

# Make predictions
predictions = model.predict(images_array)
predicted_labels = np.argmax(predictions, axis=1)

# Create a label mapping
label_mapping = {
    0: '0',
    1: '1',
    2: '2',
    3: '3',
    4: '4',
    5: '5',
    6: '6',
    7: '7',
    8: '8',
    9: '9',
    10: 'd/dx'
}

# Print predictions with confidence scores
for filename, probs in zip(filenames_list, predictions):
    predicted_label = np.argmax(probs)
    confidence = probs[predicted_label]
    class_name = label_mapping.get(predicted_label, 'Unknown')
    print(f"Image: {filename} --> Predicted Label: {class_name} (Confidence: {confidence * 100:.2f}%)")

# Optional: Visualize the images with predictions
for idx, (filename, image_array) in enumerate(zip(filenames_list, images_array)):
    plt.imshow(image_array.reshape(28, 28), cmap='gray')
    plt.title(f"Predicted: {label_mapping.get(predicted_labels[idx], 'Unknown')}")
    plt.axis('off')
    plt.show()
