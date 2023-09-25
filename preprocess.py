from PIL import Image
import os
import numpy as np
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        # Set memory growth for each GPU
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    except RuntimeError as e:
        pass

# Step 1: Load Images from Directory
data_dir = 'E:\\PlantIdentification\\dataset'
class_labels = os.listdir(data_dir)  # List of class labels (folder names)
image_data = []
labels = []

for label in class_labels:
    label_dir = os.path.join(data_dir, label)
    for filename in os.listdir(label_dir):
        image_path = os.path.join(label_dir, filename)
        image = Image.open(image_path)
        image_data.append(image)
        labels.append(label)

# Step 2: Resize Images
target_size = (224, 224)
resized_images = [image.resize(target_size) for image in image_data]

images_arrays= [np.array(image) for image in resized_images]
'''
grayscale_images = [image.convert('L') for image in resized_images]
 
# Step 3: Convert Images to NumPy Arrays
grayscale_arrays = [np.array(image) for image in grayscale_images]

grayscale_arrays = [np.expand_dims(image, axis=-1) for image in grayscale_arrays]

# Stack the images along a new batch dimension
image_arrays = np.stack(grayscale_arrays, axis=0)

images_arrays = image_arrays.reshape(-1, 224, 224, 1)
'''
#print(image_arrays.shape)


# Step 4: Data Augmentation (optional)

from PIL import ImageFilter

smoothed_images = [image.filter(ImageFilter.GaussianBlur(radius=2)) for image in resized_images]

import matplotlib.pyplot as plt
'''
plt.imshow(image_arrays[0])
plt.title(f'Class: {labels[0]}')
plt.show() '''

def datasets():
    return images_arrays, labels