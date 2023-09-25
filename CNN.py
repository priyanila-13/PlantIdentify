import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from preprocess import datasets

image_arrays, labels = datasets()

# Convert labels to numerical format using LabelEncoder
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

num_classes = len(label_encoder.classes_)

unique_labels = []
unique_encoded_labels=[]

# Iterate through the original array and add elements to unique_labels if they are not already in the list
for label in labels:
    if label not in unique_labels:
        unique_labels.append(label)

for label in encoded_labels:
    if label not in unique_encoded_labels:
        unique_encoded_labels.append(label)

label_dict = {unique_encoded_labels[i]: unique_labels[i] for i in range(len(unique_labels))}

def labelDict():
    return label_dict

def uniqueLabel():
    return unique_labels

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(np.array(image_arrays), encoded_labels, test_size=0.2, random_state=42)

# Define the CNN model
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(num_classes, activation='softmax')  # num_classes is the number of unique labels
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Evaluate the model on the validation set
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f'Validation loss: {val_loss}, Validation accuracy: {val_accuracy}')

model.save('E:\PlantIdentification\identify_model.keras')

