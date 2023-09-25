import tensorflow as tf
from keras.preprocessing import image
import numpy as np
from CNN import uniqueLabel


# Load the pre-trained model
model = tf.keras.models.load_model('E:\PlantIdentification\identify_model.keras')

# Load and preprocess the input image
img_path = 'testing/test1.jpeg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = tf.keras.applications.mobilenet.preprocess_input(img_array)  # Adjust this preprocessing function as needed for your model

# Make predictions
predictions = model.predict(img_array)

print(predictions)

# Get the predicted class label
predicted_class = np.argmax(predictions, axis=1)

#print(predicted_class)
# Map the class index to the flower name or label
#flower_labels = ['label1', 'label2', ...]  # Replace with your actual labels
#predicted_flower = flower_labels[predicted_class[0]]

# Assuming predictions is a list of class indices
#decoded_predictions = [label_mapping[prediction] for prediction in predictions]

#labels=datasets()[1]
#predicted_plant = labels[predicted_class[0]]

#unique_labels = set(labels)

# Convert the set back to a list if needed
plant_labels = uniqueLabel()
predicted_plant=plant_labels[predicted_class[0]]
#print(plant_labels)

print(f"The model predicts that the plant is: {predicted_plant}")
