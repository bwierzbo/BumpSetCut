import os
import cv2 as cv
import numpy as np
import keras
import tensorflow as tf
from keras.models import load_model
import shutil

# Load your trained model

modelpath = ("models/windowsAdvanced.keras")
print(os.path.exists(modelpath))
model = load_model(modelpath)

# Directories
image_folder_path = 'images/back'
ball_path = os.path.join(image_folder_path, 'ball')
not_ball_path = os.path.join(image_folder_path, 'notBall')

# Ensure output directories exist
os.makedirs(ball_path, exist_ok=True)
os.makedirs(not_ball_path, exist_ok=True)

def preprocess_for_model(img, size=(224, 224)):
    img = cv.resize(img, size)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # Model expects batch dimension
    return img

# Iterate through all images in the folder
for image_name in os.listdir(image_folder_path):
    image_path = os.path.join(image_folder_path, image_name)
    
    # Make sure to process files only (skip subdirectories)
    if os.path.isfile(image_path):
        # Load and preprocess the image
        img = cv.imread(image_path)
        if img is not None:  # Proceed if the image is loaded successfully
            preprocessed_img = preprocess_for_model(img)

            # Predict
            prediction = model.predict(preprocessed_img)
            predicted_class = 'notBall' if prediction[0][0] > 0.5 else 'ball'
            print(f"Predicted: {predicted_class} for {image_name} with confidence {prediction[0][0]}")

            # Move the image to the appropriate folder
            new_path = os.path.join(ball_path if predicted_class == 'ball' else not_ball_path, image_name)
            shutil.move(image_path, new_path)
        else:
            print(f"Failed to load image {image_name}. Skipping.")