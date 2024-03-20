import os
import cv2
import pandas as pd
import numpy as np
import random

import keras
import tensorflow as tf
from keras.applications import MobileNetV2
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt



# Define the base folder for your images and categories
base_folder = "images/"
categories = ['ball', 'notBall']

# Function to read and resize an image
def preprocess_image(filename, folder, size=(224, 224)):
    filepath = os.path.join(folder, filename)
    img = cv2.imread(filepath)
    img = cv2.resize(img, size)
    img = img / 255.0  # Normalize pixel values
    return img

# Process images and store the data in a list
data = []

for category in categories:
    folder = os.path.join(base_folder, category)
    for filename in os.listdir(folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):  # Check for image files
            processed_img = preprocess_image(filename, folder)
            data.append({
                'filename': filename,
                'category': category,
                'processed_image': processed_img
            })

# Create a DataFrame from the list
df = pd.DataFrame(data)

df['processed_image'] = df['processed_image'].apply(np.array)

# Convert 'category' to numerical labels
encoder = LabelEncoder()
df['category'] = encoder.fit_transform(df['category'])

# Separating features and labels
X = np.array(df['processed_image'].tolist())
y = df['category'].values

# Load pre-trained MobileNetV2
# Modify the base model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base model

# Build the model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu', kernel_regularizer=l2(0.01))(x)  # L2 Regularization
x = BatchNormalization()(x)  # Batch Normalization
x = Dropout(0.5)(x)  # Dropout
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Early Stopping
early_stop = EarlyStopping(monitor='val_loss', patience=5)


# Split data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Train the model with Data Augmentation
train_generator = datagen.flow(X_train, y_train, batch_size=32)
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=(X_val, y_val),
    callbacks=[early_stop]
)

model.save('my_model.keras')  # Save the model to disk
