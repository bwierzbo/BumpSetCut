import os
import shutil
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Define paths
base_dir = '../BumpSetCut/images/'  # Replace with the path to your dataset
classes = ['ball', 'notBall']
output_dirs = ['../BumpSetCut/images/train_dir', '../BumpSetCut/images/val_dir', '../BumpSetCut/images/test_dir']


# Desired size and normalization parameters
image_size = (224, 224)  # Example size, change as needed

# Split ratios
train_size = 0.7
val_size = 0.2
# Test size is implicitly determined as 0.1

# Creating train, val, and test directories
for d in output_dirs:
    path = os.path.join(base_dir, d)
    if not os.path.exists(path):
        os.makedirs(path)
        for c in classes:
            os.makedirs(os.path.join(path, c))

# Function to resize and normalize images, then copy them
def process_and_copy_files(files, class_name, dir_name):
    for f in files:
        # Load image
        src_path = os.path.join(base_dir, class_name, f)
        image = cv2.imread(src_path)
        if image is not None:
            # Resize image
            image = cv2.resize(image, image_size)

            # Normalize pixel values to be between 0 and 1
            image = image / 255.0

            # Save the processed image
            dst_path = os.path.join(base_dir, dir_name, class_name, f)
            cv2.imwrite(dst_path, image * 255)  # Multiply by 255 to revert normalization for saving

# Function to split data
def split_data(class_name):
    files = os.listdir(os.path.join(base_dir, class_name))
    train_files, test_files = train_test_split(files, train_size=train_size + val_size, random_state=42)
    val_files, test_files = train_test_split(test_files, train_size=val_size / (1 - train_size), random_state=42)

    # Process and copy files
    process_and_copy_files(train_files, class_name, 'train_dir')
    process_and_copy_files(val_files, class_name, 'val_dir')
    process_and_copy_files(test_files, class_name, 'test_dir')

# Splitting data for each class
for c in classes:
    split_data(c)

print("Data splitting and processing complete.")
