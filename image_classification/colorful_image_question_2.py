# Copyright (c) 2023, Trien Phat Tran (Mr. Troy).
# The Malaria dataset available on the official National Institutes of Health (NIH) website is in the public domain and
# does not have any specific license or copyright restrictions.

# Binary (2-classes) image classification
# Dataset: Malaria.
# Direct link 1: https://data.lhncbc.nlm.nih.gov/public/Malaria/cell_images.zip (~350 Megabytes)
# This dataset comprises 2 classes namely Parasitized and Uninfected, and it is not split into training and test sets
# yet. The images' resolutions are varied.
# Create a classifier for the given dataset. The required input shape must be 40x40x3 (RGB images).

# Your task is to fill in the missing parts of the code block (where commented as "YOUR CODE HERE").

import os

from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.saving.save import load_model
from keras.utils import image_dataset_from_directory, get_file


def binary_model():
    dataset_url = 'https://data.lhncbc.nlm.nih.gov/public/Malaria/cell_images.zip'
    data_folder = '/tmp/cell_images'
    if not os.path.exists(data_folder):
        # download and extract the dataset
        zip_path = get_file('cell_images.zip', dataset_url, extract=True, cache_subdir='tmp')
        os.rename(os.path.join(os.path.dirname(zip_path), 'cell_images'), data_folder)

    # Define image size and batch size
    img_size =  # YOUR CODE HERE
    batch_size =  # YOUR CODE HERE

    # Create the training dataset
    train_ds = image_dataset_from_directory(
        # YOUR CODE HERE
    )

    # Create the validation dataset
    val_ds = image_dataset_from_directory(
        # YOUR CODE HERE
    )

    # Define the model architecture
    model = Sequential([
        # YOUR CODE HERE
    ])

    # YOUR CODE HERE

    return model


# ===============DO NOT EDIT THIS PART================================
if __name__ == '__main__':
    # Run and save your model
    my_model = binary_model()
    model_name = "binary_model.h5"
    my_model.save(model_name)

    # Reload the saved model
    saved_model = load_model(model_name)

    # Show the model architecture
    saved_model.summary()
