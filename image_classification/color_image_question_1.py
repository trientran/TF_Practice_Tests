# Copyright (c) 2023, Trien Phat Tran (Mr. Troy).

# Question:

# Multiclass image classification
# Dataset: Mr Troy Fruits.
# Direct link: https://trientran.github.io/images/mr-troy-fruits.zip (~11 Megabytes)
# This dataset comprises 3 classes (Banana, Orange, and Apple), and it is not split into training and test sets yet.
# Create a classifier for the given dataset. The required input shape must be 200x200x3 (RGB images).

# Your task is to fill in the missing parts of the code block (where commented as "YOUR CODE HERE").


import os
import zipfile
from urllib.request import urlretrieve

from keras import Sequential
from keras.layers import Dense
from keras.saving.save import load_model
from keras_preprocessing.image import ImageDataGenerator


def multiclass_model():
    # After the data set has been downloaded below, we must double-check if the extracted data folder path is correct
    # by looking at the project browser on the left side. If not, please change it accordingly!
    data_folder = "data/mr-troy-fruits"

    # download and extract the dataset if not existing
    if not os.path.exists(data_folder):
        dataset_url = 'https://trientran.github.io/images/mr-troy-fruits.zip'
        local_zip = 'mr-troy-fruits.zip'
        urlretrieve(dataset_url, local_zip)
        zip_ref = zipfile.ZipFile(local_zip, 'r')
        zip_ref.extractall('data/')
        zip_ref.close()

    training_datagen = ImageDataGenerator(
        # YOUR CODE HERE
    )

    train_generator =  # YOUR CODE HERE

    validation_generator =  # YOUR CODE HERE

    model = Sequential([
        # YOUR CODE HERE
        Dense(3, activation='softmax')
    ])

    # Compile and fit data to the model
    # YOUR CODE HERE

    return model


# ===============DO NOT EDIT THIS PART================================
if __name__ == '__main__':
    # Run and save your model
    my_model = multiclass_model()
    model_name = "multiclass_model.h5"
    my_model.save(model_name)

    # Reload the saved model
    saved_model = load_model(model_name)

    # Show the model architecture
    saved_model.summary()
