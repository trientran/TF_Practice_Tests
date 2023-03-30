# Copyright (c) 2023, Trien Phat Tran (Mr. Troy).
# Yann LeCun and Corinna Cortes hold the copyright of MNIST dataset, which is a derivative work from original NIST
# datasets. MNIST dataset is made available under the terms of the Creative Commons Attribution-Share Alike 3.0 license.

# Question:

# Create a classifier for the MNIST dataset which includes black-and-white images of 10 digits (0-9). The input shape
# should be 28x28 monochrome and the model should classify 10 classes.

# Your task is to fill in the missing parts of the code block (where commented as "YOUR CODE HERE").

from keras.datasets import mnist
from keras.saving.save import load_model


# Use Keras dataset
def my_model():
    dataset = mnist

    # YOUR CODE HERE

    return model


# ===============DO NOT EDIT THIS PART================================
if __name__ == '__main__':
    # Run and save your model
    my_model = my_model()
    model_name = "grayscale_model_1.h5"
    my_model.save(model_name)

    # Reload the saved model
    saved_model = load_model(model_name)

    # Show the model architecture
    saved_model.summary()