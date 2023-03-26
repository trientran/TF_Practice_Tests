# Copyright (c) 2023, Trien Phat Tran (Mr. Troy).
# Yann LeCun and Corinna Cortes hold the copyright of MNIST dataset, which is a derivative work from original NIST
# datasets. MNIST dataset is made available under the terms of the Creative Commons Attribution-Share Alike 3.0 license.

# Question:

# Create a classifier for the MNIST dataset. The input shape should be 28x28 monochrome and the model should
# classify 10 classes. Do not resize the data, and make sure your input layer accepts (28,28) as the input shape only.

# Your task is to fill in the missing parts of the code block (where commented as "YOUR CODE HERE").

import tensorflow_datasets as tfds
import tensorflow as tf
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.saving.save import load_model


# Use Keras dataset
def model_1():
    # Load the MNIST dataset
    dataset = mnist
    (x_train, y_train), (x_test, y_test) = dataset.load_data()

    # Normalize the input data
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Define the model architecture
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Define the early stopping callback
    early_stop = EarlyStopping(monitor='val_accuracy', patience=1, verbose=1, min_delta=0.01, baseline=0.99)

    # Train the model with the early stopping callback
    model.fit(x_train.reshape(-1, 28, 28, 1), y_train, epochs=10,
              validation_data=(x_test.reshape(-1, 28, 28, 1), y_test), callbacks=[early_stop])

    return model


# Use Tensorflow datasets
def model_2():
    # Load the MNIST dataset
    (train_ds, test_ds), info = tfds.load(name='mnist', split=['train', 'test'], with_info=True, as_supervised=True)

    # Normalize the data
    def normalize(image, label):
        return tf.cast(image, tf.float32) / 255.0, label

    # Preprocess the training data
    train_ds = train_ds.map(normalize).cache().shuffle(info.splits['train'].num_examples).batch(32)

    # Preprocess the test data
    test_ds = test_ds.map(normalize).cache().batch(32)

    # Define the model architecture
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Define the early stopping callback
    early_stop = EarlyStopping(monitor='val_accuracy', patience=1, verbose=1, min_delta=0.01, baseline=0.99)

    # Train the model
    model.fit(train_ds, epochs=10, validation_data=test_ds, callbacks=[early_stop])

    return model


# ===============DO NOT EDIT THIS PART================================
if __name__ == '__main__':
    # Run and save your model
    my_model = model_2()
    model_name = "grayscale_model.h5"
    my_model.save(model_name)

    # Reload the saved model
    saved_model = load_model(model_name)

    # Show the model architecture
    saved_model.summary()
