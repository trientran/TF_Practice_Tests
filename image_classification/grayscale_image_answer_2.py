# Copyright (c) 2023, Trien Phat Tran (Mr. Troy).
# Yann LeCun and Corinna Cortes hold the copyright of MNIST dataset, which is a derivative work from original NIST
# datasets. MNIST dataset is made available under the terms of the Creative Commons Attribution-Share Alike 3.0 license.

# Question:

# Create a classifier for the MNIST dataset which includes black-and-white images of 10 digits (0-9). The input shape
# should be 28x28 monochrome and the model should classify 10 classes.

# Your task is to fill in the missing parts of the code block (where commented as "YOUR CODE HERE").

import tensorflow as tf
import tensorflow_datasets as tfds
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.saving.save import load_model


# Use Tensorflow datasets
def my_model():
    # Load the MNIST dataset
    (train_ds, test_ds), info = tfds.load(name='mnist', split=['train', 'test'], with_info=True, as_supervised=True)

    # YOUR CODE HERE
    # Normalize the data
    def normalize(image, label):
        return tf.cast(x=image, dtype=tf.float32) / 255.0, label

    # Preprocess the training data
    train_ds = train_ds.map(normalize).cache().shuffle(info.splits['train'].num_examples).batch(32)

    # Preprocess the test data
    test_ds = test_ds.map(normalize).cache().batch(32)

    # Define the model architecture
    model = Sequential([
        Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(units=64, activation='relu'),
        Dense(units=10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Define the early stopping callback
    early_stop = EarlyStopping(monitor='val_accuracy', patience=5, min_delta=0.01, verbose=1)

    # Train the model
    model.fit(x=train_ds, epochs=10, validation_data=test_ds, callbacks=[early_stop])

    return model


# ===============DO NOT EDIT THIS PART================================
if __name__ == '__main__':
    # Run and save your model
    my_model = my_model()
    filepath = "grayscale_model_2.h5"
    my_model.save(filepath)

    # Reload the saved model
    saved_model = load_model(filepath)

    # Show the model architecture
    saved_model.summary()
