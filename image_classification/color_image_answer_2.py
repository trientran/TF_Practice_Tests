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
    data_folder = 'temp/cell_images/'
    if not os.path.exists(data_folder):
        # download and extract the dataset
        zip_path = get_file(fname='cell_images.zip',origin= dataset_url, extract=True, cache_subdir='tmp')
        os.rename(os.path.join(os.path.dirname(zip_path), 'cell_images'), data_folder)

    # Define image size and batch size
    img_size = (40, 40)
    batch_size = 32

    # Create the training dataset
    train_ds = image_dataset_from_directory(
        directory=data_folder,
        validation_split=0.2,
        subset='training',
        seed=42,
        image_size=img_size,
        batch_size=batch_size
    )

    # Create the validation dataset
    val_ds = image_dataset_from_directory(
        directory=data_folder,
        validation_split=0.2,
        subset='validation',
        seed=42,
        image_size=img_size,
        batch_size=batch_size
    )

    # Define the model architecture
    model = Sequential([
        Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(units=64, activation='relu'),
        Dense(units=2, activation='softmax')
        # Dense(1, activation='sigmoid') # The last layer can be replaced with this but expect different output
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # Replace sparse_categorical_crossentropy with binary_crossentropy if sigmoid activation function is used

    # Define the early stopping callback for val_accuracy
    early_stop = EarlyStopping(monitor='val_accuracy', patience=5, verbose=1, min_delta=0.01)

    # Train the model with early stopping callback
    model.fit(x=train_ds, epochs=15, validation_data=val_ds, callbacks=[early_stop])

    return model


# ===============DO NOT EDIT THIS PART================================
if __name__ == '__main__':
    # Run and save your model
    my_model = binary_model()
    filepath = "binary_rgb_model.h5"
    my_model.save(filepath)

    # Reload the saved model
    saved_model = load_model(filepath)

    # Show the model architecture
    saved_model.summary()
