# Copyright (c) 2023, Trien Phat Tran (Mr. Troy).
# Makerere AI Lab hold the copyright of the iBean dataset under the MIT License.
# The Malaria dataset available on the official National Institutes of Health (NIH) website is in the public domain and
# does not have any specific license or copyright restrictions.


import zipfile
from urllib.request import urlretrieve

import tensorflow as tf
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras_preprocessing.image import ImageDataGenerator


# Multiclass image classification
# Dataset: iBean.
# Direct link: https://storage.googleapis.com/ibeans/train.zip (~137 Megabytes)
# This dataset comprises 3 classes, and it is not split into training and test sets yet. The images' sizes are varied.
# Create a classifier for the given dataset. The required input shape must be 150x150x3 (RGB images).

# Your task is to fill in the missing parts of the code block (where commented as "YOUR CODE HERE").

def multiclass_model():
    url = 'https://storage.googleapis.com/ibeans/train.zip'
    urlretrieve(url, 'iBean.zip')
    local_zip = 'iBean.zip'
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('iBean')
    zip_ref.close()

    training_dir = "iBean/train"
    training_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_generator = training_datagen.flow_from_directory(
        training_dir,
        target_size=(150, 150),
        class_mode='categorical'
    )

    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(3, activation='softmax')
    ])

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    model.fit(train_generator, epochs=20, callbacks=[early_stop])

    return model


# ===============DO NOT EDIT THIS PART================================
if __name__ == '__main__':
    # Run and save your model
    my_model = multiclass_model()
    model_name = "multiclass_model.h5"
    my_model.save(model_name)

    # Reload the saved model
    saved_model = tf.keras.models.load_model(model_name)

    # Show the model architecture
    saved_model.summary()
