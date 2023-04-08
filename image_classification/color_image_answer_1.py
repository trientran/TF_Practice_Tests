# Copyright (c) 2023, Trien Phat Tran (Mr. Troy).

# Question:

# Multiclass image classification
# Dataset: Mr Troy Fruits.
# Direct link: http://dl.dropboxusercontent.com/s/a32yc71tgfvfvku/mr-troy-fruits.zip (~11 Megabytes)
# Back-up direct link: https://trientran.github.io/images/mr-troy-fruits.zip
# This dataset comprises 3 classes (Banana, Orange, and Apple), and it is not split into training and test sets yet.
# Create a classifier for the given dataset. The required input shape must be 200x200x3 (RGB images).

# Your task is to fill in the missing parts of the code block (where commented as "YOUR CODE HERE").


import os
import zipfile
from urllib.request import urlretrieve

from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from keras.saving.save import load_model
from keras_preprocessing.image import ImageDataGenerator


def multiclass_model():
    # After the data set has been downloaded below, we must double-check if the extracted data folder path is correct
    # by looking at the project browser on the left side. If not, please change it accordingly!
    data_folder = "temp/mr-troy-fruits/"

    # download and extract the dataset if not existing
    if not os.path.exists(data_folder):
        dataset_url = 'http://dl.dropboxusercontent.com/s/a32yc71tgfvfvku/mr-troy-fruits.zip'
        local_zip = 'mr-troy-fruits.zip'
        urlretrieve(url=dataset_url, filename=local_zip)
        zip_ref = zipfile.ZipFile(file=local_zip, mode='r')
        zip_ref.extractall(path='temp/')
        zip_ref.close()

    training_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    # Very important: we must set the input image size (input shape/input layer) as required in the Question.
    # For example, if the question mentions the required image size of 200x200 or 100x100 or 150x150, we must set that
    # value here
    img_size = (200, 200)

    train_generator = training_datagen.flow_from_directory(
        directory=data_folder,
        target_size=img_size,
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = training_datagen.flow_from_directory(
        directory=data_folder,
        target_size=img_size,
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    model = Sequential([
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dropout(rate=0.5),
        Dense(units=512, activation='relu'),
        Dense(units=3, activation='softmax')  # 3 is the number of classes in the dataset.
    ])

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    # Define an early stopping callback
    early_stop = EarlyStopping(monitor='val_accuracy', patience=5, verbose=1, min_delta=0.01)

    model.fit(x=train_generator, epochs=15, validation_data=validation_generator, callbacks=[early_stop])

    return model


# ===============DO NOT EDIT THIS PART================================
if __name__ == '__main__':
    # Run and save your model
    my_model = multiclass_model()
    filepath = "multiclass_rgb_model.h5"
    my_model.save(filepath)

    # Reload the saved model
    saved_model = load_model(filepath)

    # Show the model architecture
    saved_model.summary()
