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
        dataset_url = 'https://trientran.github.io/tf-practice-exams/mr-troy-fruits.zip'
        local_zip = 'mr-troy-fruits.zip'
        urlretrieve(dataset_url, local_zip)
        zip_ref = zipfile.ZipFile(local_zip, 'r')
        zip_ref.extractall('temp/')
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
        data_folder,
        target_size=img_size,
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = training_datagen.flow_from_directory(
        data_folder,
        target_size=img_size,
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dropout(0.5),
        Dense(512, activation='relu'),
        Dense(3, activation='softmax')  # 3 is the number of classes in the dataset.
    ])

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    # Define an early stopping callback
    early_stop = EarlyStopping(monitor='val_accuracy', patience=5, verbose=1, min_delta=0.01)

    model.fit(train_generator, epochs=15, validation_data=validation_generator, callbacks=[early_stop])

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
