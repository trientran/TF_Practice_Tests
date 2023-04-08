# Copyright (c) 2023, Trien Phat Tran (Mr. Troy).

# Question:

# Build and train a Sequential model that can predict the level of humidity for My City using the my-city-humidity.csv
# dataset. The normalized dataset should have a mean absolute error (MAE) of 0.13 or less.

# Your task is to fill in the missing parts of the code block (where commented as "YOUR CODE HERE").


import csv
import os
from urllib.request import urlretrieve

import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.callbacks import EarlyStopping, Callback
from keras.layers import Conv1D, LSTM, Dense
from keras.losses import Huber
from keras.optimizers import SGD
from keras.saving.save import load_model


def sequences_model():
    csv_file = 'my-city-humidity.csv'
    if not os.path.exists(csv_file):
        url = 'https://trientran.github.io/tf-practice-exams/my-city-humidity.csv'
        urlretrieve(url=url, filename=csv_file)

    humidity = []

    with open(csv_file) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for row in reader:
            humidity.append(float(row[2]))

    series = np.array(humidity)

    # Normalize the data
    min_value = np.min(series)
    max_value = np.max(series)
    series -= min_value
    series /= max_value

    # The data is split into training and validation sets at time step 2900 (~90% of the number of records). YOU MUST
    # REPLACE THIS VALUE WITH THE ONE GIVEN IN THE REAL TEST BECAUSE THE REAL TEST'S DATASET MAY BE BIGGER OR SMALLER
    split_time = 2900

    # In this particular problem, we only need to predict the sunspot activity based on the previous values of the
    # series, so we don't need to include the time step as a feature in the model. Therefore, we only use the x_train
    # and x_valid variables (not time_train nor time_valid), which contain the normalized sunspot activity values for
    # the training and validation sets.
    x_train = series[:split_time]
    x_valid = series[split_time:]

    # DO NOT CHANGE THIS CODE
    window_size = 30
    batch_size = 32
    shuffle_buffer_size = 1000

    train_set = windowed_dataset(
        series=x_train,
        window_size=window_size,
        batch_size=batch_size,
        shuffle_buffer=shuffle_buffer_size
    )

    valid_set = windowed_dataset(
        series=x_valid,
        window_size=window_size,
        batch_size=batch_size,
        shuffle_buffer=shuffle_buffer_size
    )

    model = Sequential([
        Conv1D(filters=60, kernel_size=5, strides=1, padding="causal", activation="relu", input_shape=[None, 1]),
        LSTM(60, return_sequences=True),
        LSTM(60, return_sequences=True),
        Dense(units=30, activation="relu"),
        Dense(units=10, activation="relu"),
        Dense(units=1)
    ])

    # Compile and fit data to the model
    # Compile the model
    model.compile(loss=Huber(), optimizer=SGD(learning_rate=1e-5, momentum=0.9), metrics=["mae"])

    # Optional: Define an early stopping callback.
    early_stop = EarlyStopping(monitor='val_mae', mode='min', patience=10, verbose=1, min_delta=0.005)

    # Fit the model
    model.fit(x=train_set, epochs=200, verbose=1, validation_data=valid_set, callbacks=[MyCallback()])

    return model


# If you are aiming at achieving a certain limit of Mean Absolute Error, this callback class will be handy.
class MyCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        val_mae = logs.get('val_mae')
        if val_mae <= 0.12:  # Very importantly, you must change this number if the test expects a certain limit of MAE
            print(f"\nReached {val_mae} Mean Absolute Error after {epoch} epochs so stopping training!")
            self.model.stop_training = True


# ===============DO NOT EDIT THIS PART================================
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(size=window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    return ds.batch(batch_size).prefetch(1)


if __name__ == '__main__':
    # Run and save your model
    my_model = sequences_model()
    filepath = "sequences_model.h5"
    my_model.save(filepath)

    # Reload the saved model
    saved_model = load_model(filepath)

    # Show the model architecture
    saved_model.summary()
