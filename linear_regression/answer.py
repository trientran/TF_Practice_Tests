import numpy as np
from keras import Sequential
from keras.layers import Dense


def regression_model():
    # Define the input and output data (corresponding to "y = 10x + 5")
    x_array = np.array([-1, 0, 1, 2, 3, 4, 5, 6], dtype=int)
    y_array = np.array([-5, 5, 15, 25, 35, 45, 55, 65], dtype=int)

    # Define the model architecture
    model = Sequential([
        Dense(units=1, input_shape=[1])
    ])

    # Compile the model
    model.compile(optimizer='sgd', loss='mean_squared_error')

    # Train the model
    model.fit(x_array, y_array, epochs=1000)

    return model
