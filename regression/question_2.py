# Copyright here

# Question:

# You have been given two arrays: x_array and y_array, each containing a number of floating-point values.
# The x_array contains input values and y_array contains corresponding output values. Using TensorFlow,
# create a neural network model that can predict the output of a given input value x based on the relationship
# between x and y.

# Your task is to fill in the missing parts of the regression_model function (where commented as "YOUR CODE HERE").

import tensorflow as tf
import numpy as np


def regression_model():
    # Define the input and output data (corresponding to "y = 0.3x - 4.7")
    x_array = np.array([-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0], dtype=float)
    y_array = np.array([-7.0, -6.5, -4.7, -3.5, -2.0, 0.5, 3.2, 6.5], dtype=float)

    # Define the model architecture
    model = tf.keras.Sequential([
        # YOUR CODE HERE
    ])

    # Compile the model
    # YOUR CODE HERE

    # Train the model
    # YOUR CODE HERE

    return model


# Run and save your model
my_model = regression_model()
model_name = "regression_model_2.h5"
my_model.save(model_name)

# Reload the saved model
saved_model = tf.keras.models.load_model(model_name)

# Show the model architecture
saved_model.summary()

# Test the model on some new data
x_test = np.array([-15.0, -2.0, 7.0, 18.0])
y_test = np.array([-9.0, -4.1, -2.4, 1.4])
predictions = saved_model.predict(x_test)

# Print the predictions and expected values
for i in range(len(x_test)):
    print("x = {:.1f}, expected y = {:.1f}, predicted y = {:.1f}".format(x_test[i], y_test[i], predictions[i][0]))

# Evaluate the model on the test data
test_loss = my_model.evaluate(x_test, y_test)
print("Test loss: {:.2f}".format(test_loss))
