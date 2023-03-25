# Copyright here
import tensorflow as tf
import numpy as np


def regression_model():
    x_array = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    y_array = np.array([-7.0, -3.0, 1.0, 5.0, 9.0, 13.0], dtype=float)

    # Define the model architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=1, input_shape=[1])
    ])

    # Compile the model
    model.compile(optimizer='sgd', loss='mean_squared_error')

    # Train the model
    model.fit(x_array, y_array, epochs=1000)

    return model


# Run and save your model
my_model = regression_model()
model_name = "regression_model_1.h5"
my_model.save(model_name)

# Reload the saved model
saved_model = tf.keras.models.load_model(model_name)

# Show the model architecture
saved_model.summary()

# Test the model on some new data
x_test = np.array([7, 8, 9, 10])
y_test = np.array([75, 85, 95, 105])
predictions = saved_model.predict(x_test)

# Print the predictions and expected values
for i in range(len(x_test)):
    print("x = {:.0f}, expected y = {:.0f}, predicted y = {:.0f}".format(x_test[i], y_test[i], predictions[i][0]))

# Evaluate the model on the test data
test_loss = saved_model.evaluate(x_test, y_test)
print("Test loss: {:.2f}".format(test_loss))
