# Copyright (c) 2023, Trien Phat Tran (Mr. Troy).

# Question:

# Build and train a binary classifier for the language classification dataset. The dataset is typically a JSON array
# of 500 JSON objects. Each object has 3 keys: sentence, language_code, and is_english.
# We want our model to be able to determine whether a piece of text is "English or not".
# Condition: The final layer of the model should have 1 neuron activated by the sigmoid function.

# Your task is to fill in the missing parts of the code block (where commented as "YOUR CODE HERE").


import os
from urllib.request import urlretrieve

from keras import Sequential
from keras.saving.save import load_model


def nlp_binary_model():
    json_file = 'language-classification.json'
    if not os.path.exists(json_file):
        url = 'https://trientran.github.io/tf-practice-exams/language-classification.json'
        urlretrieve(url=url, filename=json_file)

    max_length = 25
    trunc_type = 'pre'  # May be replaced with 'post'
    vocab_size = 500
    padding_type = 'pre'  # May be replaced with 'post'
    embedding_dim = 32
    oov_tok = "<OOV>"

    # Load the dataset
    # YOUR CODE HERE

    # Extract the texts and labels
    texts = []
    labels = []
    # YOUR CODE HERE

    # Define the model architecture
    model = Sequential([
        # YOUR CODE HERE
    ])

    # Compile and fit data to the model
    # YOUR CODE HERE

    return model


# ===============DO NOT EDIT THIS PART================================
if __name__ == '__main__':
    # Run and save your model
    my_model = nlp_binary_model()
    filepath = "nlp_binary_model.h5"
    my_model.save(filepath)

    # Reload the saved model
    saved_model = load_model(filepath)

    # Show the model architecture
    saved_model.summary()
