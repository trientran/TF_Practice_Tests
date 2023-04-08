# Copyright (c) 2023, Trien Phat Tran (Mr. Troy).

# Question:

# Build and train a binary classifier for the language classification dataset. The dataset is typically a JSON array
# of 500 JSON objects. Each object has 3 keys: sentence, language_code, and is_english.
# We want our model to be able to indicate which language a piece of text or sentence is written in.
# There are 5 languages need to be classified. Below is the language_code and its corresponding language name.
# 0: English
# 1: Vietnamese
# 2: Spanish
# 3: Portuguese
# 4: Italian

# Your task is to fill in the missing parts of the code block (where commented as "YOUR CODE HERE").


import os

from keras import Sequential
from keras.saving.save import load_model
from keras.utils.data_utils import urlretrieve


def nlp_multiclass_model():
    json_file = 'language-classification.json'
    if not os.path.exists(json_file):
        url = 'https://trientran.github.io/tf-practice-exams/language-classification.json'
        urlretrieve(url=url, filename=wjson_file)

    max_length = 25
    trunc_type = 'pre'  # Can be replaced with 'post'
    vocab_size = 500
    padding_type = 'pre'  # Can be replaced with 'post'
    embedding_dim = 32
    oov_tok = "<OOV>"

    # Load the dataset
    # YOUR CODE HERE

    # Extract the texts and labels
    texts = []
    labels = []
    # YOUR CODE HERE

    # Tokenize the texts
    # YOUR CODE HERE

    # Pad the sequences
    # YOUR CODE HERE

    # Convert the labels to numpy array
    # YOUR CODE HERE

    # Split the dataset into training and validation sets
    # YOUR CODE HERE

    # Define the number of classes
    # YOUR CODE HERE

    # Build the model
    model = Sequential([
        # YOUR CODE HERE
    ])

    # Compile and fit data to the model
    # YOUR CODE HERE

    return model


# ===============DO NOT EDIT THIS PART================================
if __name__ == '__main__':
    # Run and save your model
    my_model = nlp_multiclass_model()
    filepath = "nlp_multiclass_model.h5"
    my_model.save(filepath)

    # Reload the saved model
    saved_model = load_model(filepath)

    # Show the model architecture
    saved_model.summary()
