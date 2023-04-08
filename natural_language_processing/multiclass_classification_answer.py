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


import json
import os

import numpy as np
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Embedding, Dropout, Conv1D, MaxPooling1D, LSTM, Dense
from keras.saving.save import load_model
from keras.utils import pad_sequences, to_categorical
from keras.utils.data_utils import urlretrieve
from keras_preprocessing.text import Tokenizer


def nlp_multiclass_model():
    json_file = 'language-classification.json'
    if not os.path.exists(json_file):
        url = 'https://trientran.github.io/tf-practice-exams/language-classification.json'
        urlretrieve(url=url, filename=json_file)

    max_length = 25
    trunc_type = 'pre'  # Can be replaced with 'post'
    vocab_size = 500
    padding_type = 'pre'  # Can be replaced with 'post'
    embedding_dim = 32
    oov_tok = "<OOV>"

    # Load the dataset
    with open(file=json_file, mode='r', encoding='utf-8') as f:
        datastore = json.load(f)

    # Extract the texts and labels
    texts = []
    labels = []
    for item in datastore:
        texts.append(item['sentence'])
        labels.append(item['language_code'])

    # Tokenize the texts
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    # Pad the sequences
    padded_sequences = pad_sequences(sequences=sequences, maxlen=max_length, padding=padding_type,
                                     truncating=trunc_type)

    # Convert the labels to numpy array
    labels = np.array(labels)

    # Split the dataset into training and validation sets
    num_samples = len(texts)
    num_train_samples = int(0.8 * num_samples)
    indices = np.random.permutation(num_samples)
    train_indices = indices[:num_train_samples]
    val_indices = indices[num_train_samples:]
    x_train = padded_sequences[train_indices]
    y_train = labels[train_indices]
    x_val = padded_sequences[val_indices]
    y_val = labels[val_indices]

    # Define the number of classes
    num_classes = len(set(labels))

    # Build the model
    model = Sequential([
        Embedding(input_dim=vocab_size + 1, output_dim=embedding_dim, input_length=max_length),
        Dropout(rate=0.2),
        Conv1D(filters=64, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=4),
        LSTM(64),
        Dense(units=num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Define an early stopping callback. Feel free to adjust the parameters' values if you want to fine-tune this model
    early_stop = EarlyStopping(monitor='val_accuracy', patience=5, min_delta=0.01, verbose=1)

    # Train the model
    y_train_categorical = to_categorical(y=y_train, num_classes=num_classes)
    y_val_categorical = to_categorical(y=y_val, num_classes=num_classes)
    model.fit(
        x=x_train,
        y=y_train_categorical,
        epochs=50,
        validation_data=(x_val, y_val_categorical),
        callbacks=[early_stop]
    )

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
