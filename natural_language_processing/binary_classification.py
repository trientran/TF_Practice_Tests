import json
from urllib.request import urlretrieve
import numpy as np
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Embedding, GlobalAveragePooling1D, Dense
from keras.preprocessing.text import Tokenizer
from keras.saving.save import load_model
from keras.utils import pad_sequences


def nlp_binary_model():
    url = 'https://trientran.github.io/tf-practice-exams/is-english.json'
    urlretrieve(url, 'is-english.json')

    # Load the dataset
    with open('is-english.json', 'r') as f:
        datastore = json.load(f)

    # Extract the texts and labels
    texts = []
    labels = []
    for item in datastore:
        texts.append(item['sentence'])
        labels.append(item['is_english'])

    # Tokenize the texts
    tokenizer = Tokenizer(oov_token='<OOV>')
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    # Pad the sequences
    padded_sequences = pad_sequences(sequences, padding='post')

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

    # Build the model
    model = Sequential([
        Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=32),
        GlobalAveragePooling1D(),
        Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Define an early stopping callback
    early_stop = EarlyStopping(monitor='val_accuracy', patience=5, min_delta=0.01, verbose=1)

    # Train the model
    model.fit(x_train, y_train, epochs=50, validation_data=(x_val, y_val), callbacks=[early_stop])

    return model


# ===============DO NOT EDIT THIS PART================================
if __name__ == '__main__':
    # Run and save your model
    my_model = nlp_binary_model()
    model_name = "nlp_binary_model.h5"
    my_model.save(model_name)

    # Reload the saved model
    saved_model = load_model(model_name)

    # Show the model architecture
    saved_model.summary()
