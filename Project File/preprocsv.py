import pandas as pd
import tensorflow as tf
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import os

# Define the directory where the dataset files are located
dataset_dir = 'Genre Classification Dataset'

# Load the train dataset from a CSV file
train_data_file = os.path.join(dataset_dir, 'train_data_english.csv')
train_data = pd.read_csv(train_data_file)
train_data.drop(columns=['ID'], inplace=True)

# Clean and preprocess data
train_data_cleaned = train_data.dropna().drop_duplicates()

# Combine title and description for training data
train_text = train_data_cleaned['Title'] + " " + train_data_cleaned['Description']
train_label = train_data_cleaned['Genre']

# Tokenize the text
tokzer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokzer.fit_on_texts(train_text)

# Create a dictionary of words using text to sequence, it replaces each unique word with an index
train_sequences = tokzer.texts_to_sequences(train_text)

# Padding of sequence to ensure uniform inputs
train_padded = pad_sequences(train_sequences, padding='post', maxlen=200)

# Encode the labels
label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(train_label)
train_labels_encoded = label_tokenizer.texts_to_sequences(train_label)
train_labels_encoded = [label[0] for label in train_labels_encoded]  # Flatten the list of lists
train_labels_encoded = tf.keras.utils.to_categorical(train_labels_encoded, num_classes=len(label_tokenizer.word_index) + 1)

# Save preprocessed data
np.save('Prepro/train_padded_csv.npy', train_padded)
np.save('Prepro/train_labels_encoded_csv.npy', train_labels_encoded)
with open('Prepro/tokenizer_csv.pkl', 'wb') as f:
    pickle.dump(tokzer, f)
with open('Prepro/label_tokenizer_csv.pkl', 'wb') as f:
    pickle.dump(label_tokenizer, f)