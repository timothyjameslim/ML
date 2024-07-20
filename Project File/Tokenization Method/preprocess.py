import pandas as pd
import numpy as np
import tensorflow as tf
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

# Load the train dataset from a text file
train_data_file = 'Genre Classification Dataset/train_data.txt'
train_data = pd.read_csv(train_data_file, sep=' ::: ', names=['ID', 'Title', 'Genre', 'Description'], engine='python')
train_data.drop(columns=['ID'], inplace=True)

# Load the test dataset from a text file
test_data_file = 'Genre Classification Dataset/test_data.txt'
test_data = pd.read_csv(test_data_file, sep=' ::: ', names=['ID', 'Title', 'Description'], engine='python')
test_data.drop(columns=['ID'], inplace=True)

# Load the test dataset from a text file
test_sol_file = 'Genre Classification Dataset/test_data_solution.txt'
test_sol = pd.read_csv(test_sol_file, sep=' ::: ', names=['ID', 'Title', 'Genre', 'Description'], engine='python')
test_sol.drop(columns=['ID'], inplace=True)

# Ensure the solutions are aligned with the test data
actual_labels = test_sol['Genre'].tolist()

# Clean and preprocess data
train_data_cleaned = train_data.dropna().drop_duplicates()
test_data_cleaned = test_data.dropna().drop_duplicates()

# Combine title and description for training data
train_text = train_data_cleaned['Title'] + " " + train_data_cleaned['Description']
train_label = train_data_cleaned['Genre']

# Combine title and description for test data
test_text = test_data_cleaned['Title'] + " " + test_data_cleaned['Description']

# Tokenize the text, we vectorize the text. based on TF-IDF
tokzer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokzer.fit_on_texts(train_text)

# Create a dictionary of words using text to sequence, it replaces each unique word with an index
train_sequences = tokzer.texts_to_sequences(train_text)
test_sequences = tokzer.texts_to_sequences(test_text)

# Padding of sequence to ensure uniform inputs
train_padded = pad_sequences(train_sequences, padding='post', maxlen=200)
test_padded = pad_sequences(test_sequences, padding='post', maxlen=200)

# Encode the labels
label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(train_label)
train_labels_encoded = label_tokenizer.texts_to_sequences(train_label)
train_labels_encoded = [label[0] for label in train_labels_encoded]  # Flatten the list of lists
train_labels_encoded = tf.keras.utils.to_categorical(train_labels_encoded, num_classes=len(label_tokenizer.word_index) + 1)

# Save preprocessed data
np.save('Prepro/train_padded.npy', train_padded)
np.save('Prepro/test_padded.npy', test_padded)
np.save('Prepro/train_labels_encoded.npy', train_labels_encoded)
np.save('Prepro/actual_labels.npy', actual_labels)
with open('Prepro/tokenizer.pkl', 'wb') as f:
    import pickle
    pickle.dump(tokzer, f)
with open('Prepro/label_tokenizer.pkl', 'wb') as f:
    pickle.dump(label_tokenizer, f)