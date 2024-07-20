import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import pickle

print("This is running Python 3.12")

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

# Vectorize the text using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=10000)
train_tfidf = tfidf_vectorizer.fit_transform(train_text).toarray()
test_tfidf = tfidf_vectorizer.transform(test_text).toarray()

# Encode the labels
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_label)
train_labels_encoded = tf.keras.utils.to_categorical(train_labels_encoded, num_classes=len(label_encoder.classes_))

# Save preprocessed data
np.save('Prepro/train_tfidf.npy', train_tfidf)
np.save('Prepro/test_tfidf.npy', test_tfidf)
np.save('Prepro/train_labels_encoded.npy', train_labels_encoded)
np.save('Prepro/actual_labels.npy', actual_labels)

with open('Prepro/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)
with open('Prepro/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)