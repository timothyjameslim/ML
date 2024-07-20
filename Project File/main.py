import pandas as pd
import tensorflow as tf
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import keras

# Load the train dataset from a text file
train_data_file = 'Genre Classification Dataset/train_data.txt'
train_data = pd.read_csv(train_data_file, sep=' ::: ', names=['ID', 'Title', 'Genre', 'Description'],
                         engine='python')
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

# Vectorize the text using TF-IDF and remove stopwords
tfidf_vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
train_tfidf = tfidf_vectorizer.fit_transform(train_text).toarray()
test_tfidf = tfidf_vectorizer.transform(test_text).toarray()

# Encode the labels
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_label)
train_labels_encoded = tf.keras.utils.to_categorical(train_labels_encoded, num_classes=len(label_encoder.classes_))

# Ask the user if the program should use an existing model or create a new model
new = int(input("Enter 1 to train a new model or 0 to use an existing model: "))

if new == 1:
    # Build the model
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=(train_tfidf.shape[1],)),  # Changed input_shape to shape
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    # Create a TensorBoard callback
    log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True)

    # Define the ModelCheckpoint callback
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath='best_model.keras',
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )

    # Define the EarlyStopping callback
    early_stopping_callback = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        verbose=1,
        mode='min',
        restore_best_weights=True
    )

    # Train the model with TensorBoard callback
    history = model.fit(train_tfidf, train_labels_encoded, epochs=20, validation_split=0.2, batch_size=32,
                        callbacks=[tensorboard_callback, checkpoint_callback, early_stopping_callback])

    # Load the best model
    model.load_weights('best_model.keras')

if new == 0:
    # Load the saved model
    model = tf.keras.models.load_model('best_model.keras')

# Evaluate the model
test_predictions = model.predict(test_tfidf)
predicted_labels = [label_encoder.classes_[pred] for pred in test_predictions.argmax(axis=1)]

# Compute accuracy
correct_predictions = sum([1 for actual, predicted in zip(actual_labels, predicted_labels) if actual == predicted])
total_predictions = len(predicted_labels)
accuracy = correct_predictions / total_predictions

print(f"Accuracy: {accuracy * 100:.2f}%\n\n")