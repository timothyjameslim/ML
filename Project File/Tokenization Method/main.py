import subprocess
import numpy as np
import tensorflow as tf
from datetime import datetime
import keras
import pickle

# Ask the user if program should use a existing model or create a new model
pp = int(input("Enter 1 to pre-process .csv or 0 to pre-process .txt: "))

if pp == 0:
    # Run the preprocessing script
    subprocess.run(['python', 'preprocess.py'])

    # Load preprocessed data
    train_padded = np.load('Prepro/train_padded.npy')
    test_padded = np.load('Prepro/test_padded.npy')
    train_labels_encoded = np.load('Prepro/train_labels_encoded.npy')
    actual_labels = np.load('Prepro/actual_labels.npy', allow_pickle=True)

    with open('Prepro/tokenizer.pkl', 'rb') as f:
        tokzer = pickle.load(f)
    with open('Prepro/label_tokenizer.pkl', 'rb') as f:
        label_tokenizer = pickle.load(f)

if pp == 1:

    # Load preprocessed data
    train_padded = np.load('Prepro/train_padded_csv.npy')
    train_labels_encoded = np.load('Prepro/train_labels_encoded_csv.npy')

    with open('Prepro/tokenizer_csv.pkl', 'rb') as f:
        tokzer = pickle.load(f)
    with open('Prepro/label_tokenizer_csv.pkl', 'rb') as f:
        label_tokenizer = pickle.load(f)

# Ask the user if program should use a existing model or create a new model
new = int(input("Enter 1 to train a new model or 0 to use an existing model: "))

if new == 1:
    # Build the model
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=10000, output_dim=128, input_length=200),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dense(len(label_tokenizer.word_index)+1, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    # Create a TensorBoard callback
    log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True)

    # Define the ModelCheckpoint callback
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath='best_model.keras',
        monitor='val_loss',  # you can also monitor 'val_loss' for the lowest loss
        save_best_only=True,
        mode='min',  # for 'val_loss', use 'min'
        verbose=1
    )

    # Define the EarlyStopping callback
    early_stopping_callback = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,  # Number of epochs with no improvement after which training will be stopped
        verbose=1,
        mode='min',
        restore_best_weights=True  #Restore model weights from the epoch with the best value of the monitored quantity
    )

    # Train the model with TensorBoard callback
    history = model.fit(train_padded, train_labels_encoded, epochs=20, validation_split=0.2, batch_size=64,
                        callbacks=[tensorboard_callback, checkpoint_callback, early_stopping_callback])

    # Load the best model
    model.load_weights('best_model.keras')
if new == 0:
    # Load the saved model
    # model = tf.keras.models.load_model('Neural_Models/5091_accuracy.keras')
    model = tf.keras.models.load_model('best_model.keras')

# Evaluate the model
test_predictions = model.predict(test_padded)
predicted_labels = [label_tokenizer.index_word[pred] for pred in test_predictions.argmax(axis=1)]

# Compute accuracy
correct_predictions = sum([1 for actual, predicted in zip(actual_labels, predicted_labels) if actual == predicted])
total_predictions = len(predicted_labels)
accuracy = correct_predictions / total_predictions

print(f"Accuracy: {accuracy * 100:.2f}%\n\n")


# # Print misclassified genres
# print("Misclassified genres:")
# for i in range(len(test_data_cleaned)):
#     if actual_labels[i] != predicted_labels[i]:
#         print(f"Title: {test_data_cleaned.iloc[i]['Title']}")
#         print(f"Description: {test_data_cleaned.iloc[i]['Description']}")
#         print(f"Actual Genre: {actual_labels[i]}")
#         print(f"Predicted Genre: {predicted_labels[i]}")
#         print("-------------------------------------------------------")