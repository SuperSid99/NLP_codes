import os
import numpy as np
import tensorflow as tf
from keras import layers, models
from python_speech_features import mfcc # for feature extraction
import librosa

# Load data
data_dir = '/Users/siddharthsharma/Desktop/NLP_implementations/data_voip_en/train'
labels = os.listdir(data_dir)
X = []
y = []
for label in labels:
    dir_path = os.path.join(data_dir, label)
    for filename in os.listdir(dir_path):
        audio_path = os.path.join(dir_path, filename)
        audio, sr = librosa.load(audio_path)
        features = mfcc(audio, sr)
        X.append(features)
        y.append(label)

# Convert data to NumPy arrays and normalize
X = np.array(X)
y = np.array(y)
X = (X - np.mean(X)) / np.std(X)

# Define the model architecture
input_shape = (X.shape[1], X.shape[2])
model = models.Sequential([
    layers.LSTM(128, input_shape=input_shape),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(len(labels), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=10, validation_split=0.2)

# Save the model
model.save('speech_recognition_model.h5')
