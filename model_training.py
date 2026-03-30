import os
import numpy as np
from feature_extraction import extract_features
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models

dataset_path = "dataset"

X = []
y = []

labels = ["real", "fake"]

for label in labels:

    folder = os.path.join(dataset_path, label)

    for file in os.listdir(folder):

        path = os.path.join(folder, file)

        features = extract_features(path)

        X.append(features)

        if label == "real":
            y.append(0)
        else:
            y.append(1)

X = np.array(X)
y = np.array(y)

# reshape for CNN
X = X.reshape(X.shape[0], X.shape[1], 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

inputs = layers.Input(shape=(X.shape[1],1))

x = layers.Conv1D(64,3,activation="relu")(inputs)
x = layers.MaxPooling1D(2)(x)

x = layers.Conv1D(128,3,activation="relu")(x)
x = layers.MaxPooling1D(2)(x)

x = layers.Bidirectional(
    layers.LSTM(64, return_sequences=True)
)(x)

attention = layers.Attention()([x,x])

x = layers.GlobalAveragePooling1D()(attention)

x = layers.Dense(64,activation="relu")(x)

outputs = layers.Dense(1,activation="sigmoid")(x)

model = models.Model(inputs,outputs)

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_test,y_test)
)

model.save("deepfake_voice_detector.h5")

print("Model Saved!")