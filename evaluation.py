import os
import numpy as np
from feature_extraction import extract_features
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model

model = load_model("deepfake_voice_detector.h5")

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

X = X.reshape(X.shape[0], X.shape[1], 1)

pred = model.predict(X)

pred = (pred > 0.5).astype(int)

print(confusion_matrix(y,pred))

print(classification_report(y,pred))