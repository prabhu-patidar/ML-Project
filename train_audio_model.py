import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

data = []
labels = []

path = "audio"

for label in os.listdir(path):
    folder = os.path.join(path, label)

    if not os.path.isdir(folder):
        continue

    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)

        try:
            y, sr = librosa.load(file_path, duration=3)

            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc = np.mean(mfcc.T, axis=0)

            data.append(mfcc)
            labels.append(label)

        except:
            continue

X = np.array(data)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)

joblib.dump(model, "audio_model.pkl")