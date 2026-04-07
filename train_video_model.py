import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

data = []
labels = []

path = "video"

for label in os.listdir(path):
    folder = os.path.join(path, label)

    if not os.path.isdir(folder):
        continue

    for file in os.listdir(folder):
        video_path = os.path.join(folder, file)

        cap = cv2.VideoCapture(video_path)

        frames = []

        # take first 3 frames
        for i in range(3):
            success, frame = cap.read()
            if success:
                frame = cv2.resize(frame, (64,64))
                frames.append(frame.flatten())

        if len(frames) > 0:
            avg_frame = np.mean(frames, axis=0)
            data.append(avg_frame)
            labels.append(label)

        cap.release()

X = np.array(data)
y = np.array(labels)

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, "video_model.pkl")

print("Video model trained successfully")