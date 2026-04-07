import streamlit as st
import numpy as np
import cv2
import joblib

# Load model
model = joblib.load("video_model.pkl")

st.title("Workout Detection")

option = st.radio("Choose Input Method", ["Upload Video", "Use Camera"])

# Upload video
if option == "Upload Video":

    video_file = st.file_uploader("Upload Workout Video")

    if video_file is not None:

        with open("temp.mp4", "wb") as f:
            f.write(video_file.read())

        st.video("temp.mp4")

        cap = cv2.VideoCapture("temp.mp4")

        frames = []

        for i in range(3):
            success, frame = cap.read()
            if success:
                frame = cv2.resize(frame, (64,64))
                frames.append(frame.flatten())

        cap.release()

        if len(frames) == 0:
            st.error("Could not process video")
        else:
            avg_frame = np.mean(frames, axis=0).reshape(1, -1)

            proba = model.predict_proba(avg_frame)
            confidence = np.max(proba)
            prediction = model.classes_[np.argmax(proba)]

            if confidence < 0.6:
                st.error("Invalid / Unknown Exercise")
            else:
                st.success(f"{prediction} ({confidence*100:.2f}%)")

# Camera input
else:

    img_file = st.camera_input("Take a picture")

    if img_file is not None:

        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)

        st.image(frame, channels="BGR")

        frame = cv2.resize(frame, (64,64))
        frame_flat = frame.flatten().reshape(1, -1)

        proba = model.predict_proba(frame_flat)
        confidence = np.max(proba)
        prediction = model.classes_[np.argmax(proba)]

        if confidence < 0.6:
            st.error("Invalid / Unknown Exercise")
        else:
            st.success(f"{prediction} ({confidence*100:.2f}%)")