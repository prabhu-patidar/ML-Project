import streamlit as st
import numpy as np
import cv2
import joblib

# Load model
model = joblib.load("flower_model.pkl")

st.title("Flower Image Classification")

file = st.file_uploader("Upload Flower Image")

if file is not None:

    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    if img is None:
        st.error("Invalid image file")
    else:
        img_resized = cv2.resize(img, (64, 64))

        st.image(img, caption="Uploaded Image", channels="BGR")

        img_flat = img_resized.flatten().reshape(1, -1)

        proba = model.predict_proba(img_flat)
        confidence = np.max(proba)
        prediction = model.classes_[np.argmax(proba)]

        if confidence < 0.6:
            st.error("Invalid Image / Not a Flower")
        else:
            st.success(f"Predicted Flower: {prediction} ({confidence*100:.2f}%)")