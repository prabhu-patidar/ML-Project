import streamlit as st
import numpy as np
import librosa
import joblib

# Load model
model = joblib.load("audio_model.pkl")

st.title("Voice Gender Recognition")

audio_file = st.file_uploader("Upload Voice (.wav file)")

if audio_file is not None:

    if not audio_file.name.endswith(".wav"):
        st.error("Please upload a valid .wav file")

    else:
        with open("temp.wav", "wb") as f:
            f.write(audio_file.read())

        st.audio("temp.wav")

        y, sr = librosa.load("temp.wav", duration=3)
        y = y / np.max(np.abs(y))

        if np.max(np.abs(y)) < 0.01:
            st.error("Invalid or silent audio")
        else:
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc = np.mean(mfcc.T, axis=0).reshape(1, -1)

            prediction = model.predict(mfcc)

            st.success(f"Predicted Gender: {prediction[0]}")