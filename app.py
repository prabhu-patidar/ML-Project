import numpy as np
import streamlit as st
import joblib
import pandas as pd
import cv2
import librosa
import sounddevice as sd
from scipy.io.wavfile import write

model = joblib.load("heart_model.pkl")
sentiment_model = joblib.load("sentiment_model.pkl")
flower_model = joblib.load("flower_model.pkl")
audio_model = joblib.load("audio_model.pkl")
video_model = joblib.load("video_model.pkl")
data = pd.read_csv("heart.csv")
accuracy = joblib.load("model_accuracy.pkl")

vectorizer = joblib.load("vectorizer.pkl")

menu = st.sidebar.selectbox(
        "Select Module",
        ["Heart Disease Prediction", 
         "Movie Sentiment Analysis",
         "Flower Image Classification",
         "Voice Gender Recognition",
         "Workout Detection"]
    )

if menu == "Heart Disease Prediction":
    st.title("Heart Disease Prediction System")
    
    st.write("Machine Learning Based Prediction")

    st.success(f"Model Accuracy: {round(accuracy*100,2)}%")

    # Demo buttons
    st.subheader("Demo Examples")

    col1, col2 = st.columns(2)

    # High Risk example from dataset
    if col1.button("Load High Risk Example"):
        row = data[data["target"] == 1].iloc[0]

        st.session_state.age = int(row["age"])
        st.session_state.sex = "Male" if row["sex"] == 1 else "Female"
        st.session_state.cp = int(row["cp"])
        st.session_state.trestbps = int(row["trestbps"])
        st.session_state.chol = int(row["chol"])
        st.session_state.fbs = int(row["fbs"])
        st.session_state.restecg = int(row["restecg"])
        st.session_state.thalach = int(row["thalach"])
        st.session_state.exang = int(row["exang"])
        st.session_state.oldpeak = float(row["oldpeak"])
        st.session_state.slope = int(row["slope"])
        st.session_state.ca = int(row["ca"])
        st.session_state.thal = int(row["thal"])

    # Low Risk example from dataset
    if col2.button("Load Low Risk Example"):
        row = data[data["target"] == 0].iloc[0]

        st.session_state.age = int(row["age"])
        st.session_state.sex = "Male" if row["sex"] == 1 else "Female"
        st.session_state.cp = int(row["cp"])
        st.session_state.trestbps = int(row["trestbps"])
        st.session_state.chol = int(row["chol"])
        st.session_state.fbs = int(row["fbs"])
        st.session_state.restecg = int(row["restecg"])
        st.session_state.thalach = int(row["thalach"])
        st.session_state.exang = int(row["exang"])
        st.session_state.oldpeak = float(row["oldpeak"])
        st.session_state.slope = int(row["slope"])
        st.session_state.ca = int(row["ca"])
        st.session_state.thal = int(row["thal"])



    age = st.number_input("Age", 1,120, key="age")
    sex = st.selectbox("Sex", ["Female","Male"], key="sex")
    cp = st.number_input("Chest Pain Type (0–3)",0,3,key="cp")
    trestbps = st.number_input("Resting Blood Pressure",80,200,key="trestbps")
    chol = st.number_input("Cholesterol",100,600,key="chol")

    fbs = st.selectbox("Fasting Blood Sugar >120 mg/dl",[0,1],key="fbs")
    restecg = st.number_input("Rest ECG (0–2)",0,2,key="restecg")
    thalach = st.number_input("Max Heart Rate Achieved",60,220,key="thalach")
    exang = st.selectbox("Exercise Induced Angina",[0,1],key="exang")

    oldpeak = st.number_input("ST Depression",0.0,10.0,key="oldpeak")
    slope = st.number_input("Slope (0–2)",0,2,key="slope")
    ca = st.number_input("Number of Major Vessels (0–3)",0,3,key="ca")
    thal = st.number_input("Thal (0–3)",0,3,key="thal")


    if sex == "Male":
        sex = 1
    else:
        sex = 0


    if st.button("Predict Heart Disease"):

        input_data = pd.DataFrame(
            [[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]],
            columns=[
                "age","sex","cp","trestbps","chol","fbs",
                "restecg","thalach","exang","oldpeak",
                "slope","ca","thal"
            ]
        )

        prediction = model.predict(input_data)

        if prediction[0] == 1:
            st.error("⚠ High Risk of Heart Disease")
        else:
            st.success("✓ Low Risk of Heart Disease")


if menu == "Movie Sentiment Analysis":

    st.title("Movie Review Sentiment Analysis")

    review = st.text_area("Enter Movie Review")

    if st.button("Predict Sentiment"):

         # empty check
        if review.strip() == "":
            st.error("Please enter a review before prediction")

        # short text check
        elif len(review.strip()) < 5:
            st.warning("Review too short for meaningful prediction")

        # valid input
        else:
            vector = vectorizer.transform([review])

            prediction = sentiment_model.predict(vector)

            st.success(f"Sentiment: {prediction[0]}")

if menu == "Flower Image Classification":

    st.title("Flower Image Classifier")

    file = st.file_uploader("Upload Flower Image")

    if file is not None:

        # convert file to image
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)


        img = cv2.imdecode(file_bytes, 1)

        if img is None:
            st.error("Invalid image file")
        else:
        
            img_resized = cv2.resize(img, (64,64))

            st.image(img, caption="Uploaded Image", channels="BGR")

            img_flat = img_resized.flatten().reshape(1, -1)

            proba = flower_model.predict_proba(img_flat)
            confidence = np.max(proba)
            prediction = flower_model.classes_[np.argmax(proba)]

            if confidence < 0.6:
                st.error("Invalid Image / Not a Flower")

        
            else:
                st.success(f"Predicted Flower: {prediction} ({confidence*100:.2f}%)")

if menu == "Voice Gender Recognition":

    st.title("Voice Gender Recognition")

    option = st.radio("Choose Input Method", ["Upload Audio", "Record Voice"])

    # ------------------ UPLOAD OPTION ------------------
    if option == "Upload Audio":

        audio_file = st.file_uploader("Upload Voice (.wav file)")

        if audio_file is not None:
            if not audio_file.name.endswith(".wav"):
                st.error("Please upload a valid .wav file")
            else:
                audio_bytes = audio_file.read()

                st.audio(audio_bytes, format="audio/wav")

                with open("temp.wav", "wb") as f:
                    f.write(audio_bytes)

            y, sr = librosa.load("temp.wav", duration=3)

            if np.max(np.abs(y)) < 0.01:
                st.error("Invalid or silent audio")
            else:
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                mfcc = np.mean(mfcc.T, axis=0).reshape(1, -1)

                prediction = audio_model.predict(mfcc)

                st.success(f"Predicted Gender: {prediction[0]}")

    # ------------------ RECORD OPTION ------------------
    else:

        duration = st.slider("Recording Duration (seconds)", 1, 5, 3)

        if st.button("Record Voice"):

            fs = 44100

            st.info("Recording... Speak now")

            recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
            sd.wait()

            # check if recording is empty / silent
            if recording is None or len(recording) == 0:
                st.error("No voice recorded")
            else:
                write("recorded.wav", fs, recording)

                st.success("Recording Done")

                # playback
                st.audio("recorded.wav")

                y, sr = librosa.load("recorded.wav", duration=3)

                # silence check (IMPORTANT)
                if np.max(np.abs(y)) < 0.01:
                    st.error("No voice detected. Please speak louder.")
                else:
                    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                    mfcc = np.mean(mfcc.T, axis=0)
                    mfcc = mfcc.reshape(1, -1)

                    prediction = audio_model.predict(mfcc)

                    st.success(f"Predicted Gender: {prediction[0]}")


if menu == "Workout Detection":

    st.title("Workout Detection")

    option = st.radio("Choose Input Method", ["Upload Video", "Record from Camera"])

    # ---------------- UPLOAD ----------------
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

                proba = video_model.predict_proba(avg_frame)
                confidence = np.max(proba)
                prediction = video_model.classes_[np.argmax(proba)]

                if confidence < 0.6:
                    st.error("Invalid / Unknown Exercise")
                else:
                    st.success(f"{prediction} ({confidence*100:.2f}%)")

    # ---------------- CAMERA ----------------
    elif option == "Record from Camera":

        img_file = st.camera_input("Take a picture")

        if img_file is not None:

            file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, 1)

            st.image(frame, channels="BGR")

            frame = cv2.resize(frame, (64,64))
            frame_flat = frame.flatten().reshape(1, -1)

            proba = video_model.predict_proba(frame_flat)
            confidence = np.max(proba)
            prediction = video_model.classes_[np.argmax(proba)]

            if confidence < 0.6:
                st.error("Invalid / Unknown Exercise")
            else:
                st.success(f"{prediction} ({confidence*100:.2f}%)")

