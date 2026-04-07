import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("Movie Review Sentiment Analysis")

review = st.text_area("Enter Movie Review")

if st.button("Predict Sentiment"):

    # Empty check
    if review.strip() == "":
        st.error("Please enter a review before prediction")

    # Short text check
    elif len(review.strip()) < 5:
        st.warning("Review too short for meaningful prediction")

    else:
        vector = vectorizer.transform([review])
        prediction = model.predict(vector)

        st.success(f"Sentiment: {prediction[0]}")