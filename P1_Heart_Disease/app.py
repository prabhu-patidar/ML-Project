import streamlit as st
import joblib
import pandas as pd

# Load model and data
model = joblib.load("heart_model.pkl")
data = pd.read_csv("heart.csv")
accuracy = joblib.load("model_accuracy.pkl")

st.title("Heart Disease Prediction System")
st.write("Machine Learning Based Prediction")

st.success(f"Model Accuracy: {round(accuracy*100,2)}%")

# Demo buttons
st.subheader("Demo Examples")

col1, col2 = st.columns(2)

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

# Inputs
age = st.number_input("Age", 1, 120, key="age")
sex = st.selectbox("Sex", ["Female", "Male"], key="sex")
cp = st.number_input("Chest Pain Type (0–3)", 0, 3, key="cp")
trestbps = st.number_input("Resting Blood Pressure", 80, 200, key="trestbps")
chol = st.number_input("Cholesterol", 100, 600, key="chol")

fbs = st.selectbox("Fasting Blood Sugar >120 mg/dl", [0, 1], key="fbs")
restecg = st.number_input("Rest ECG (0–2)", 0, 2, key="restecg")
thalach = st.number_input("Max Heart Rate Achieved", 60, 220, key="thalach")
exang = st.selectbox("Exercise Induced Angina", [0, 1], key="exang")

oldpeak = st.number_input("ST Depression", 0.0, 10.0, key="oldpeak")
slope = st.number_input("Slope (0–2)", 0, 2, key="slope")
ca = st.number_input("Number of Major Vessels (0–3)", 0, 3, key="ca")
thal = st.number_input("Thal (0–3)", 0, 3, key="thal")

# Convert
sex = 1 if sex == "Male" else 0

# Prediction
if st.button("Predict Heart Disease"):

    input_data = pd.DataFrame(
        [[age, sex, cp, trestbps, chol, fbs, restecg, thalach,
          exang, oldpeak, slope, ca, thal]],
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