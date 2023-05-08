import pickle
import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder
import streamlit as st

# Load the saved XGBoost model from the pickle file
with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

# Define a function that takes user inputs and returns a DataFrame with a single row of data
def get_user_inputs():
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    age = st.slider("Age", min_value=0, max_value=120, value=25)
    hypertension = st.selectbox("Has Hypertension?", ["No", "Yes"])
    heart_disease = st.selectbox("Has Heart Disease?", ["No", "Yes"])
    ever_married = st.selectbox("Ever Married?", ["No", "Yes"])
    work_type = st.selectbox("Work Type", ["Govt_job", "Never_worked", "Private", "Self-employed"])
    residence_type = st.selectbox("Residence Type", ["Rural", "Urban"])
    avg_glucose_level = st.slider("Average Glucose Level (in mg/dL)", min_value=0.0, max_value=300.0, value=100.0)
    bmi = st.slider("BMI", min_value=0.0, max_value=100.0, value=20.0)
    smoking_status = st.selectbox("Smoking Status", ["Unknown", "never smoked", "formerly smoked", "smokes"])

    # Convert the user inputs into a DataFrame with a single row of data
    data = {
        "id": [str(random.randint(1000, 99999))],
        "gender": [gender],
        "age": [age],
        "hypertension": [1 if hypertension == "Yes" else 0],
        "heart_disease": [1 if heart_disease == "Yes" else 0],
        "ever_married": [1 if ever_married == "Yes" else 0],
        "work_type": [work_type],
        "Residence_type": [residence_type],
        "avg_glucose_level": [avg_glucose_level],
        "bmi": [bmi],
        "smoking_status": [smoking_status]
    }
    user_inputs = pd.DataFrame(data)
    return user_inputs

def label_encode_columns(dataset, columns):
    encoded_dataset = dataset.copy()  # make a copy of the dataset to avoid modifying the original
    for column in columns:
        encoder = LabelEncoder()
        encoded_dataset[column] = encoder.fit_transform(encoded_dataset[column])
    return encoded_dataset

st.title("Stroke Risk Prediction App")
st.write("Please select the values for each input variable below:")

user_inputs = get_user_inputs()
categorical_cols = [col for col in user_inputs.columns if user_inputs[col].dtype == 'object']
user_inputs = label_encode_columns(user_inputs, categorical_cols)

if st.button('Predict Stroke Risk'):
    prediction_proba = model.predict_proba(user_inputs)[0]
    if prediction_proba[1] > 0.5:
        st.write("The model predicts that the user is at risk of stroke with a probability of {:.2f}%".format(prediction_proba[1]*100))
    else:
        st.write("The model predicts that the user is not at risk of stroke with a probability of {:.2f}%".format(prediction_proba[0]*100))
