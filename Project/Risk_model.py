import streamlit as st
import pandas as pd
import joblib

model = joblib.load('Risk_model1.pkl')

st.title("Healthcare Risk Stratification App")

age = st.number_input("Age",min_value=0)
length_of_stay = st.number_input("Length of Stay (days)",min_value=0)
treatment_cost = st.number_input("treatment Cost",min_value=0)


if st.button("Predict"):
    Input_data = pd.DataFrame([[age,length_of_stay,treatment_cost]],columns=['Age','LengthOfstay', 'TreatmentCost'])
    prediction = model.predict(Input_data)[0]
    probablity = model.predict_proba(Input_data)[0][1]

    st.write(f"Risk Prediction: {'High risk' if prediction == 1 else 'Low risk'}")
    st.write(f"Risk Probablity: {round(probablity,2)}")