# -*- coding: utf-8 -*-
"""
Created on 21st Feb  12:57 2025

@author: Tobiloba
"""

# Import necessary libraries
import streamlit as st
import pandas as pd
import joblib

# Load the pre-trained model
model = joblib.load(r"C:\Users\HP\OneDrive\Desktop\DATA SCIENCE BOOTCAMP\MACHINE LEARNING\expressoModel.pkl")  

# Set up the Streamlit app
st.title('Expresso Client Churn Prediction')
st.write("This app predicts the churn probability for Expresso clients based on their behavior.")


# Input fields for user to enter feature values
frequence_rech = st.number_input('Recharge Frequency (FREQUENCE_RECH)', min_value=1.0, max_value=114.0, value=11.44, step=1.0)
revenue = st.number_input('Revenue (REVENUE)', min_value=1.0, max_value=165166.0, value= 5454.27, step=0.1)
frequence = st.number_input('Frequency of usage (FREQUENCE)', min_value=1.0, max_value=91.0,value = 13.88, step=1.0)
data_volume = st.number_input('Data Volume (DATA_VOLUME)', min_value=0.0, max_value=560933.0,value = 3165.06, step=0.1)
on_net = st.number_input('On Net Usage (ON_NET)', min_value=0.0, max_value=20837.0,value = 272.18, step=1.0)
orange = st.number_input('Orange Network Usage (ORANGE)', min_value=0.0, max_value=4743.0, value = 96.23, step=1.0)
regularity = st.number_input('Regularity of usage (REGULARITY)', min_value=0.0, max_value=1346.0, value=7.93, step=1.0)
freq_top_pack = st.number_input('Frequency of Top Pack (FREQ_TOP_PACK)', min_value=1.0, max_value=320.0, value=9.20, step=1.0)

# Create a dictionary with the input data
input_data = {
    'FREQUENCE_RECH': frequence_rech,
    'REVENUE': revenue,
    'FREQUENCE': frequence,
    'DATA_VOLUME': data_volume,
    'ON_NET': on_net,
    'ORANGE': orange,
    'REGULARITY': regularity,
    'FREQ_TOP_PACK': freq_top_pack
}

# Convert the dictionary to a DataFrame
input_df = pd.DataFrame([input_data])

# Predict churn probability using the loaded model
if st.button('Predict Churn Probability'):
    prediction = model.predict_proba(input_df)[:, 1]  # Probability of churn
    churn_probability = round(prediction[0] * 100, 2)
    st.write(f"The predicted churn probability is {churn_probability}%")

# Option to display input data
if st.checkbox('Show Input Data'):
    st.write(input_df)
