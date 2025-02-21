# -*- coding: utf-8 -*-
"""
Created on Fri Feb  21 14:06:19 2025

@author: Tobiloba
"""

# streamlit_app.py

import streamlit as st
import pandas as pd
import joblib


# Title for the web app
st.title('Bank Account Prediction in East Africa')

# Load the trained model
model = joblib.load("model.pkl")  

# Input features from the user
st.header('Input Features')

location_type = st.selectbox('Location Type', ['Rural', 'Urban'])
cellphone_access = st.selectbox('Cellphone Access', ['Yes', 'No'])
household_size = st.number_input('Household Size', min_value=1, max_value=10, value=3)
age_of_respondent = st.number_input('Age of Respondent', min_value=18, max_value=100, value=30)
is_married = st.selectbox('Marital Status', ['Married', 'Single'])
has_income = st.selectbox('Has Income?', ['Yes', 'No'])

# Job type selection
job_types = [
    'Farming and Fishing',
    'Formally employed Government',
    'Formally employed Private',
    'Government Dependent',
    'Informally employed',
    'No Income',
    'Other Income',
    'Self employed'
]

# Select job type
selected_job_type = st.selectbox('Job Type', job_types)

# Education level features (since it's already binary-encoded)
primary_education = st.checkbox('Primary Education')
no_education = st.checkbox('No Formal Education')
secondary_education = st.checkbox('Secondary Education')
tertiary_education = st.checkbox('Tertiary Education')
other_education = st.checkbox('Other Education')

# Create a dictionary for the model input
input_data = {
    'location_type': [1 if location_type == 'Urban' else 0],
    'cellphone_access': [1 if cellphone_access == 'Yes' else 0],
    'household_size': [household_size],
    'age_of_respondent': [age_of_respondent],
    'job_type_Farming and Fishing': [1 if selected_job_type == 'Farming and Fishing' else 0],
    'job_type_Formally employed Government': [1 if selected_job_type == 'Formally employed Government' else 0],
    'job_type_Formally employed Private': [1 if selected_job_type == 'Formally employed Private' else 0],
    'job_type_Government Dependent': [1 if selected_job_type == 'Government Dependent' else 0],
    'job_type_Informally employed': [1 if selected_job_type == 'Informally employed' else 0],
    'job_type_No Income': [1 if selected_job_type == 'No Income' else 0],
    'job_type_Other Income': [1 if selected_job_type == 'Other Income' else 0],
    'job_type_Self employed': [1 if selected_job_type == 'Self employed' else 0],
    'has_income': [1 if has_income == 'Yes' else 0],
    'is_married': [1 if is_married == 'Married' else 0],
    'is_single': [1 if is_married == 'Single' else 0],  # Adjusted for logical consistency
    'primary_education': [1 if primary_education else 0],
    'no_education': [1 if no_education else 0],
    'secondary_education': [1 if secondary_education else 0],
    'tertiary_education': [1 if tertiary_education else 0],
    'other_education': [1 if other_education else 0]
}

# Convert input data to dataframe
input_df = pd.DataFrame(input_data)

# Display the input data
st.subheader('Input Data')
st.write(input_df)

# Prediction
if st.button('Predict'):
    prediction = model.predict(input_df)
    prediction_prob = model.predict_proba(input_df)
    
    if prediction[0] == 1:
        st.success(f"The model predicts that this individual has a bank account with {prediction_prob[0][1] * 100:.2f}% probability.")
    else:
        st.error(f"The model predicts that this individual does NOT have a bank account with {prediction_prob[0][0] * 100:.2f}% probability.")
