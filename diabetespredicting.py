# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 17:59:04 2024

@author: DELL
"""

import numpy as np
import pickle
import streamlit as st

# Load the model
loaded_model = pickle.load(open('C:\\Users\\DELL\\Downloads\\trained_model.sav', 'rb'))

# Define the prediction function
def prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    p = loaded_model.predict(input_data_reshaped)

    if p[0] == 0:
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'

# Define the main function for the Streamlit app
def main():
    st.title('Diabetes Prediction App')

    # Get user inputs
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure')
    SkinThickness = st.text_input('Skin Thickness')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('Body Mass Index value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age of the person')

    diagnosis = ''

    # When the button is pressed, perform the prediction
    if st.button('TEST RESULT'):
        diagnosis = prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        st.success(diagnosis)

if __name__ == '__main__':
    main()
