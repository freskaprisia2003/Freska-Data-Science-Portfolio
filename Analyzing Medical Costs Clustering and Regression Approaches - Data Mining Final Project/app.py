
import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Load the trained model
model = joblib.load("log_regression_model.pkl")  # Update to correct model filename

# Function to preprocess input data
def preprocess_input_data(features):
    # Create a DataFrame from the input features for preprocessing
    df = pd.DataFrame([features], columns=['age', 'bmi', 'children', 'sex', 'smoker', 'region'])
    
    # Preprocess data using the same transformations as during training
    preprocessor = model.named_steps['preprocessor']
    return preprocessor.transform(df)

# Streamlit Interface
st.title('Insurance Charges Prediction')

# Input Fields
age = st.number_input('Age', min_value=18, max_value=100, value=30)
bmi = st.number_input('BMI', min_value=10.0, max_value=50.0, value=25.0)
children = st.number_input('Children', min_value=0, max_value=10, value=2)

sex = st.selectbox('Sex', ['male', 'female'])
smoker = st.selectbox('Smoker', ['yes', 'no'])
region = st.selectbox('Region', ['northwest', 'southeast', 'southwest', 'northeast'])

# Combine inputs into a list (ensure order matches the model's expected input)
features = [age, bmi, children, sex, smoker, region]

# When the user clicks the "Predict" button
if st.button('Predict'):
    # Preprocess the input features
    preprocessed_data = preprocess_input_data(features)
    
    # Make a prediction
    prediction = model.predict(preprocessed_data)
    
    # Display the prediction
    st.write(f'Predicted Insurance Charges: ${prediction[0]:,.2f}')
