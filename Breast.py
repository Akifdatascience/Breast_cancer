#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import pickle

# Load the trained model from the pickle file
with open("breast.pkl", 'rb') as file:
    model = pickle.load(file)

# Function to predict breast cancer
def predict_cancer(mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness):
    input_data = pd.DataFrame({
        'mean_radius': [mean_radius],
        'mean_texture': [mean_texture],
        'mean_perimeter': [mean_perimeter],
        'mean_area': [mean_area],
        'mean_smoothness': [mean_smoothness]
    })

    prediction = model.predict(input_data)[0]
    return prediction

# Streamlit app
def main():
    st.title('Breast Cancer Prediction')

    st.write('Enter the following features:')
    mean_radius = st.number_input('mean_radius', min_value=0.0, max_value=200.0, value=15.0)
    mean_texture = st.number_input('mean_texture', min_value=0.0, max_value=200.0, value=20.0)
    mean_perimeter = st.number_input('mean_perimeter', min_value=0.0, max_value=200.0, value=100.0)
    mean_area = st.number_input('mean_area', min_value=0.0, max_value=5000.0, value=500.0)
    mean_smoothness = st.number_input('mean_smoothness', min_value=0.0, max_value=1.0, value=0.1)

    if st.button('Predict'):
        prediction = predict_cancer(mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness)
        if prediction == 0:
            st.write('The person is predicted to have **benign** (non-cancerous) breast cancer.')
        else:
            st.write('The person is predicted to have **malignant** (cancerous) breast cancer.')

if __name__ == '__main__':
    main()


# In[ ]:




