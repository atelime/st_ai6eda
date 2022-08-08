import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor

st.write("""
# House Price Prediction App
This app predicts the ** House Price**!
""")
st.write('---')

# Loads the House Price Dataset
house = datasets.fetch_california_housing()
X = pd.DataFrame(house.data, columns=house.feature_names)
Y = pd.DataFrame(house.target, columns=[house.target_names])

# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')

def user_input_features():
    data = {}
    for feature in house.feature_names:
        data[feature] = st.sidebar.slider(feature, X[feature].min(), X[feature].max(), X[feature].mean())
        features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()
# Main Panel

# Print specified input parameters
st.header('Specified Input parameters')
st.write(df)
st.write('---')

# Build Regression Model
model = RandomForestRegressor()
model.fit(X, Y)
# Apply Model to Make Prediction
prediction = model.predict(df)

st.header(f'Prediction of {house.target_names}')
st.write(prediction)
st.write('---')
