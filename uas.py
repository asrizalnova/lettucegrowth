import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load Dataset
dataframe = pd.read_csv("lettuce_dataset.csv", encoding='latin-1')
dataframe['Date'] = pd.to_datetime(dataframe['Date'])

# Sidebar
st.sidebar.title("Lettuce Growth Prediction")

# Date Range Input
date_range = st.sidebar.date_input("Select Date Range", [dataframe['Date'].min(), dataframe['Date'].max()])

# Convert date_range values to numpy.datetime64
start_date = np.datetime64(date_range[0])
end_date = np.datetime64(date_range[1])

# Filter dataframe based on date range
filtered_dataframe = dataframe[(dataframe['Date'] >= start_date) & (dataframe['Date'] <= end_date)]

# Sidebar menu
menu = st.sidebar.selectbox('Menu', ['Home', 'Dataset Overview', 'Correlation Heatmap', 'Histograms', 'Scatter Plots', 'Box Plots', 'Prediction'])

# Content
st.title("Lettuce Growth Analysis")

if menu == 'Home':
    st.image('selada.jpg', caption='Lettuce Growth', use_column_width=True)

# ... (Other menu options)

elif menu == 'Prediction':
    st.subheader("Lettuce Growth Prediction")

    # Select features and target variable
    features = ['Temperature (°C)', 'Humidity (%)', 'TDS Value (ppm)', 'pH Level']
    target = 'Growth Days'

    # Split the data into features and target variable
    X = filtered_dataframe[features]
    y = filtered_dataframe[target]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the testing set
    predictions = model.predict(X_test)

    # Display the predictions and actual values
    st.write("Predictions vs Actual Values:")
    result_df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
    st.write(result_df)

    # Display a scatter plot of actual vs predicted values
    st.subheader("Scatter Plot of Actual vs Predicted Values")
    fig = px.scatter(result_df, x='Actual', y='Predicted', labels={'Actual': 'Actual Values', 'Predicted': 'Predicted Values'})
    st.plotly_chart(fig)
