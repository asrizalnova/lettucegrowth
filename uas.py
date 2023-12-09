import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# proses dataset
dataframe = pd.read_csv("lettuce_dataset.csv", encoding='latin-1')
dataframe['Date'] = pd.to_datetime(dataframe['Date'])

# Sidebar
st.sidebar.title("Lettuce Growth Prediction")

# Date Range Input
date_range = st.sidebar.date_input("Pilih range tanggal(3 Agustus - 19 September)", [dataframe['Date'].min(), dataframe['Date'].max()])


start_date = np.datetime64(date_range[0])
end_date = np.datetime64(date_range[1])

# Filter dataframe range tanggal
filtered_dataframe = dataframe[(dataframe['Date'] >= start_date) & (dataframe['Date'] <= end_date)]

# Sidebar menu
menu = st.sidebar.selectbox('Menu', ['Home', 'Dataset Overview', 'Correlation Heatmap', 'Histograms', 'Scatter Plots', 'Box Plots', 'Prediction'])

# Content
st.title("Lettuce Growth Analysis")

if menu == 'Home':
    st.image('selada.jpg', caption='Lettuce Growth', use_column_width=True)

elif menu == 'Dataset Overview':
    st.subheader("Dataset Overview")
    st.write("Top 10 Rows of the Dataset:")
    st.write(dataframe.head(10))

    st.write("Bottom 10 Rows of the Dataset:")
    st.write(dataframe.tail(10))

    st.write("Dataset Shape:")
    st.write(dataframe.shape)

    st.write("Dataset Info:")
    st.write(dataframe.info())

    st.write("Summary Statistics:")
    st.write(dataframe.describe())

    st.write("Missing Values Count:")
    st.write(dataframe.isna().sum())

    st.write("Duplicate Rows Count:")
    st.write(dataframe.duplicated().sum())

    st.write("Dataframe yang Difilter:")
    st.write(filtered_dataframe)

elif menu == 'Correlation Heatmap':
    st.subheader("Correlation Heatmap")
    plt.figure(figsize=(8, 8))
    sns.heatmap(dataframe.corr(), annot=True, cmap="Reds", fmt=".2f")
    st.pyplot()

elif menu == 'Histograms':
    st.subheader("Histograms")
    st.write("Temperature Histogram")
    sns.histplot(data=dataframe, x="Temperature (°C)", kde=True, color="blue")
    st.pyplot()

    st.write("Humidity Histogram")
    sns.histplot(data=dataframe, x="Humidity (%)", kde=True, color="red")
    st.pyplot()

    st.write("TDS Value Histogram")
    sns.histplot(data=dataframe, x="TDS Value (ppm)", kde=True, color="green")
    st.pyplot()

    st.write("Growth Days Histogram")
    sns.histplot(data=dataframe, x="Growth Days", kde=True, color="brown")
    st.pyplot()

    st.write("pH Level Histogram")
    sns.histplot(data=dataframe, x="pH Level", kde=True, color="purple")
    st.pyplot()

    st.write("Monthly Growth Days Count")
    sns.countplot(data=dataframe, x=dataframe["Date"].dt.month.map({8: "August", 9: "September"}), color="brown")
    st.pyplot()

elif menu == 'Scatter Plots':
    st.subheader("Scatter Plots")
    st.write("Growth Days Vs Temperature")
    fig = px.scatter(dataframe, x="Temperature (°C)", y="Growth Days", marginal_x="histogram", marginal_y="histogram", title="Growth Days Vs Temperature", color_discrete_sequence=["red"])
    st.plotly_chart(fig)

    st.write("Growth Days Vs TDS Value")
    fig = px.scatter(dataframe, x="TDS Value (ppm)", y="Growth Days", marginal_x="histogram", marginal_y="histogram", title="Growth Days Vs TDS Value (ppm)", color_discrete_sequence=["green"])
    st.plotly_chart(fig)

    st.write("Growth Days Vs Date")
    fig = px.scatter(dataframe, x="Date", y="Growth Days", marginal_x="histogram", marginal_y="histogram", title="Growth Days Vs Date", color_discrete_sequence=["purple"])
    st.plotly_chart(fig)

elif menu == 'Box Plots':
    st.subheader("Box Plots")
    st.write("Growth Days Vs pH Level")
    fig = px.box(dataframe, x="pH Level", y="Growth Days", title="Growth Days Vs pH Level", color_discrete_sequence=["blue"])
    st.plotly_chart(fig)

    st.write("Growth Days Vs Humidity")
    fig = px.box(dataframe, x="Humidity (%)", y="Growth Days", title="Growth Days Vs Humidity (%)", color_discrete_sequence=["black"])
    st.plotly_chart(fig)

elif menu == 'Prediction':
    st.subheader("Lettuce Growth Prediction")

    # memilih target features dan target
    features = ['Temperature (°C)', 'Humidity (%)', 'TDS Value (ppm)', 'pH Level']
    target = 'Growth Days'

    # memisah data ke variable features dan target
    X = filtered_dataframe[features]
    y = filtered_dataframe[target]

    # memisah data ke training dan testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)  # You can adjust n_estimators as needed
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # menampilkan predictions dan actual values
    st.write("Predictions vs Actual Values:")
    result_df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
    st.write(result_df)

    # menampilkan scatter plot of actual vs predicted values
    st.subheader("Scatter Plot of Actual vs Predicted Values")
    fig = px.scatter(result_df, x='Actual', y='Predicted', labels={'Actual': 'Actual Values', 'Predicted': 'Predicted Values'})
    st.plotly_chart(fig)

st.set_option('deprecation.showPyplotGlobalUse', False)
