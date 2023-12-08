import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# load Dataset nya
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
menu = st.sidebar.selectbox('Menu', ['Home', 'Dataset Overview', 'Correlation Heatmap', 'Histograms', 'Scatter Plots', 'Box Plots'])

# Content
st.title("Lettuce Growth Analysis")

if menu == 'Home':
    st.image('selada.jpg', caption='Lettuce Growth', use_column_width=True)

elif menu == 'Dataset Overview':
    st.subheader("Dataset Overview")
    st.write("Top 10 Rows of the Dataset:")
    st.write(filtered_dataframe.head(10))

    st.write("Bottom 10 Rows of the Dataset:")
    st.write(filtered_dataframe.tail(10))

    st.write("Dataset Shape:")
    st.write(filtered_dataframe.shape)

    st.write("Dataset Info:")
    st.write(filtered_dataframe.info())

    st.write("Summary Statistics:")
    st.write(filtered_dataframe.describe())

    st.write("Missing Values Count:")
    st.write(filtered_dataframe.isna().sum())

    st.write("Duplicate Rows Count:")
    st.write(filtered_dataframe.duplicated().sum())

elif menu == 'Correlation Heatmap':
    st.subheader("Correlation Heatmap")
    plt.figure(figsize=(8, 8))
    sns.heatmap(filtered_dataframe.corr(), annot=True, cmap="Reds", fmt=".2f")
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

st.set_option('deprecation.showPyplotGlobalUse', False)


# ... (Remaining code for other menu options)

# You can continue modifying the code for other menu options as needed.
