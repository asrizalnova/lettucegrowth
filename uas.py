import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st

# load Dataset nya
dataframe = pd.read_csv("lettuce_dataset.csv", encoding='latin-1')
dataframe['Date'] = pd.to_datetime(dataframe['Date'])

# Sidebar
st.sidebar.title("Lettuce Growth Prediction")

# Sidebar menu

menu = st.sidebar.selectbox('Menu', ['Home', 'Dataset Overview', 'Correlation Heatmap', 'Histograms', 'Scatter Plots', 'Box Plots'])

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

st.set_option('deprecation.showPyplotGlobalUse', False)
