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


# Sidebar menu
st.sidebar.title("Lettuce Growth Prediction")

# Sidebar menu
menu = st.sidebar.selectbox('Menu', ['Home', 'Exploratory Data Analysis', 'Prediction'])

# Content
st.title("Lettuce Growth")

if menu == 'Home':
    st.image('selada.jpg', caption='Lettuce Growth', use_column_width=True)

elif menu == 'Exploratory Data Analysis':
    # Submenu selection
    eda_submenu = st.selectbox('Pilih Visualisasi yang ingin di lihat:', ['Dataset Overview', 'Correlation Heatmap', 'Histograms', 'Scatter Plots', 'Box Plots'])

    if eda_submenu == 'Dataset Overview':
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

    elif eda_submenu == 'Correlation Heatmap':
        st.subheader("Correlation Heatmap")
        plt.figure(figsize=(8, 8))
        sns.heatmap(dataframe.corr(), annot=True, cmap="Reds", fmt=".2f")
        st.pyplot()

    elif eda_submenu == 'Histograms':
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

    elif eda_submenu == 'Scatter Plots':
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

    elif eda_submenu == 'Box Plots':
        st.subheader("Box Plots")
        st.write("Growth Days Vs pH Level")
        fig = px.box(dataframe, x="pH Level", y="Growth Days", title="Growth Days Vs pH Level", color_discrete_sequence=["blue"])
        st.plotly_chart(fig)

        st.write("Growth Days Vs Humidity")
        fig = px.box(dataframe, x="Humidity (%)", y="Growth Days", title="Growth Days Vs Humidity (%)", color_discrete_sequence=["green"])
        st.plotly_chart(fig)

elif menu == 'Prediction':
    st.subheader("Lettuce Growth Prediction")

    # Date Range Input
    date_range = st.date_input("Select Date Range (Data is available from August 3 - September 19)", [dataframe['Date'].min(), dataframe['Date'].max()])

    # Select Features
    features = st.multiselect("Select Features:", ['Temperature (°C)', 'Humidity (%)', 'TDS Value (ppm)', 'pH Level'], default=['Temperature (°C)'])

    # Create sliders for each selected feature
    sliders = {}
    for feature in features:
        value = st.slider(f"Estimator {feature} range", min_value=1, max_value=50, value=[1,3], step=1)
        sliders[feature] = (value)

    if len(date_range) == 2:
        start_date = np.datetime64(date_range[0])
        end_date = np.datetime64(date_range[1])

        # Filter dataframe by date range
        filtered_dataframe = dataframe[(dataframe['Date'] >= start_date) & (dataframe['Date'] <= end_date)]

        st.write("Filtered DataFrame:")
        st.write(filtered_dataframe)

        # Separate data into features and target
        X = filtered_dataframe[features]
        y = filtered_dataframe['Growth Days']

        # Check for missing values
        st.write("Missing Values in X:")
        st.write(X.isnull().sum())

        st.write("Missing Values in y:")
        st.write(y.isnull().sum())

        # Check data types
        st.write("Data Types in X:")
        st.write(X.dtypes)

        # Check target variable
        st.write("Data Types in y:")
        st.write(y.dtypes)

        # Separate data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize and train Random Forest model
        total_estimators = sum(max_value for _, max_value in sliders.values())
        model = RandomForestRegressor(n_estimators=total_estimators, random_state=42)
        model.fit(X_train, y_train)


        # Make predictions
        predictions = model.predict(X_test)

        # Display predictions and actual values
        st.write("Predictions vs Actual Values:")
        result_df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
        st.write(result_df)

        # Display scatter plot of actual vs predicted values
        st.subheader("Scatter Plot of Actual vs Predicted Values")
        fig = px.scatter(result_df, x='Actual', y='Predicted', labels={'Actual': 'Actual Values', 'Predicted': 'Predicted Values'})
        st.plotly_chart(fig)

        # Display feature importance
        st.subheader("Feature Importance")
        feature_importance = model.feature_importances_
        feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance})
        st.write(feature_importance_df)

        # Create bar chart for feature importance
        st.subheader("Bar Chart of Feature Importance")
        fig_bar = px.bar(feature_importance_df, x='Feature', y='Importance', title='Feature Importance')
        st.plotly_chart(fig_bar)


    else:
        st.warning("Silahkan pilih 2 tanggal/ jika ingin 1 tanggal klik tanggal tsb 2x.")


st.set_option('deprecation.showPyplotGlobalUse', False)
