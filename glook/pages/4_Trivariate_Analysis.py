# page2.py
import streamlit as st
import pandas as pd
from scipy.stats import zscore
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import plotly.figure_factory as ff
st.title("3️⃣ Trivariate Analysis")
try:
    st.write("Session State:->", st.session_state["shared"])
    # Access the DataFrame from session_state
    if "df" in st.session_state:
        df = st.session_state.df
        # st.dataframe(df)

    else:
        st.write("DataFrame not found.")

    print(type(df))

    dff = df
# numeric_columns = []
# categorical_columns = []
# choice = st.sidebar.radio("choice", ("none", "s", "u"), captions=["nul", "sl", "us"], index= 0)
# if choice == "u":
#     for column in dff.columns:
#         if dff[column].dtype == 'int64' or dff[column].dtype == 'float64':
#             numeric_columns.append(column)
#             # Check missing values in numeric columns
#             numeric_missing_values = dff.select_dtypes(include=['int64', 'float64']).isna().sum()
#             # Impute missing values with mean
#             mean_imputation_value = dff[column].mean()
#             dff[column].fillna(mean_imputation_value, inplace=True)
#         elif dff[column].dtype == 'object':
#             categorical_columns.append(column)
#             # Check missing values in categorical columns
#             categorical_missing_values = dff.select_dtypes(include=['object', 'category']).isna().sum()
#             # Impute missing values with mode
#             mode_imputation_value = dff[column].mode()[0]  # Mode may return multiple values, so we take the first one
#             dff[column].fillna(mode_imputation_value, inplace=True)


#     # print("Numeric Columns:", numeric_columns)
#     # print("Categorical Columns:", categorical_columns)
#     # Check if there are any remaining missing values
#     print("\nRemaining Missing Values:")
#     print(dff.isna().sum())

#     # Print the DataFrame to verify imputation
#     print("\nDataFrame after imputation:")
#     # print(dff.head())


#     # Assuming dff is your DataFrame

#     # Get the shape of the DataFrame before dropping duplicates
#     original_shape = dff.shape

#     # Drop duplicate rows
#     dff.drop_duplicates(inplace=True)

#     # Get the shape of the DataFrame after dropping duplicates
#     new_shape = dff.shape

#     # Calculate the number of rows dropped
#     rows_dropped = original_shape[0] - new_shape[0]

#     print("Rows dropped due to duplicates:", rows_dropped)


#     # Assuming dff is your DataFrame with numeric columns already processed

#     # Calculate z-scores for numeric columns
#     z_scores = dff.select_dtypes(include=['int64', 'float64']).apply(zscore)

#     # Define threshold for outlier detection
#     threshold = 3

#     # Mark outliers
#     outliers = (z_scores.abs() > threshold).any(axis=1)

#     # Print rows with outliers
#     print("Rows with outliers:")
#     print(dff[outliers])

#     # Optional: Remove outliers
#     dff = dff[~outliers]

#     # Optional: Replace outliers
#     # Replace outliers with NaN or impute using other methods


#     # Assuming dff is your DataFrame
#     dff.reset_index(drop=True, inplace=True)
#     st.dataframe(dff)

#     dff_columns = dff.columns

#     # Perform one-hot encoding
#     dff_encoded = pd.get_dummies(dff, drop_first=True)

#     # Print the DataFrame after one-hot encoding
#     # print(dff_encoded.head())

#     st.dataframe(dff_encoded)



#     # Calculate variance for numeric columns
#     variances = dff.select_dtypes(include=['int64', 'float64']).var()

#     # Define threshold for zero or near-zero variance
#     threshold = 1.00001  # You can adjust this threshold as needed

#     # Identify columns with variance below the threshold
#     near_zero_variance_features = variances[variances <= threshold]

#     print("Near zero variance features:")
#     print(near_zero_variance_features)

# Streamlit UI for selecting plot type
# plot_type = st.selectbox("Select plot type", ["Scatter Plot", "Line Plot", "Bar Plot", "Histogram", "Box Plot", "Violin Plot", "ECDF", "Strip Chart", "Density Contour", "Density Heatmap"])



# Assuming dff is the DataFrame defined in your Streamlit app

# Streamlit UI to select columns
# try:
    col1, col2, col3 = st.columns(3)

    # First column
    with col1:
        x_axis_column = st.selectbox("Select x-axis column", dff.columns)
        y_axis_column = st.selectbox("Select y-axis column", dff.columns)

    # Second column
    with col2:
        z_axis_column = st.selectbox("Select z-axis column", dff.columns)
        color_column = st.selectbox("Select color column", dff.columns)

    # Third column
    with col3:
        size_column = st.selectbox("Select size column", dff.columns)
        hover_name_column = st.selectbox("Select hover name column", dff.columns)
        symbol_column = st.selectbox("Select symbol column", dff.columns)

    plot_types = ['Basic 3D Scatter Plot', 'Colorscaled 3D Scatter Plot', 'Ternary Scatter Plot', 'Distplot']
    plot_type = st.selectbox("Select plot type", plot_types)

    if plot_type == 'Basic 3D Scatter Plot':
        fig = go.Figure(data=[go.Scatter3d(x=dff[x_axis_column], y=dff[y_axis_column], z=dff[z_axis_column],
                                           mode='markers')])
        fig.update_layout(scene=dict(xaxis_title=x_axis_column, yaxis_title=y_axis_column, zaxis_title=z_axis_column))
        st.plotly_chart(fig)

    elif plot_type == 'Colorscaled 3D Scatter Plot':
        fig = go.Figure(data=[go.Scatter3d(
            x=dff[x_axis_column],
            y=dff[y_axis_column],
            z=dff[z_axis_column],
            mode='markers',
            marker=dict(
                size=12,
                color=dff[color_column],  # set color to an array/list of desired values
                colorscale='Viridis',  # choose a colorscale
                opacity=0.8
            )
        )])
        fig.update_layout(scene=dict(xaxis_title=x_axis_column, yaxis_title=y_axis_column, zaxis_title=z_axis_column))
        st.plotly_chart(fig)

    elif plot_type == 'Ternary Scatter Plot':
        fig = go.Figure(data=[go.Scatterternary(
            a=dff[x_axis_column],
            b=dff[y_axis_column],
            c=dff[z_axis_column],
            mode='markers',
            marker=dict(
                size=dff[size_column],
                color=dff[color_column],
                symbol=dff[symbol_column],
                colorscale='Viridis',
                opacity=0.8
            )
        )])
        st.plotly_chart(fig)

    elif plot_type == 'Distplot':
        # Filter out NaN and inf values
        filtered_dff = dff[[x_axis_column, y_axis_column, z_axis_column]].replace([np.inf, -np.inf], np.nan).dropna()

        # Add histogram data
        hist_data = [filtered_dff[x_axis_column], filtered_dff[y_axis_column], filtered_dff[z_axis_column]]
        group_labels = ['X-Axis Column', 'Y-Axis Column', 'Z-Axis Column']
        fig = ff.create_distplot(hist_data, group_labels, bin_size=[.1, .25, .5])
        st.plotly_chart(fig, use_container_width=True)
    # Button to show the plot in full-screen mode
    if st.button("Show Full Screen"):
        fig.show()

    
except Exception as e:
    st.error(e)