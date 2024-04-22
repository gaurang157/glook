# page2.py
import streamlit as st
import pandas as pd
from scipy.stats import zscore
import plotly.express as px
import plotly.graph_objects as go
# import matplotlib
# matplotlib.use('TkAgg')
st.title("2️⃣ Bivariate Analysis")
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

    # Streamlit UI for selecting plot type, x-axis, and y-axis columns
    plot_type = st.selectbox("Select plot type", ["Scatter Plot", "Line Plot", "Bar Plot", "Histogram", "Box Plot", "Violin Plot", "Strip Chart", "Density Contour", "Density Heatmap", "Polar Scatter Plot", "Polar Line Plot", "Polar Bar Plot"])
    
    col1, col2 = st.columns(2)
    with col1:
        x_axis_column2 = st.selectbox("Select x-axis column", dff.columns)
    with col2:
        y_axis_column2 = st.selectbox("Select y-axis column", dff.columns)

    # Create the plot based on selected plot type
    if plot_type == "Scatter Plot":
        fig = px.scatter(dff, x=x_axis_column2, y=y_axis_column2, log_x=True, size_max=60)
    elif plot_type == "Line Plot":
        fig = px.line(dff, x=x_axis_column2, y=y_axis_column2)
    elif plot_type == "Bar Plot":
        fig = px.bar(dff, x=x_axis_column2, y=y_axis_column2)
    elif plot_type == "Histogram":
        fig = px.histogram(dff, x=x_axis_column2, y=y_axis_column2, hover_data=dff.columns)
    elif plot_type == "Box Plot":
        fig = px.box(dff, x=x_axis_column2, y=y_axis_column2, notched=True)
    elif plot_type == "Violin Plot":
        fig = px.violin(dff, y=y_axis_column2, x=x_axis_column2, box=True, points="all", hover_data=dff.columns)
    elif plot_type == "Strip Chart":
        fig = px.strip(dff, x=x_axis_column2, y=y_axis_column2, orientation="h")
    elif plot_type == "Density Contour":
        fig = px.density_contour(dff, x=x_axis_column2, y=y_axis_column2)
    elif plot_type == "Density Heatmap":
        fig = px.density_heatmap(dff, x=x_axis_column2, y=y_axis_column2, marginal_x="rug", marginal_y="histogram")
    elif plot_type == "Polar Scatter Plot":
        fig = px.scatter_polar(dff, r=x_axis_column2, theta=y_axis_column2, color=x_axis_column2, symbol=x_axis_column2,
                    color_discrete_sequence=px.colors.sequential.Plasma_r)
    elif plot_type == "Polar Line Plot":
        fig = px.line_polar(dff, r=x_axis_column2, theta=y_axis_column2, color=x_axis_column2, line_close=True,
                    color_discrete_sequence=px.colors.sequential.Plasma_r)
    elif plot_type == "Polar Bar Plot":
        fig = px.bar_polar(dff, r=x_axis_column2, theta=y_axis_column2, color=x_axis_column2, template="plotly_dark",
                    color_discrete_sequence= px.colors.sequential.Plasma_r)

    # Show the plot
    st.plotly_chart(fig)
    # Button to show the plot in full-screen mode
    if st.button("Show Full Screen"):
        fig.show()
    # Handle errors
except Exception as e:
    st.error(e)
    # st.subheader("⚠️Please upload a file⚠️")
    st.warning("Select Proper Column")






# Exclude non-numeric columns
numeric_df = df.select_dtypes(include='number')

# Convert non-numeric values to NaN
numeric_df = numeric_df.apply(pd.to_numeric, errors='coerce')

# Calculate the correlation matrix
correlation_matrix = numeric_df.corr()

# Create the heatmap plot
fig = go.Figure(data=go.Heatmap(
    z=correlation_matrix.values,
    x=correlation_matrix.columns,
    y=correlation_matrix.index,
    colorscale='aggrnyl',  # You can choose any colorscale
))

# Add title and axis labels
fig.update_layout(
    title='Correlation Coefficient Heatmap',
    xaxis=dict(title='Columns'),
    yaxis=dict(title='Columns'),
    width=800,  # Adjust width
    height=800,  # Adjust height
)

# Show the plot
st.plotly_chart(fig)



