# page2.py
import streamlit as st
import pandas as pd
from scipy.stats import zscore
import plotly.express as px
import plotly.graph_objects as go
# import matplotlib
# matplotlib.use('TkAgg')
st.set_page_config(
	# page_title="ML-Automation", 
	page_title="Auto - EDA", 
	page_icon="👁️", 
	layout="wide", 
	initial_sidebar_state = "expanded",
	menu_items={
        'Get Help': 'https://github.com/gaurang157/glook/blob/main/README.md',
        'Report a bug': "https://github.com/gaurang157/glook/issues",
        'About': "# This is Auto EDA Library. This is an *extremely* cool app!"
		}
		)
st.title("2️⃣ Bivariate Analysis")
try:
    st.write("Session State:->", st.session_state["shared"])
    # Access the DataFrame from session_state
    if "df" in st.session_state:
        oragnial_df = st.session_state.df
        df_to_pre = st.session_state.df_pre
        # st.warning("kl")
        to_select = st.selectbox("Select Data Frame (Recommended Oragnial DF)", ["oragnial_df", "df_to_pre_process"], index=0)
        if to_select == "oragnial_df":
            df = oragnial_df
        elif to_select == "df_to_pre_process":
            df = df_to_pre
		
        # st.dataframe(df)

    else:
        st.write("DataFrame not found.")

    print(type(df))

    dff = df

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
		text=correlation_matrix.values,  # Values to display
		texttemplate='%{text:.2f}',  # Format for displaying values
		showscale=True  # Display the color scale on the side        
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


    # Streamlit UI for selecting plot type, x-axis, and y-axis columns
    plot_type = st.selectbox("Select plot type", ["Scatter Plot", "Bar Plot", "Box Plot", "Violin Plot", "Strip Chart", "Density Contour", "Density Heatmap", "Polar Bar Plot"])
    
    col1, col2 = st.columns(2)
    with col1:
        x_axis_column2 = st.selectbox("Select x-axis column", dff.columns)
    with col2:
        y_axis_column2 = st.selectbox("Select y-axis column", dff.columns)

    # Create the plot based on selected plot type
    if plot_type == "Scatter Plot":
        fig = px.scatter(dff, x=x_axis_column2, y=y_axis_column2, log_x=True, size_max=60)
    elif plot_type == "Bar Plot":
        catch = st.toggle("Show All Columns value on hover")
        if catch:
            fig = px.bar(dff, x=x_axis_column2, y=y_axis_column2, hover_data=dff.columns)
        else:
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
    elif plot_type == "Polar Bar Plot":
        fig = px.bar_polar(dff, r=x_axis_column2, theta=y_axis_column2, color=x_axis_column2, template="plotly_dark",
                    color_discrete_sequence= px.colors.sequential.Plasma_r)

    # Show the plot
    st.plotly_chart(fig)
    # Button to show the plot in full-screen mode
    if st.button("Show Full Screen"):
        fig.show()
    # Handle errors

except NameError:
    st.warning("Data not Uploaded")
except Exception as e:
    # st.error(e)
    print(e)
    # st.subheader("⚠️Please upload a file⚠️")
    st.warning("Select Proper Column")
