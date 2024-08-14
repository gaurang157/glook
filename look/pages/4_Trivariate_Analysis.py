# page2.py
import streamlit as st
import pandas as pd
from scipy.stats import zscore
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import plotly.figure_factory as ff
# import matplotlib.pyplot as plt
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
        # size_column = st.selectbox("Select size column", dff.columns)
        hover_name_column = st.selectbox("Select hover name column", dff.columns)

    plot_types = ['Basic 3D Scatter Plot', 'Colorscaled 3D Scatter Plot', 'Distplot']
    plot_type = st.selectbox("Select plot type", plot_types)

    if plot_type == 'Basic 3D Scatter Plot':
        fig = go.Figure(data=[go.Scatter3d(x=dff[x_axis_column], y=dff[y_axis_column], z=dff[z_axis_column], mode='markers', hovertext=hover_name_column)])
        fig.update_layout(scene=dict(xaxis_title=x_axis_column, yaxis_title=y_axis_column, zaxis_title=z_axis_column))
        st.plotly_chart(fig)

    elif plot_type == 'Colorscaled 3D Scatter Plot':
        try:
            fig = go.Figure(data=[go.Scatter3d(
                x=dff[x_axis_column],
                y=dff[y_axis_column],
                z=dff[z_axis_column],
                mode='markers',
                marker=dict(
                    size=12,
                    color=dff[color_column],  # set color to an array/list of desired values
                    colorscale='Viridis',  # choose a colorscale
                    opacity=0.8,
                ),hovertext=hover_name_column,
            )])
            fig.update_layout(scene=dict(xaxis_title=x_axis_column, yaxis_title=y_axis_column, zaxis_title=z_axis_column))
            st.plotly_chart(fig)
        except Exception as e:
            st.warning("Select Numeric Column for 'color column'")
    # elif plot_type == 'Ternary Scatter Plot':
    #     fig = go.Figure(data=[go.Scatterternary(
    #         a=dff[x_axis_column],
    #         b=dff[y_axis_column],
    #         c=dff[z_axis_column],
    #         mode='markers',
    #         marker=dict(
    #             color=dff[color_column],
    #             symbol=dff[symbol_column],
    #             colorscale='Viridis',
    #             opacity=0.8,
    #         )
    #     )])
    #     st.plotly_chart(fig)

    elif plot_type == 'Distplot':
        # Filter out NaN and inf values
        filtered_dff = dff[[x_axis_column, y_axis_column, z_axis_column]].replace([np.inf, -np.inf], np.nan).dropna()

        # Add histogram data
        hist_data = [filtered_dff[z_axis_column], filtered_dff[y_axis_column], filtered_dff[x_axis_column]]
        group_labels = [z_axis_column , y_axis_column, x_axis_column]
        fig = ff.create_distplot(hist_data, group_labels, bin_size=[.1, .25, .5])
        st.plotly_chart(fig, use_container_width=True)



    # Button to show the plot in full-screen mode
    if st.button("Show Full Screen in New Tab"):
        fig.show()
except NameError:
    st.warning("Data not Uploaded")
except Exception as e:
    # st.warning("Select Proper Column")
    st.warning("Select Numeric Column for 'color column' as well as for X, Y & Z axis")
    # st.error(e)
    # print(e)