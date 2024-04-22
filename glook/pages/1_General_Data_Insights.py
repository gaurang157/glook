import streamlit as st
import pandas as pd
import plotly.graph_objects as go
st.set_page_config(page_title="GLook", layout="wide")
st.write("Session State:->", st.session_state["shared"])


if "df" in st.session_state and st.session_state.df is not None:
	df = st.session_state.df
	# Get number of rows
	num_rows = df.shape[0]

	# Get number of columns
	num_columns = df.shape[1]

	# Check for duplicates
	num_duplicates = df.duplicated().sum()

	# Get memory usage
	memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)  # Convert bytes to KB
	# Get number of features
	num_features = len(df.columns)

	# Get number of categorical features
	num_categorical = len(df.select_dtypes(include=['object']).columns)

	# Get number of numerical features
	num_numerical = len(df.select_dtypes(include=['number']).columns)

	
	# st.dataframe(df)
	# Exclude non-numeric columns
	numeric_df = df.select_dtypes(include='number')

	# Convert non-numeric values to NaN
	numeric_df = numeric_df.apply(pd.to_numeric, errors='coerce')
	st.header(":green[Correlation Coefficient Heatmap]")
	# Calculate the correlation matrix
	correlation_matrix = numeric_df.corr()
	# Define the list of colorscales
	# colorscales = ['aggrnyl', 'agsunset', 'algae', 'amp', 'armyrose', 'balance', 'blackbody', 'bluered', 'blues', 'blugrn', 'bluyl', 'brbg', 'brwnyl', 'bugn', 'bupu', 'burg', 'burgyl', 'cividis', 'curl', 'darkmint', 'deep', 'delta', 'dense', 'earth', 'edge', 'electric', 'emrld', 'fall', 'geyser', 'gnbu', 'gray', 'greens', 'greys', 'haline', 'hot', 'hsv', 'ice', 'icefire', 'inferno', 'jet', 'magenta', 'magma', 'matter', 'mint', 'mrybm', 'mygbm', 'oranges', 'orrd', 'oryel', 'oxy', 'peach', 'phase', 'picnic', 'pinkyl', 'piyg', 'plasma', 'plotly3', 'portland', 'prgn', 'pubu', 'pubugn', 'puor', 'purd', 'purp', 'purples', 'purpor', 'rainbow', 'rdbu', 'rdgy', 'rdpu', 'rdylbu', 'rdylgn', 'redor', 'reds', 'solar', 'spectral', 'speed', 'sunset', 'sunsetdark', 'teal', 'tealgrn', 'tealrose', 'tempo', 'temps', 'thermal', 'tropic', 'turbid', 'turbo', 'twilight', 'viridis', 'ylgn', 'ylgnbu', 'ylorbr', 'ylorrd']
	# Number input for selecting the colorscale index
	# colorscale_index = st.number_input('Select a colorscale index:', min_value=0, max_value=len(colorscales)-1, value=0)
	# st.write(colorscales[colorscale_index])
	# Create the heatmap plot
	fig = go.Figure(data=go.Heatmap(
		z=correlation_matrix.values,
		x=correlation_matrix.columns,
		y=correlation_matrix.index,
		# colorscale=colorscales[colorscale_index],  # You can choose any colorscale
		colorscale='aggrnyl'
	))

	# Add title and axis labels
	fig.update_layout(
		title='Correlation Coefficient Heatmap',
		xaxis=dict(title='Columns'),
		yaxis=dict(title='Columns'),
		width=700,  # Adjust width
		height=700,  # Adjust height
	)

	# Show the plot
	st.plotly_chart(fig)

	st.data_editor(df)
	
	# Display the insights
	st.subheader(f"Data dimensions: :green[{num_rows} rows, {num_columns} columns]")
	st.subheader(f"Number of rows: :green[{num_rows}]")
	st.subheader(f"Number of duplicates: :green[{num_duplicates}]")
	st.subheader(f"Memory usage: :green[{memory_usage:.5f} KB]")
	st.subheader(f"Number of features: :green[{num_features}]")
	st.subheader(f"Number of categorical features: :green[{num_categorical}]")
	st.subheader(f"Number of numerical features: :green[{num_numerical}]")
	


else:
	st.write("DataFrame not yet uploaded.")
# if st.sidebar.button("Univariate Analysis"):
	# st.switch_page("pages/2_Univariate_Analysis.py")

if st.button("Univariate_Analysis", use_container_width=True):
	st.switch_page("pages/2_Univariate_Analysis.py")
