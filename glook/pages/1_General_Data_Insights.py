import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from io import StringIO
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
st.write("Session State:->", st.session_state["shared"])


if "df" in st.session_state and st.session_state.df is not None:
	# df = st.session_state.df
	oragnial_df = st.session_state.df
	df_to_pre = st.session_state.df_pre
	# st.warning("kl")
	to_select = st.selectbox("Select Data Frame (Recommended Oragnial DF)", ["oragnial_df", "df_to_pre_process"], index=0)
	if to_select == "oragnial_df":
		df = oragnial_df
	elif to_select == "df_to_pre_process":
		df = df_to_pre
		
	# Get number of rows
	num_rows = df.shape[0]

	# Get number of columns
	num_columns = df.shape[1]

	# Check for duplicates
	num_duplicates = df.duplicated().sum()

	# Get memory usage
	memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)  # Convert bytes to MB
	# Calculate total memory usage in bytes
	# memory_usage = df.memory_usage(deep=True).sum()

	# Convert bytes to megabytes
	# memory_usage = memory_usage / (1024 ** 2)
	# Get number of features
	num_features = len(df.columns)

	# Get number of categorical features
	num_categorical = len(df.select_dtypes(include=['object']).columns)

	# Get number of numerical features
	num_numerical = len(df.select_dtypes(include=['number']).columns)

	st.header("Dataframe :green[Overview]")
	st.dataframe(df)
	# Display the insights
	st.subheader(f"Data dimensions: :green[{num_rows} rows, {num_columns} columns]")
	st.subheader(f"Number of rows: :green[{num_rows}]")
	st.subheader(f"Number of duplicates: :green[{num_duplicates}]")
	st.subheader(f"Deep Memory usage: :green[{memory_usage:.3f} MB]")
	st.subheader(f"Number of features: :green[{num_features}]")
	st.subheader(f"Number of categorical features: :green[{num_categorical}]")
	st.subheader(f"Number of numerical features: :green[{num_numerical}]")
	st.subheader("Customize DataFrame Values to Fit Your Requirements:")
	st.text("Note: Double tap to Change Values")
	df = st.data_editor(df)
	if to_select == "oragnial_df":
		st.session_state.df = df
	elif to_select == "df_to_pre_process":
		st.session_state.df_pre = df

	try:
		st.header("Numerical :green[Data Overview]")
		# df_info = df.describe().T
		# styled_df_info = df_info.style.highlight_max(axis=0).highlight_min(axis=0).format("{:.2f}")
		df_info = (
			df.describe().T.style.set_table_styles(
				[
					{'selector': 'th', 'props': 'background-color: lightgreen; color: black;'},  # Table headers
					{'selector': 'td', 'props': 'background-color: lightblue; border: 1px solid black;'},  # Table cells
					{'selector': 'tr', 'props': 'border: 1px solid black;'}  # Table rows
				]
			)
			.highlight_max(axis=0, props='background-color: darkgreen; color: white;')  # Highlight max values with specific color
			.highlight_min(axis=0, props='background-color: yellowgreen; color: black;')  # Highlight min values with another color
			.format("{:.2f}")
		)
		st.dataframe(df_info)
	except ValueError:
		st.dataframe()
	except Exception as e:
		st.dataframe()
		pass

	try:
		st.header("Categorical :green[Data Overview]")

		df_info1 = df.select_dtypes(include=['object']).describe().T
		# Style to highlight the most frequent category
		styled_df_info1 = (
		df_info1.style.set_table_styles(
				[
					{'selector': 'th', 'props': 'background-color: lightgreen; color: black;'},  # Table headers
					{'selector': 'td', 'props': 'background-color: lightblue; border: 1px solid black;'},  # Table cells
					{'selector': 'tr', 'props': 'border: 1px solid black;'}  # Table rows
				]
			)
			.highlight_max(axis=0, props='background-color: darkgreen; color: white;')  # Highlight max values with specific color
			.highlight_min(axis=0, props='background-color: yellowgreen; color: black;')  # Highlight min values with another color
			# .format("{:.2f}")
		)

		st.dataframe(styled_df_info1)
	except ValueError:
		st.dataframe()
	except Exception as e:
		st.dataframe()
		pass


	# Capture the output of df.info()
	info_buffer = StringIO()
	df.info(buf=info_buffer)
	info_str = info_buffer.getvalue()

	# Display the info in Streamlit
	st.subheader("DataFrame :green[Info:]")
	st.code(info_str, language="neon")
	


else:
	st.write("DataFrame not yet uploaded.")
# if st.sidebar.button("Univariate Analysis"):
	# st.switch_page("pages/2_Univariate_Analysis.py")

if st.button("Univariate_Analysis", use_container_width=True):
	st.switch_page("pages/2_Univariate_Analysis.py")