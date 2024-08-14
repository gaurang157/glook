# pip or conda to install openpyxl.
# page 1
import streamlit as st
import pandas as pd
import numpy as np
st.set_page_config(
	# page_title="ML-Automation", 
	page_title="Auto - EDA", 
	page_icon="👁️", 
	layout="centered", 
	initial_sidebar_state = "expanded",
	menu_items={
        'Get Help': 'https://github.com/gaurang157/glook/blob/main/README.md',
        'Report a bug': "https://github.com/gaurang157/glook/issues",
        'About': "# This is Auto EDA Library. This is an *extremely* cool app!"
		}
		)
# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('TkAgg')
if "shared" not in st.session_state:
	st.session_state["shared"] = True

st.title("G-Look Auto EDA",)
# st.write("`")
gg = st.file_uploader("Input:", type=['csv', 'xlsx'])

# df = None
# if st.button("process", use_container_width=True):
# 	try:
# 		# Try reading as CSV
# 		df = pd.read_csv(gg)
# 	except:
# 		try:
# 			# Try reading as Excel
# 			df = pd.read_excel(gg, engine='openpyxl')
# 		except Exception as e:
# 			st.error(f"Error reading file: {e}")
# 	# Store the DataFrame in session_state
# 	st.session_state.df = df
# 	st.switch_page("pages/2_Univariate_Analysis.py")
# if gg is not None:
# 	try:
# 		# Try reading as CSV
# 		df = pd.read_csv(gg)
# 	except:
# 		try:
# 			# Try reading as Excel
# 			df = pd.read_excel(gg, engine='openpyxl')
# 		except Exception as e:
# 			# st.error(f"Error reading file: {e}")
# 			print(e)

# if df is not None:
# 	# st.dataframe(df)
# 	# Get number of rows
# 	num_rows = df.shape[0]

# 	# Get number of columns
# 	num_columns = df.shape[1]

# 	# Check for duplicates
# 	num_duplicates = df.duplicated().sum()

# 	# Get memory usage
# 	memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)  # Convert bytes to KB

# 	# Get number of features
# 	num_features = len(df.columns)

# 	# Get number of categorical features
# 	num_categorical = len(df.select_dtypes(include=['object']).columns)

# 	# Get number of numerical features
# 	num_numerical = len(df.select_dtypes(include=['number']).columns)

# 	# Display the insights
# 	st.subheader(f"Data dimensions: :green[{num_rows} rows, {num_columns} columns]")
# 	st.subheader(f"Number of rows: :green[{num_rows}]")
# 	st.subheader(f"Number of duplicates: :green[{num_duplicates}]")
# 	st.subheader(f"Memory usage: :green[{memory_usage:.5f} KB]")
# 	st.subheader(f"Number of features: :green[{num_features}]")
# 	st.subheader(f"Number of categorical features: :green[{num_categorical}]")
# 	st.subheader(f"Number of numerical features: :green[{num_numerical}]")
# 	st.dataframe(df)



# else:
# 	st.write("DataFrame not yet uploaded.")

# print(9)

if st.button("process", type='primary', use_container_width=True):
	try:
		# Try reading as CSV
		df = pd.read_csv(gg)
	except:
		try:
			# Try reading as Excel
			df = pd.read_excel(gg, engine='openpyxl')
		except Exception as e:
			st.error(f"Error reading file: {e}")
	# Store the DataFrame in session_state
	st.session_state.df = df
	st.session_state.df0 = df
	st.switch_page("pages/1_General_Data_Insights.py")
	# st.switch_page("pages/2_Univariate_Analysis.py")