# page2.py
try:
	import streamlit as st
	st.set_page_config(
			page_title="ML-Automation", 
			page_icon="🏗️", 
			layout="wide", 
			initial_sidebar_state = "expanded")

	import pandas as pd
	import seaborn as sns
	import matplotlib.pyplot as plt
	import plotly.express as px
	import plotly.graph_objects as go
	import numpy as np
	import statsmodels.api as sm
	st.set_option('deprecation.showPyplotGlobalUse', False)
	import matplotlib
	matplotlib.use('Agg')
except Exception as e:
	print("z", e)

# from wordcloud import WordCloud
# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('TkAgg')
# st.title("1️⃣Univariate Analysis")


st.write("Session State:->", st.session_state["shared"])
# st.session_state["shared"]
# Function to limit the number of unique values for a categorical column
def limit_unique_values(series, limit=20):
	if series.nunique() > limit:
		top_categories = series.value_counts().index[:limit]
		return series.apply(lambda x: x if x in top_categories else 'Other')
	else:
		return series

def limit_unique_values_cloud(series, limit=200):
	if series.nunique() > limit:
		top_categories = series.value_counts().index[:limit]
		return top_categories
	else:
		return series.unique()

# def limit_unique_values_cloud(series, limit=200):
# 	if series.nunique() > limit:
# 		top_categories = series.value_counts().index[:limit]
# 		return series.apply(lambda x: x if x in top_categories else 'Other')
# 	else:
# 		return series



def calculate_insights(column_data):
	insights = {}

	# Count of distinct values
	insights['Distinct'] = len(column_data.dropna().unique())

	# Percentage of distinct values
	insights['Distinct (%)'] = len(column_data.dropna().unique()) / len(column_data.dropna()) * 100

	# Count of missing values
	insights['Missing'] = column_data.isnull().sum()

	# Percentage of missing values
	insights['Missing (%)'] = column_data.isnull().sum() / len(column_data) * 100

	# Count of infinite values
	insights['Infinite'] = np.isinf(column_data).sum()

	# Percentage of infinite values
	insights['Infinite (%)'] = np.isinf(column_data).sum() / len(column_data) * 100

	# Mean
	insights['Mean'] = column_data.mean()

	# Median
	insights['Median'] = column_data.median()

	# Mode
	insights['Mode'] = column_data.mode().iloc[0]

	# Minimum
	insights['Minimum'] = column_data.min()

	# Maximum
	insights['Maximum'] = column_data.max()

	# Zeros count
	insights['Zeros'] = (column_data == 0).sum()

	# Percentage of zeros
	insights['Zeros (%)'] = (column_data == 0).sum() / len(column_data) * 100

	# Negative values count
	insights['Negative'] = (column_data < 0).sum()

	# Percentage of negative values
	insights['Negative (%)'] = (column_data < 0).sum() / len(column_data) * 100

	# Memory size
	insights['Memory size'] = column_data.memory_usage(deep=True)

	# 5th percentile
	insights['5-th percentile'] = np.percentile(column_data.dropna(), 5)

	# Q1 (25th percentile)
	insights['Q1'] = np.percentile(column_data.dropna(), 25)

	# Median
	insights['Median'] = np.median(column_data.dropna())

	# Q3 (75th percentile)
	insights['Q3'] = np.percentile(column_data.dropna(), 75)

	# 95th percentile
	insights['95-th percentile'] = np.percentile(column_data.dropna(), 95)

	# Range
	insights['Range'] = insights['Maximum'] - insights['Minimum']

	# Interquartile range (IQR)
	insights['Interquartile range'] = insights['Q3'] - insights['Q1']

	# Descriptive statistics
	insights['Descriptive statistics'] = column_data.describe()

	# Standard deviation
	insights['Standard deviation'] = column_data.std()

	# Coefficient of variation (CV)
	insights['Coefficient of variation (CV)'] = insights['Standard deviation'] / insights['Mean']

	# Kurtosis
	insights['Kurtosis'] = column_data.kurtosis()

	# Median Absolute Deviation (MAD)
	mad = np.median(np.abs(column_data.dropna() - np.median(column_data.dropna())))
	insights['Median Absolute Deviation (MAD)'] = mad

	# Skewness
	insights['Skewness'] = column_data.skew()

	# Sum
	insights['Sum'] = column_data.sum()

	# Variance
	insights['Variance'] = column_data.var()

	# Monotonicity (not calculated)

	return insights


# if "df" in st.session_state:
# 	df = st.session_state.df



try:
	# Assuming df is already defined and stored in session_state
	
	if "df" in st.session_state and st.session_state.df is not None:
		df = st.session_state.df
		# import time
		# s = time.time()
		# # Get number of rows
		# num_rows = df.shape[0]
		
		# # Get number of columns
		# num_columns = df.shape[1]
		
		# # Check for duplicates
		# num_duplicates = df.duplicated().sum()
		
		# # Get memory usage
		# memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)  # Convert bytes to KB
		
		# # Get number of features
		# num_features = len(df.columns)
		
		# # Get number of categorical features
		# num_categorical = len(df.select_dtypes(include=['object']).columns)
		
		# # Get number of numerical features
		# num_numerical = len(df.select_dtypes(include=['number']).columns)
		
		# # Display the general insights
		# st.subheader(f"Data dimensions: :green[{num_rows} rows, {num_columns} columns]")
		# st.subheader(f"Number of duplicates: :green[{num_duplicates}]")
		# st.subheader(f"Memory usage: :green[{memory_usage:.1f} KB]")
		# st.subheader(f"Number of features: :green[{num_features}]")
		# st.subheader(f"Number of categorical features: :green[{num_categorical}]")
		# st.subheader(f"Number of numerical features: :green[{num_numerical}]")
		
		# Display insights for each column
		st.title("1️⃣ Univariate Analysis")                                                                  # PROBLEM
		# st.header(":green[Column Insights:]⤵️")
		# html_content = ""
		for idx, column in enumerate(df.columns):
			# st.subheader(f':green[**Column {idx + 1}:** {column}]')
			# subh = f""":green[Column {idx + 1}: {column}]"""
			# st.subheader(subh)
			# subh = f"**Column {idx + 1}:** `{column}`"
			# st.markdown(subh, unsafe_allow_html=True)
			# subh = "**Column " + str(idx + 1) + ":** `" + column + "`"
			# st.markdown(subh, unsafe_allow_html=True)
			html_content = f"<div class='column-header'>Column {idx + 1}: <code>{column}</code></div>"

			css_style = """
			<style>
			.column-header {
				margin-bottom: 10px;
				font-weight: bold;
				font-size: 26px;
				color: green; /* Change color as needed */
			}
			</style>
			"""

			# Embed HTML and CSS code into Streamlit
			st.markdown(css_style, unsafe_allow_html=True)
			st.markdown(html_content, unsafe_allow_html=True)
			if df[column].dtype == 'object':
				col1, col2 = st.columns(2)
				with col1:
					unique_values = df[column].nunique()
					st.write(f"  - Data Type: :green[{df[column].dtype}]")
					st.write(f"  - Number of Unique Values: :green[{unique_values}]")
					if unique_values <= 20:
						st.write(f"  - Unique Values: :green[{', '.join(map(str, df[column].unique()))}]")
					else:
						st.write(f"  - Top 20 Unique Values:")
						st.write(f":green[{', '.join(map(str, df[column].value_counts().head(20).index))}]")
				with col2:
					# """ distinct plot """
					plt.figure(figsize=(10, 6))  # Adjust figure size
					try:
						sns.countplot(x=limit_unique_values(df[column]), data=df, color="green")
					except:
						sns.countplot(x=df[column], data=df, color="green")
					plt.xticks(rotation=45)  # Rotate x-axis labels
					st.pyplot()
					plt.close()
	 				# Assuming df is your DataFrame and column is the specific column you want to plot
					# def distinct_plot(df, column):
					# 	try:
					# 		fig = px.histogram(df, x=limit_unique_values(df[column]), color_discrete_sequence=["green"])
					# 	except:
					# 		fig = px.histogram(df, x=df[column], color_discrete_sequence=["green"])
						
					# 	fig.update_layout(xaxis_tickangle=-45)  # Rotate x-axis labels
					# 	st.plotly_chart(fig, use_container_width=True)

					# # Call the function with your DataFrame and column
					# distinct_plot(df, column)
				with st.expander("More Info"):
					# st.write("More Info ⬇")
					# tab1, tab2, tab3 = st.tabs(["Insights", "Donut chart", "WordCloud"])
					tab1, tab2 = st.tabs(["Insights", "Donut chart"])
					with tab1:
						col7, col8, col9 = st.columns(3)
						with col7:
							# Insights
							st.write("## Insights")
							approximate_distinct_count = df[column].nunique()
							approximate_unique_percent = (approximate_distinct_count / len(df)) * 100
							missing = df[column].isna().sum()
							missing_percent = (missing / len(df)) * 100
							memory_size = df[column].memory_usage(deep=True)
							st.write(f"Approximate Distinct Count: :green[{approximate_distinct_count}]")
							st.write(f"Approximate Unique (%): :green[{approximate_unique_percent:.2f}%]")
							st.write(f"Missing: :green[{missing}]")
							st.write(f"Missing (%): :green[{missing_percent:.2f}%]")
							st.write(f"Memory Size: :green[{memory_size}]")

							# st.header("An owl")
							# st.image("https://static.streamlit.io/examples/owl.jpg", width=200)
						with col8:
							# Mode and Standard Deviation
							# st.write("## Mode")
							# # if df[column].dtype == 'object':
							# mode = df[column].mode().iloc[0]  # Mode can return multiple values, taking the first one
							# st.write(f"Mode: :green[{mode}]")
							multi_mode = df[column].value_counts()
							st.write("## Frequency")
							if len(multi_mode) >= 5:
								st.dataframe(multi_mode, height=200)

							elif len(multi_mode) < 5:
								st.dataframe(multi_mode)
								# st.write("Standard Deviation cannot be calculated for non-numeric data.")
							# else:
								# mode = df[column].mode().iloc[0]  # Mode can return multiple values, taking the first one
								# std_dev = df[column].astype(float).std()
								# st.write(f"Mode: {mode}")
								# st.write(f"Standard Deviation: {std_dev}")
						with col9:
							# First 5 Sample Rows
							st.write("## First 5 Sample Rows")
							st.write(df[column].head())

					with tab2:
						# Prepare data for donut chart
						data = limit_unique_values(df[column]).value_counts().reset_index()
						data.columns = [column, 'count']

						# fig = px.pie(data, values='count', names=column, hole=0.4)

						# Create a pie chart with a donut style (hole)
						# Create a pie chart with a donut style (hole)
						fig = px.pie(data, values='count', names=column, hole=0.5)

						# Customize the traces
						fig.update_traces(
							textposition='inside',  # Place labels inside pie slices
							textinfo='percent+label',  # Show percentage and label
							marker=dict(
								line=dict(color='black', width=3.5)  # Black border for contrast
							),
							pull=0.05,  # Pull segments outward slightly when hovered
							hoverinfo='label+percent+name',  # Display information on hover
							hoverlabel=dict(bgcolor='white', font_size=12, font_color='black')  # Customize hover label appearance
						)

						# Customize the layout with proper legend position and margins
						fig.update_layout(
							title=f'Category Distribution for {str(column)}',  # Title of the pie chart
							legend=dict(
								orientation="v",  # Vertical legend
								yanchor="top",  # Anchor to the top
								y=1,  # Position along y-axis
								xanchor="left",  # Anchor to the left
								x=-0.35  # Left position (x-axis) to ensure it's away from the plot
							),
							width=600,  # Adjust the width
							height=600,  # Adjust the height
							margin=dict(l=50, r=50, t=50, b=50),  # Increase left margin for more space
							showlegend=True  # Display the legend
						)
						st.plotly_chart(fig)
						print("j")
						# st.header("An owl")
						# st.image("https://static.streamlit.io/examples/owl.jpg", width=200)

					# with tab3:
					# 	# Concatenate all values of the column into a single string
					# 	cleanstrng = ' '.join(df[column].dropna())

					# 	# Check if there are words to plot
					# 	if cleanstrng:
					# 		# Generate word cloud
					# 		word_freq = pd.Series(cleanstrng.split()).value_counts()
							
					# 		# Select top 200 words if available, otherwise take the exact number of words
					# 		if len(word_freq) >= 50:
					# 			top_words = word_freq.head(50)
					# 		else:
					# 			top_words = word_freq

					# 		words_dict = top_words.to_dict()
							
					# 		try:
					# 			# Generate word cloud from frequencies
					# 			wordcloud_ip = WordCloud(background_color='white', width=2800, height=2400).generate_from_frequencies(words_dict)

					# 			# Display word cloud
					# 			plt.figure(figsize=(10, 7))
					# 			plt.imshow(wordcloud_ip, interpolation='bilinear')
					# 			plt.title(f'Word Cloud for {column}')
					# 			plt.axis('off')
					# 			plt.show()
					# 			st.pyplot(plt, use_container_width=True)
					# 		except Exception as e:
					# 			st.write(f"Error generating word cloud: {e}")
					# 	else:
					# 		st.write(f"No words to plot for column '{column}'")
					
			elif pd.api.types.is_numeric_dtype(df[column]):
				print("0a")
				col3, col4 = st.columns(2)
				with col3:
					print("a")
					st.write(f"  - Data Type: :green[{df[column].dtype}]")
					print("b")
					st.write(f"  - Mean: :green[{df[column].mean()}]")
					st.write(f"  - Standard Deviation: :green[{df[column].std()}]")
					st.write(f"  - Min Value: :green[{df[column].min()}]")
					st.write(f"  - Max Value: :green[{df[column].max()}]")
				with col4:
					# """ KDE plot """
					plt.figure(figsize=(10, 6))  # Adjust figure size
					sns.histplot(df[column], kde=True, color="green")
					st.pyplot()
					plt.close()
	 				# KDE plot using Plotly
					# fig = px.histogram(df, x=df[column], marginal="rug", nbins=30, histnorm='probability density', 
                  	#  title="KDE Plot", color_discrete_sequence=['green'])
					# # Create KDE plot using Plotly
					# # Show the plot using Streamlit
					# st.plotly_chart(fig, use_container_width=True)
						# Create KDE plot using Plotly Express
					# fig = px.histogram(df, x=df[column], marginal="rug", nbins=30, histnorm='probability density', 
					# 				title="KDE Plot", color_discrete_sequence=['green'],opacity=0.5)

					# # Add labels and annotations
					# fig.update_layout(
					# 	xaxis_title=column,
					# 	yaxis_title="Density",
					# 	showlegend=False,
					# 	autosize=True,
					# 	width=800,
					# 	height=500,
					# 	margin=dict(l=40, r=40, t=40, b=40),
					# )

					# # Customize the appearance of the KDE plot
					# fig.update_traces(marker=dict(opacity=0.5))

					# # Display the plot using Streamlit
					# st.plotly_chart(fig, use_container_width=True)
				with st.expander("More Info"):
					# st.write("The chart above shows some numbers I picked for you.")
					tab1, tab2, tab3 = st.tabs(["Insights", "Box plot", "QQ plot"])
					with tab1:
						# st.header("An owl")
						# st.image("https://static.streamlit.io/examples/owl.jpg", width=200)
						col4, col5, col6 = st.columns(3)
						with col4:
							st.write("#### Basic Statistics")
							insights = calculate_insights(df[column])
							basic_stats = {key: value for key, value in insights.items() if key in [
								'Mean', 'Median', 'Mode', 'Standard deviation', 'Variance', 'Kurtosis', 'Skewness']}
							for key, value in basic_stats.items():
								st.write(f"**{key}:** :green[{value:.3f}]")
							st.write(f"**Memory size:** :green[{insights.get('Memory size', 'N/A'):.3f}]")
							st.write(f"**Range:** :green[{insights.get('Range', 'N/A'):.3f}]")
							st.write(f"**Interquartile range:** :green[{insights.get('Interquartile range', 'N/A'):.3f}]")

						with col5:
							st.write("#### Percentiles")
							descriptive_stats = insights.get('Descriptive statistics')
							if descriptive_stats is not None:
								percentiles = descriptive_stats.loc[['min', '25%', '50%', '75%', 'max']]
								if '5%' in descriptive_stats.index:
									percentiles['5%'] = descriptive_stats['5%']
								if '95%' in descriptive_stats.index:
									percentiles['95%'] = descriptive_stats['95%']
								st.write(percentiles)

						with col6:
							st.write("#### Additional Statistics")
							additional_stats = {key: value for key, value in insights.items() if key in [
								'Distinct', 'Distinct (%)', 'Missing', 'Missing (%)', 'Zeros', 'Zeros (%)', 'Negative', 'Negative (%)']}
							for key, value in additional_stats.items():
								st.write(f"**{key}:** :green[{value:.3f}]")
							# st.write(f"**Memory size:** {insights.get('Memory size', 'N/A')}")
							# st.write(f"**Range:** {insights.get('Range', 'N/A')}")
							# st.write(f"**Interquartile range:** {insights.get('Interquartile range', 'N/A')}")
							st.write(f"**Coefficient of variation (CV):** :green[{insights.get('Coefficient of variation (CV)', 'N/A'):.3f}]")
							st.write(f"**Median Absolute Deviation (MAD):** :green[{insights.get('Median Absolute Deviation (MAD)', 'N/A'):.3f}]")
							st.write(f"**Sum:** :green[{insights.get('Sum', 'N/A'):.3f}]")
						

					with tab2:
						fig = px.box(df, y=column, title=str(column))
						# Center the title by setting title_x to 0.5
					# Customize the box plot with trace-specific properties
						fig.update_traces(
							marker=dict(color='forestgreen'), #yellowgreen # Set the color to yellow-green
							boxmean=True,  # Display the mean as a line in the box plot
						)

						# Set the plot title, center it, and adjust the height
						fig.update_layout(
							title={
								'text': f'Box Plot for {column}',
								'x': 0.5,  # Center the title
								'xanchor': 'center',  # Title alignment to be centered
								'yanchor': 'top'  # Title anchor point
							},
							height=500,  # Set the plot height
						)
						st.plotly_chart(fig)

					with tab3:
						# plt.figure(figsize=(10, 6))  # Adjust figure size
						missing_value = additional_stats.get('Missing')
						if missing_value != 0:
							st.write(":grey[Note: The theoretical line will be visible when there are no missing values.]")
						# Generate QQ plot
						qqplot_data = sm.qqplot(df[column], line='s').gca().lines

						fig = go.Figure()
						fig.add_trace({
							'type': 'scatter',
							'x': qqplot_data[0].get_xdata(),
							'y': qqplot_data[0].get_ydata(),
							'mode': 'markers',
							'marker': {
								'color': '#1f77b4' # Blue color #1f77b4  # blue
							},
							'name': 'Sample Data Points',
						})

						fig.add_trace({
							'type': 'scatter',
							'x': qqplot_data[1].get_xdata(),
							'y': qqplot_data[1].get_ydata(),
							'mode': 'lines',
							'line': {
								'color': 'red' # Bright green
							},
							'name': 'Theoretical Line',
						})

						# Add reference line (identity line)
						x_min = min(qqplot_data[0].get_xdata())
						x_max = max(qqplot_data[0].get_xdata())
						y_min = min(qqplot_data[0].get_ydata())
						y_max = max(qqplot_data[0].get_ydata())
						fig.add_trace(go.Scatter(x=[x_min, x_max], y=[y_min, y_max], mode='lines', line=dict(color='#00ff00', width=2), name='Identity Line'))

						fig.update_layout(
							title=f'QQ Plot for {column}',
							xaxis=dict(title='Theoretical Quantiles', zeroline=False),
							yaxis=dict(title='Sample Quantiles'),
							showlegend=True,  # Now shows the legend
							width=800,
							height=700
						)

						st.plotly_chart(fig)
		# st.text("hi")
		# et = time.time()
		# tt = et - s
		# st.text(tt)
			# st.write("hehe")
	else:
		st.write("DataFrame not found.")

		# st.write("``")

except ZeroDivisionError as ee:
	print("ee", ee)
	pass
except AttributeError as eee:
	print(eee)
	pass
except Exception as e:
	# handle_error(e)
	st.error(e)
	st.subheader("⚠️Please uploade File⚠️")
	print("f", e)
	pass




