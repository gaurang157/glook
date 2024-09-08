import streamlit as st
from sklearn.cluster import (
	KMeans, DBSCAN, AgglomerativeClustering, MeanShift, Birch,
	AffinityPropagation, SpectralClustering, OPTICS, 
	MiniBatchKMeans, FeatureAgglomeration, HDBSCAN
)
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA, NMF, FastICA, FactorAnalysis, DictionaryLearning, TruncatedSVD
from sklearn.metrics import (
	silhouette_score, davies_bouldin_score, calinski_harabasz_score,
	adjusted_rand_score, adjusted_mutual_info_score, completeness_score,
	homogeneity_score, v_measure_score
)
import numpy as np
from sklearn.metrics import explained_variance_score
import joblib
from sklearn.model_selection import train_test_split
try:
	st.write("Session State:->", st.session_state["shared"])
	# Streamlit UI for data splitting
	# st.title("Data Splitting Page")

	# Display the modified DataFrame
	# st.subheader("Modified DataFrame")
	if "df_pre" in st.session_state:
		# df = st.session_state.df
		df_to_pre = st.session_state.df_pre
		df = df_to_pre
		
		# Splitting the data into training and validation sets
		df, validation_df = train_test_split(df, test_size=0.1, random_state=42)
		st.session_state.uns_valid = validation_df
		# Assuming df is your DataFrame
		# df.to_csv('your_file.csv', index=False)
		# Data split options
		# st.write(df)
		# target_column = st.sidebar.selectbox("Select the target column:", df.columns)
		# test_size = st.sidebar.slider("Select the test size:", 0.1, 0.5, step=0.05)
		# random_state = st.sidebar.number_input("Enter the random state:", min_value=0, max_value=10000, value=42)
	else:
		pass
except:
	pass

# Available clustering algorithms
models = {
	"K-Means": KMeans(),
	"DBSCAN": DBSCAN(),
	"HDBSCAN": HDBSCAN(),
	"Agglomerative Clustering": AgglomerativeClustering(),
	"Mean Shift": MeanShift(),
	"Birch": Birch(),
	"Affinity Propagation": AffinityPropagation(),
	"Spectral Clustering": SpectralClustering(),
	"OPTICS": OPTICS(),
	"Mini Batch K-Means": MiniBatchKMeans(),
	"Feature Agglomeration": FeatureAgglomeration(),
	# Add more clustering algorithms as needed
}
methods = {
	"PCA": PCA(),
	"NMF": NMF(),
	"FastICA": FastICA(),
	"FactorAnalysis": FactorAnalysis(),
	"DictionaryLearning": DictionaryLearning(),
	"TruncatedSVD": TruncatedSVD()
}



# # Get the maximum number of clusters
# max_clusters = len(df) - 1

# # Slider for selecting the number of clusters
# n_clusters = st.slider('Select the number of clusters:', min_value=2, max_value=int(max_clusters), value=2)
# Get the maximum number of clusters
max_clusters = len(df) - 1

# # Text input for selecting the number of clusters
# n_clusters = st.number_input('Enter the number of clusters:', min_value=2, max_value=int(max_clusters), value=2, step=1)
selected_type = st.radio("Select Model Type", ["Clustering", "Clustering by Decomposition"], captions=["Normal Clustering", "First Decomposition then Clustering"])

if selected_type == "Clustering":
	# Text input for selecting the number of clusters
	n_clusters = st.number_input('Enter the number of clusters:', min_value=2, max_value=int(max_clusters), value=2, step=1)
	selected_models = st.multiselect("Select models to train", list(models.keys()))
elif selected_type == "Clustering by Decomposition":
	# Slider for selecting the number of components
	n_comp = st.slider('Select the number of Decomposition components:', min_value=1, max_value=int(min(df.shape)), value=2)
	# Text input for selecting the number of clusters
	n_clusters = st.number_input('Enter the number of clusters:', min_value=2, max_value=int(max_clusters), value=2, step=1)
	selected_method = st.selectbox("Select Decomposition method", list(methods.keys()))
	selected_models = st.multiselect("Select Clustering models to train", list(models.keys()))
def train_clustering_model_with_decomposition(data, model_name, n_clusters=None, selected_method=None, n_comp=None):
	if selected_method == "PCA":
		method = PCA(n_components=n_comp)
	elif selected_method == "NMF":
		method = NMF(n_components=n_comp)
	elif selected_method == "FastICA":
		method = NMF(n_components=n_comp)
	elif selected_method == "FactorAnalysis":
		method = NMF(n_components=n_comp)
	elif selected_method == "DictionaryLearning":
		method = NMF(n_components=n_comp)
	elif selected_method == "TruncatedSVD":
		method = NMF(n_components=n_comp)
	else:
		raise ValueError("Invalid model_name argument. Use 'PCA' or 'NMF'.")
	# method = model1(n_components=n_comp)
	transformed_data = method.fit_transform(data)
	
	# Calculate explained variance score
	variance_explained = explained_variance_score(data, method.inverse_transform(transformed_data))

	if model_name == "K-Means":
		model = KMeans(n_clusters=n_clusters)
		
	elif model_name == "DBSCAN":
		model = DBSCAN()
		
	elif model_name == "Agglomerative Clustering":
		model = AgglomerativeClustering(n_clusters=n_clusters)
		
	elif model_name == "Mean Shift":
		model = MeanShift()
		
	elif model_name == "Birch":
		model = Birch(n_clusters=n_clusters)
		
	elif model_name == "Affinity Propagation":
		model = AffinityPropagation()
		
	elif model_name == "Spectral Clustering":
		model = SpectralClustering(n_clusters=n_clusters)
		
	elif model_name == "OPTICS":
		model = OPTICS()
		
	elif model_name == "Mini Batch K-Means":
		model = MiniBatchKMeans(n_clusters=n_clusters)
		
	elif model_name == "Feature Agglomeration":
		model = FeatureAgglomeration(n_clusters=n_clusters)
		
	elif model_name == "HDBSCAN":
		model = HDBSCAN()
		
	# elif model_name == "Estimate Bandwidth":
	#     bandwidth = estimate_bandwidth(data)
	#     model = MeanShift(bandwidth=bandwidth)
	else:
		raise ValueError("Invalid model_name. Please choose from available clustering algorithms.")

	# Fit the clustering model
	model.fit(transformed_data)
	# labels = model.labels_
	# Calculate silhouette score (if applicable)
	if hasattr(model, 'labels_') and len(set(model.labels_)) > 1:
		silhouette = silhouette_score(transformed_data, model.labels_)
	else:
		silhouette = None
	
	# Calculate Davies-Bouldin score (if applicable)
	if hasattr(model, 'labels_') and len(set(model.labels_)) > 1:
		davies_bouldin = davies_bouldin_score(transformed_data, model.labels_)
	else:
		davies_bouldin = None
	
	# Calculate additional evaluation metrics
	if n_clusters is not None:
		calinski_harabasz = calinski_harabasz_score(transformed_data, model.labels_)

	else:
		calinski_harabasz = None

	
	# Format evaluation metrics
	metrics = {
		'silhouette': float("{:.3f}".format(silhouette)) if silhouette is not None else None,
		'davies_bouldin': float("{:.3f}".format(davies_bouldin)) if davies_bouldin is not None else None,
		'calinski_harabasz': float("{:.3f}".format(calinski_harabasz)) if calinski_harabasz is not None else None,
	}
	
	return model, metrics, variance_explained, transformed_data


def train_clustering_model(data, model_name, n_clusters=None,):
	if model_name == "K-Means":
		model = KMeans(n_clusters=n_clusters)
		
	elif model_name == "DBSCAN":
		model = DBSCAN()
		
	elif model_name == "Agglomerative Clustering":
		model = AgglomerativeClustering(n_clusters=n_clusters)
		
	elif model_name == "Mean Shift":
		model = MeanShift()
		
	elif model_name == "Birch":
		model = Birch(n_clusters=n_clusters)
		
	elif model_name == "Affinity Propagation":
		model = AffinityPropagation()
		
	elif model_name == "Spectral Clustering":
		model = SpectralClustering(n_clusters=n_clusters)
		
	elif model_name == "OPTICS":
		model = OPTICS()
		
	elif model_name == "Mini Batch K-Means":
		model = MiniBatchKMeans(n_clusters=n_clusters)
		
	elif model_name == "Feature Agglomeration":
		model = FeatureAgglomeration(n_clusters=n_clusters)
		
	elif model_name == "HDBSCAN":
		model = HDBSCAN()
		
	# elif model_name == "Estimate Bandwidth":
	#     bandwidth = estimate_bandwidth(data)
	#     model = MeanShift(bandwidth=bandwidth)
	else:
		raise ValueError("Invalid model_name. Please choose from available clustering algorithms.")

	# Fit the clustering model
	model.fit(data)
	# labels = model.labels_
	
	# Calculate silhouette score (if applicable)
	if hasattr(model, 'labels_') and len(set(model.labels_)) > 1:
		silhouette = silhouette_score(data, model.labels_)
	else:
		silhouette = None
	
	# Calculate Davies-Bouldin score (if applicable)
	if hasattr(model, 'labels_') and len(set(model.labels_)) > 1:
		davies_bouldin = davies_bouldin_score(data, model.labels_)
	else:
		davies_bouldin = None
	
	# Calculate additional evaluation metrics
	if n_clusters is not None:
		calinski_harabasz = calinski_harabasz_score(data, model.labels_)

	else:
		calinski_harabasz = None

	
	# Format evaluation metrics
	metrics = {
		'silhouette': float("{:.3f}".format(silhouette)) if silhouette is not None else None,
		'davies_bouldin': float("{:.3f}".format(davies_bouldin)) if davies_bouldin is not None else None,
		'calinski_harabasz': float("{:.3f}".format(calinski_harabasz)) if calinski_harabasz is not None else None,
	}
	
	return model, metrics

st.sidebar.write(":blue[After training the models, click below to proceed ⤵️]")
if st.sidebar.button("Unsupervised_Deployment_Demo"):	
	st.switch_page(r"pages/13_Unsupervised_Deployment_Demo.py")

if st.sidebar.button("Custom_Model_Training"):
	st.switch_page(r"pages/11_Custom_Model_Training.py")


all_metrics_data = []
# Determine the number of columns based on the number of selected models
num_columns = min(len(selected_models), 10)
if num_columns <= 0:
	num_columns = 1  # Set a default value of 1 if num_columns is 0 or negative
try:
	columns = st.columns(num_columns)
except NameError:
	st.write("Select Model")
	pass
# Create an empty DataFrame
# all_metrics_data = pd.DataFrame(columns=['Model', 'Silhouette Score', 'Davies-Bouldin Score', 'Calinski-Harabasz Score'])
# Train selected models in parallel
for i in range(num_columns):
	with columns[i]:
		if i < len(selected_models):
			model_name = selected_models[i]
			try:
				method_name = selected_method
			except:
				pass
			# st.subheader(model_name)
			# model_instance = models[model_name]
			model_function = models[model_name]
			if len(selected_models) <= 5 and len(str(model_function)) <= 30:
				st.write(model_function)
				# st.write(len(str(model_function)))
			else:
				st.subheader(f":violet[{model_name}]", divider=True)
			try:
				if selected_type == "Clustering":
					model, metrics = train_clustering_model(df, model_name, n_clusters)
					try:
						# Append model name and metrics to the list
						all_metrics_data.append({
							'Model': model_name,
							'Silhouette Score': metrics['silhouette'],
							'Davies-Bouldin Score': metrics['davies_bouldin'],
							'Calinski-Harabasz Score': metrics['calinski_harabasz']
							})
					except:
						pass
					st.write(metrics)
					st.write("Lables:⬇️")
					distinct_labels = np.unique(model.labels_)
					st.write(distinct_labels)
					model_info = {
						'model': model,  # The trained model
						'metrics': metrics,
						'model_name': model_name,  # Name of the model
						'labels': model.labels_,
						'distinct_labels': distinct_labels,
						'selected_type': selected_type
					}

					# Dump to a joblib file
					model_filename = f"{model_name}_clust_model.pkl"
					model_filename = model_filename.replace(" ", "")
					# Save the dictionary containing the model and its metrics
					with open(model_filename, "wb") as f:
						joblib.dump(model_info, f)
					# with st.expander("Model Evalution:"):
					if df.shape[1] == 3:
						try:
							# Select three columns for trivariate analysis
							x_column = df.columns[0]
							y_column = df.columns[1]
							z_column = df.columns[2]

							# Create a DataFrame with the selected columns and cluster labels
							trivariate_data = df[[x_column, y_column, z_column]]
							trivariate_data['Cluster'] = model.labels_
							# Check Cluster Labels
							# print("Cluster Labels:", model.labels_)

							# Get distinct labels
							distinct_labels = np.unique(model.labels_)
							# print("Distinct Labels:", distinct_labels)
							# Convert cluster labels to numeric if needed
							trivariate_data['Cluster'] = trivariate_data['Cluster'].astype(int)
							# Get distinct labels
							distinct_labels = np.unique(model.labels_)
							# Create the 3D scatter plot
							fig = go.Figure(data=[go.Scatter3d(
								x=trivariate_data[x_column],
								y=trivariate_data[y_column],
								z=trivariate_data[z_column],
								mode='markers',
								marker=dict(
									size=12,
									color=trivariate_data['Cluster'],  # Using cluster labels as colors
									colorscale='Viridis',  # You can choose any colorscale you prefer
									opacity=0.7,
									colorbar=dict(title='Cluster')  # Add color bar with title
								)
							)])

							# Update layout
							fig.update_layout(title=f'Trivariate Analysis with Cluster Color for {model_name}', scene=dict(
												xaxis_title=x_column,
												yaxis_title=y_column,
												zaxis_title=z_column))
							with st.expander("3D"):
								try:
									st.plotly_chart(fig,use_container_width=True)
								except:
									pass
							# Show the plot
							# st.plotly_chart(fig)
							# Button to show the plot in full-screen mode
							if st.button(f"{model_name} 3-D Plot Full Screen"):
								fig.show()
						except:
							pass
					
					# st.write(metrics)
				elif selected_type == "Clustering by Decomposition":
					model, metrics, variance_explained, transformed_data = train_clustering_model_with_decomposition(df, model_name, n_clusters, method_name, n_comp)
					# st.write(metrics)
					# st.write("variance_explained:")
					# st.write(f":green[{variance_explained:.3}]")

			
				# if selected_type == "Clustering":
					
				# elif selected_type == "Clustering by Decomposition":
					try:
						# Append model name and metrics to the list
						all_metrics_data.append({
							'Model': model_name,
							'Decomposition Method': method_name,
							'Silhouette Score': metrics['silhouette'],
							'Davies-Bouldin Score': metrics['davies_bouldin'],
							'Calinski-Harabasz Score': metrics['calinski_harabasz'],
							'Decomposition Explained Variance': variance_explained
							})
					except:
						pass
					st.write(metrics)
					st.write("variance_explained:")
					st.write(f":green[{variance_explained:.3}]")
					st.write("Lables:⬇️")
					distinct_labels = np.unique(model.labels_)
					st.write(distinct_labels)
					model_info = {
						'model': model,  # The trained model
						'metrics': metrics,
						'decom_method': method_name,
						'decomn_comp': n_comp,
						'variance_explained': variance_explained,
						'model_name': model_name,  # Name of the model
						'labels': model.labels_,
						'distinct_labels': distinct_labels,
						'selected_type': selected_type
					}

					# Dump to a joblib file
					model_filename = f"{model_name}_decom_clust_model.pkl"
					model_filename = model_filename.replace(" ", "")
					# Save the dictionary containing the model and its metrics
					with open(model_filename, "wb") as f:
						joblib.dump(model_info, f)
					# st.write(df.shape)
					if transformed_data.shape[1] == 3:
						try:
							# Select three columns for trivariate analysis
							x_column = df.columns[0]
							y_column = df.columns[1]
							z_column = df.columns[2]

							# Create a DataFrame with the selected columns and cluster labels
							trivariate_data = df[[x_column, y_column, z_column]]
							trivariate_data['Cluster'] = model.labels_
							# Check Cluster Labels
							# print("Cluster Labels:", model.labels_)

							# Get distinct labels
							distinct_labels = np.unique(model.labels_)
							# print("Distinct Labels:", distinct_labels)
							# Convert cluster labels to numeric if needed
							trivariate_data['Cluster'] = trivariate_data['Cluster'].astype(int)
							# Get distinct labels
							distinct_labels = np.unique(model.labels_)
							# Create the 3D scatter plot
							fig = go.Figure(data=[go.Scatter3d(
								x=trivariate_data[x_column],
								y=trivariate_data[y_column],
								z=trivariate_data[z_column],
								mode='markers',
								marker=dict(
									size=12,
									color=trivariate_data['Cluster'],  # Using cluster labels as colors
									colorscale='Viridis',  # You can choose any colorscale you prefer
									opacity=0.7,
									colorbar=dict(title='Cluster')  # Add color bar with title
								)
							)])

							# Update layout
							fig.update_layout(title=f'Trivariate Analysis with Cluster Color for {model_name}', scene=dict(
												xaxis_title=x_column,
												yaxis_title=y_column,
												zaxis_title=z_column))

							# Show the plot
							# st.plotly_chart(fig)
							with st.expander("3D"):
								try:
									st.plotly_chart(fig,use_container_width=True)
								except:
									pass
							# Button to show the plot in full-screen mode
							if st.button(f"{model_name} 3-D Plot Full Screen"):
								fig.show()
						except Exception as e:
							st.warning(e)
							pass

			except Exception as e:
				st.error(e)



# Create a DataFrame from the list of dictionaries
# all_metrics_data = pd.DataFrame(all_metrics_data)

# # Display the DataFrame
# # st.write(all_metrics_data)

# # Extract values for plotting
# model_names = all_metrics_data['Model']
# silhouette_scores = all_metrics_data['Silhouette Score']
# davies_bouldin_scores = all_metrics_data['Davies-Bouldin Score']
# calinski_harabasz_scores = all_metrics_data['Calinski-Harabasz Score']

# try:
if selected_type == "Clustering":
	try:
		# st.title('Clustering Model Comparison')
		col1, col2 = st.columns(2)
		# Create a DataFrame from the list of dictionaries
		all_metrics_df = pd.DataFrame(all_metrics_data)


		# Reset the index of the DataFrame
		all_metrics_df.reset_index(drop=True, inplace=True)


		import plotly.graph_objects as go
		import plotly.express as px
		# # Scatter Plot Matrix
		# fig_scatter_matrix = px.scatter_matrix(all_metrics_df, dimensions=['Accuracy', 'ROC AUC', 'F1 Score', 'Recall', 'Precision', 'RMSE', 'R2 Score'], 
		#                                        color='Model', title='Scatter Plot Matrix - Performance Metrics')
		# st.plotly_chart(fig_scatter_matrix)


		# Set the index of the DataFrame to 'Model' column for easier plotting
		# all_metrics_df.reset_index(inplace=True)
		with col1:
			# Bar plot
			fig_bar = px.bar(all_metrics_df, x='Model', y=['Silhouette Score', 'Davies-Bouldin Score', 'Calinski-Harabasz Score'], 
							barmode='group', title='Model Performance Metrics')
			fig_bar.update_layout(xaxis_title='Model', yaxis_title='Metric Value')
			st.plotly_chart(fig_bar)
			
			# Violin Plot
			fig_violin = go.Figure()
			for metric in ['Silhouette Score', 'Davies-Bouldin Score', 'Calinski-Harabasz Score']:
				fig_violin.add_trace(go.Violin(y=all_metrics_df[metric], x=all_metrics_df['Model'], name=metric, box_visible=True, meanline_visible=True))
			fig_violin.update_layout(title='Violin Plot - Performance Metrics', xaxis_title='Model', yaxis_title='Metric Value')
			st.plotly_chart(fig_violin)
			# Box plot
			fig_box = px.box(all_metrics_df.melt(id_vars='Model', var_name='Metric', value_name='Value'), x='Metric', y='Value', color='Model', 
							title='Model Performance Metrics')
			fig_box.update_layout(xaxis_title='Metric', yaxis_title='Metric Value')
			st.plotly_chart(fig_box)
			

			

		with col2:
			# Radar Chart
			fig_radar = go.Figure()
			for index, row in all_metrics_df.iterrows():
				fig_radar.add_trace(go.Scatterpolar(r=row[['Silhouette Score', 'Davies-Bouldin Score', 'Calinski-Harabasz Score']].values,
													theta=['Silhouette Score', 'Davies-Bouldin Score', 'Calinski-Harabasz Score'],
													fill='toself', name=row['Model']))
			fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), title='Radar Chart - Performance Metrics')
			st.plotly_chart(fig_radar)
			
			# Line plot
			fig_line = px.line(all_metrics_df, x='Model', y=['Silhouette Score', 'Davies-Bouldin Score', 'Calinski-Harabasz Score'], 
							title='Model Performance Metrics')
			fig_line.update_layout(xaxis_title='Model', yaxis_title='Metric Value')
			st.plotly_chart(fig_line)
			# Display the DataFrame with all the metrics
			st.write("All Model Metrics:")
			st.write(all_metrics_df)
	except:
		pass
elif selected_type == "Clustering by Decomposition":
	try:
		# st.title('Clustering Model Comparison')
		col1, col2 = st.columns(2)
		# Create a DataFrame from the list of dictionaries
		all_metrics_df = pd.DataFrame(all_metrics_data)


		# Reset the index of the DataFrame
		all_metrics_df.reset_index(drop=True, inplace=True)


		import plotly.graph_objects as go
		import plotly.express as px
		# # Scatter Plot Matrix
		# fig_scatter_matrix = px.scatter_matrix(all_metrics_df, dimensions=['Accuracy', 'ROC AUC', 'F1 Score', 'Recall', 'Precision', 'RMSE', 'R2 Score'], 
		#                                        color='Model', title='Scatter Plot Matrix - Performance Metrics')
		# st.plotly_chart(fig_scatter_matrix)

		# Set the index of the DataFrame to 'Model' column for easier plotting
		# all_metrics_df.reset_index(inplace=True)
		with col1:
			# Bar plot
			fig_bar = px.bar(all_metrics_df, x='Model', y=['Silhouette Score', 'Davies-Bouldin Score', 'Calinski-Harabasz Score', 'Decomposition Explained Variance'], 
							barmode='group', title='Model Performance Metrics')
			fig_bar.update_layout(xaxis_title='Model', yaxis_title='Metric Value')
			st.plotly_chart(fig_bar)
			
			# Violin Plot
			fig_violin = go.Figure()
			for metric in ['Silhouette Score', 'Davies-Bouldin Score', 'Calinski-Harabasz Score', 'Decomposition Explained Variance']:
				fig_violin.add_trace(go.Violin(y=all_metrics_df[metric], x=all_metrics_df['Model'], name=metric, box_visible=True, meanline_visible=True))
			fig_violin.update_layout(title='Violin Plot - Performance Metrics', xaxis_title='Model', yaxis_title='Metric Value')
			st.plotly_chart(fig_violin)
			# Box plot
			fig_box = px.box(all_metrics_df.melt(id_vars='Model', var_name='Metric', value_name='Value'), x='Metric', y='Value', color='Model', 
							title='Model Performance Metrics')
			fig_box.update_layout(xaxis_title='Metric', yaxis_title='Metric Value')
			st.plotly_chart(fig_box)
			

			

		with col2:
			# Radar Chart
			fig_radar = go.Figure()
			for index, row in all_metrics_df.iterrows():
				fig_radar.add_trace(go.Scatterpolar(r=row[['Silhouette Score', 'Davies-Bouldin Score', 'Calinski-Harabasz Score', 'Decomposition Explained Variance']].values,
													theta=['Silhouette Score', 'Davies-Bouldin Score', 'Calinski-Harabasz Score', 'Decomposition Explained Variance'],
													fill='toself', name=row['Model']))
			fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), title='Radar Chart - Performance Metrics')
			st.plotly_chart(fig_radar)
			
			# Line plot
			fig_line = px.line(all_metrics_df, x='Model', y=['Silhouette Score', 'Davies-Bouldin Score', 'Calinski-Harabasz Score', 'Decomposition Explained Variance'], 
							title='Model Performance Metrics')
			fig_line.update_layout(xaxis_title='Model', yaxis_title='Metric Value')
			st.plotly_chart(fig_line)
			# Display the DataFrame with all the metrics
			st.write("All Model Metrics:")
			st.write(all_metrics_df)
	except:
		pass


# # Create a bar plot for silhouette scores
# fig = go.Figure()
# fig.add_trace(go.Bar(x=model_names, y=silhouette_scores, name='Silhouette Score'))
# fig.add_trace(go.Bar(x=model_names, y=davies_bouldin_scores, name='Davies-Bouldin Score'))
# fig.add_trace(go.Bar(x=model_names, y=calinski_harabasz_scores, name='Calinski-Harabasz Score'))
# fig.update_layout(barmode='group', title='Clustering Model Comparison', xaxis_title='Clustering Models', yaxis_title='Score')
# st.plotly_chart(fig, use_container_width=True)

# import streamlit as st
# import pandas as pd
# import numpy as np


# # Assume all_metrics_data is the DataFrame containing the clustering model comparison results

# # 1. Radar Chart
# def radar_chart(all_metrics_data):
#     fig = go.Figure()
#     for model_name in all_metrics_data['Model'].unique():
#         model_metrics = all_metrics_data[all_metrics_data['Model'] == model_name].drop(columns=['Model'])
#         fig.add_trace(go.Scatterpolar(
#             r=model_metrics.values.flatten().tolist(),
#             theta=model_metrics.columns.tolist(),
#             fill='toself',
#             name=model_name
#         ))
#     fig.update_layout(
#         polar=dict(
#             radialaxis=dict(
#                 visible=True,
#                 range=[0, 1]  # Adjust range as needed
#             )
#         ),
#         showlegend=True,
#         title='Radar Chart of Clustering Model Comparison'
#     )
#     st.plotly_chart(fig)

# # 2. Parallel Coordinates Plot
# def parallel_coordinates_plot(all_metrics_data):
#     # Create a categorical colormap for the models
#     color_map = {model: f'#{i*10:06x}' for i, model in enumerate(all_metrics_data['Model'].unique())}
#     all_metrics_data['Color'] = all_metrics_data['Model'].map(color_map)
	
#     # Plot parallel coordinates
#     fig = px.parallel_coordinates(all_metrics_data, color='Color', labels={'index': 'Metric'})
#     fig.update_layout(title='Parallel Coordinates Plot of Clustering Model Comparison')
#     st.plotly_chart(fig)


# # 3. Violin Plot
# def violin_plot(all_metrics_data):
#     fig = px.violin(all_metrics_data.melt(id_vars='Model', var_name='Metric', value_name='Value'), x='Metric', y='Value', color='Model', box=True, points="all")
#     fig.update_layout(title='Violin Plot of Clustering Model Comparison')
#     st.plotly_chart(fig)





# # Plotting
# st.title('Clustering Model Comparison')

# # Assuming all_metrics_data contains the DataFrame with model comparison results
# # Assuming all_metrics_data has 'Model' column as the model names and other columns as evaluation metrics

# # Call the plot functions
# radar_chart(all_metrics_data)
# parallel_coordinates_plot(all_metrics_data)
# violin_plot(all_metrics_data)
