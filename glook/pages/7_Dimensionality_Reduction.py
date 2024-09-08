import streamlit as st
from sklearn.decomposition import PCA, NMF, FastICA, FactorAnalysis, DictionaryLearning, TruncatedSVD
# from sklearn.metrics import silhouette_score, davies_bouldin_score, explained_variance_score
from sklearn.metrics import explained_variance_score
import pandas as pd

st.write(":gray[This is just for demonstration purposes and will not be used in supervised learning yet.]")

try:
	# st.write("Session State:->", st.session_state["shared"])
	# st.write(":grey[This is just for demonstration ]")
	# Streamlit UI for data splitting
	# st.title("Data Splitting Page")

	# Display the modified DataFrame
	# st.subheader("Modified DataFrame")
	if "df_pre" in st.session_state:
		# df = st.session_state.df
		df_to_pre = st.session_state.df_pre
		df = df_to_pre
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

# Slider for selecting the number of components
n_comp = st.slider('Select the number of components:', min_value=1, max_value=int(min(df.shape)), value=2)

def decomposition(data, n_comp, model_name):
	if model_name == "PCA":
		method = PCA(n_components=n_comp)
	elif model_name == "NMF":
		method = NMF(n_components=n_comp)
	elif model_name == "FastICA":
		method = FastICA(n_components=n_comp)
	elif model_name == "FactorAnalysis":
		method = FactorAnalysis(n_components=n_comp)
	elif model_name == "DictionaryLearning":
		method = DictionaryLearning(n_components=n_comp)
	elif model_name == "TruncatedSVD":
		method = TruncatedSVD(n_components=n_comp)
	else:
		raise ValueError("Invalid model_name argument. Use 'PCA' or 'NMF'.")
	# method = model1(n_components=n_comp)
	transformed_data = method.fit_transform(data)
	
	# Calculate explained variance score
	variance_explained = explained_variance_score(data, method.inverse_transform(transformed_data))
	
	return method, transformed_data, variance_explained


methods = {
	"PCA": PCA(),
	"NMF": NMF(),
	"FastICA": FastICA(),
	"FactorAnalysis": FactorAnalysis(),
	"DictionaryLearning": DictionaryLearning(),
	"TruncatedSVD": TruncatedSVD()
}

# if selected_type == "All Models":
selected_models = st.multiselect("Select models to train", list(methods.keys()))

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
# Train selected models in parallel
for i in range(num_columns):
	with columns[i]:
		if i < len(selected_models):
			model_name = selected_models[i]
			# st.subheader(model_name)
			# model_instance = models[model_name]
			model_function = methods[model_name]
			if len(selected_models) <= 5 and len(str(model_function)) <= 30:
				st.write(model_function)
				# st.write(len(str(model_function)))
			else:
				st.subheader(f":violet[{model_name}]", divider=True)
				
			try:
				method, transformed_data, variance_explained = decomposition(df, n_comp, model_name)

				# print(method); print(transformed_data); print(variance_explained)
				st.write("explained_variance_score:")
				st.subheader(f":green[{variance_explained:.3}]")
				# Scatter plot
				# st.subheader("Scatter plot of Transformed Data")
				tdf = pd.DataFrame(transformed_data, columns=[f"Component {i+1}" for i in range(n_comp)])
				st.write(tdf)
			except Exception as e:
				# st.error(e)
				if "Negative values in data passed to NMF" in str(e):
					st.warning("Please use Min-Max Scaling to use other Dimension Reduction Techniques.")
				else:
					st.error(e)
			# st.scatter_chart(transformed_data)  # Scatter plot of the first two components
			# if method is not None:
			# 	print(method)
			# if transformed_data is not None:
			# 	print(transformed_data)
			# if silhouette is not None:
			# 	print(silhouette)
			# if davies_bouldin is not None:
			# 	print(davies_bouldin)
			# if variance_explained is not None:
			# 	print(variance_explained)

			# except:
				# pass

				
			# confirm_change = st.button(
			# 	f'Confirm Change with {model_name}', 
			# 	use_container_width=True
			# 	)
			# if confirm_change:
			# 	st.session_state.df = tdf
			# 	# st.switch_page("pages/7_Model_Building.py")
			# 	# st.switch_page("pages/8_Supervised_Learning.py")
			# 	st.rerun()

			