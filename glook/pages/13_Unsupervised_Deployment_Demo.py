import streamlit as st
import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import glob
import seaborn as sns
# st.write(os.getcwd())
import numpy as np
from scipy.stats.mstats import winsorize
from scipy.stats import boxcox, yeojohnson
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA, NMF, FastICA, FactorAnalysis, DictionaryLearning, TruncatedSVD
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

if "X_test" in st.session_state or "df" in st.session_state:
	try:
		X_test0 = st.session_state.X_test0
	except:
		pass
	try:
		X_test = st.session_state.X_test
	except:
		pass
	try:
		df = st.session_state.df
	except:
		pass
	# try:
	# 	df0 = st.session_state.df0
	# except:
	# 	pass
	try:
		df0 = st.session_state.uns_valid
	except:
		pass
try:
	dfo = X_test0
except:
	pass
# print(X_test0)


def apply_pipeline_d_tpye(df, pipeline):
	try:
		# Make a copy of the DataFrame to avoid warnings or unintentional changes
		modified_df = df.copy()

		column_name = pipeline["column"]
		new_dtype = pipeline["new_dtype"]

		# Apply the new data type using .astype() without inplace
		modified_df[column_name] = modified_df[column_name].astype(new_dtype)

		return modified_df
	
	except Exception as e:
		print(f"Error applying pipeline:1 {e}")
		return df  # Return the original DataFrame if there's an error


def apply_pipeline_drop(df, pipeline):
	try:
		# Make a copy of the DataFrame to avoid warnings or unintentional changes
		modified_df = df.copy()

		column_name = pipeline["column"]

		# Apply the new data type using .astype() without inplace
		modified_df.drop(column_name, axis=1, inplace=True)
		return modified_df
	
	except Exception as e:
		print(f"Error applying pipeline:2 {e}")
		return df  # Return the original DataFrame if there's an error


# Function to apply a pipeline that drops duplicates
def apply_pipeline_drop_dup(df, pipeline):
	try:
		if pipeline["action"] == "drop_duplicates":
			# Drop duplicates and return the modified DataFrame
			df = df.drop_duplicates()
			return df
	except Exception as e:
		print(f"Error applying pipeline: 3{e}")
		return df  # Return the original DataFrame if there's an error


def apply_outliers_for_col(df, pipeline):
	method = pipeline['method']
	column_name = pipeline['column']
	lower_limit = pipeline['lower_limit']
	upper_limit = pipeline['upper_limit']
	z_score_threshold = pipeline['z_score_threshold']
	dff = df.copy()
	try:
		if method == "Delete Outliers":
			# Define your outlier detection and deletion method here
			# For example, you can use z-score method to detect and delete outliers
			z_scores = np.abs((dff[column_name] - dff[column_name].mean()) / dff[column_name].std())
			dff = dff[z_scores < float(z_score_threshold)]  # Remove rows with z-scores greater than 3
			
			# st.success("Outliers deleted successfully.")
			return dff
		elif method == "Winsorization":

			# Apply Winsorization to the column
			modified_column = winsorize(dff[column_name], limits=[float(lower_limit), float(upper_limit)])
			dff[column_name] = modified_column
			return dff
	except Exception as e:
		print("e4:=",e)
		return df
	 
def apply_outliers_full_df(df, pipeline):
	method = pipeline['method']
	lower_limit = pipeline['lower_limit']
	upper_limit = pipeline['upper_limit']
	z_score_threshold = pipeline['z_score_threshold']
	try:
		# Get a list of all numerical columns
		numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()

		if method == "Delete Outliers":
			# Remove rows with z-scores greater than the specified threshold for each numerical column
			for column_name in numerical_columns:
				z_scores = np.abs((df[column_name] - df[column_name].mean()) / df[column_name].std())
				df = df[z_scores < float(z_score_threshold)]  # Retain rows with z-scores less than the threshold
			# st.success("Outliers deleted successfully.")
			return df

		elif method == "Winsorization":
			# Apply Winsorization to each numerical column with the specified limits
			for column_name in numerical_columns:
				df[column_name] = winsorize(df[column_name], limits=[float(lower_limit), float(upper_limit)])
			# st.success("Outliers treated using Winsorization successfully.")
			return df

		else:
			return df

	except Exception as e:
		print("e5",e)
		return df

def treat_missing_vaues_for_col(df, pipeline):
	method = pipeline['method']
	column_name = pipeline['column']
	dff = df.copy()
	try:
		if method == "Delete Missing Values":
			dff = dff.dropna(subset=[column_name])
			# st.success("Missing values deleted successfully.")
			return dff
		elif method == "Mean Imputation":
			imputer = SimpleImputer(strategy="mean")
			dff[column_name] = imputer.fit_transform(dff[[column_name]])
			# st.success("Missing values imputed using mean successfully.")
			return dff
		elif method == "Median Imputation":
			imputer = SimpleImputer(strategy="median")
			dff[column_name] = imputer.fit_transform(dff[[column_name]])
			# st.success("Missing values imputed using median successfully.")
			return dff
		elif method == "Mode Imputation":
			imputer = SimpleImputer(strategy="most_frequent")
			dff[column_name] = imputer.fit_transform(dff[[column_name]])
			# st.success("Missing values imputed using mode successfully.")
			return dff
	except Exception as e:
		print("e6",e)
		return df
	 

def treat_missing_values_in_full_df(df, pipeline):
	# numeric_treatment="Mean", numeric_strategy=None, categorical_treatment="Mode", categorical_strategy=None
	numeric_treatment = pipeline['numeric_treatment']
	numeric_strategy = pipeline['numeric_strategy']
	categorical_treatment = pipeline['categorical_treatment']
	categorical_strategy = pipeline['categorical_strategy']

	# Get numeric and categorical columns
	numeric_columns = df.select_dtypes(include=np.number).columns
	categorical_columns = df.select_dtypes(include=['object', 'category']).columns
	
	# Copy the DataFrame to avoid modifying the original data
	modified_df = df.copy()
	
	# Apply treatment for numeric columns
	for column in numeric_columns:
		if numeric_treatment == "Mean":
			modified_df[column].fillna(modified_df[column].mean(), inplace=True)
		elif numeric_treatment == "Median":
			modified_df[column].fillna(modified_df[column].median(), inplace=True)
		elif numeric_treatment == "Mode":
			modified_df[column].fillna(modified_df[column].mode()[0], inplace=True)
		elif numeric_treatment == "Random":
			if numeric_strategy == 'uniform':
				# Assuming a uniform distribution based on min and max values
				min_val = modified_df[column].min()
				max_val = modified_df[column].max()
				modified_df[column].fillna(np.random.uniform(min_val, max_val), inplace=True)
			elif numeric_strategy == 'normal':
				# Assuming a normal distribution with mean and std deviation
				mean = modified_df[column].mean()
				std = modified_df[column].std()
				modified_df[column].fillna(np.random.normal(mean, std), inplace=True)
			else:
				# Default random fill with a choice from existing non-null values
				modified_df[column].fillna(np.random.choice(modified_df[column].dropna().values), inplace=True)

	# Apply treatment for categorical columns
	for column in categorical_columns:
		if categorical_treatment == "Mode":
			modified_df[column].fillna(modified_df[column].mode()[0], inplace=True)
		elif categorical_treatment == "Random":
			if categorical_strategy == 'uniform':
				# Randomly fill with a choice from existing non-null values
				unique_vals = modified_df[column].dropna().unique()
				modified_df[column].fillna(np.random.choice(unique_vals), inplace=True)
			elif categorical_strategy == 'normal':
				# In categorical columns, we can only select from existing unique values
				unique_vals = modified_df[column].dropna().unique()
				modified_df[column].fillna(np.random.choice(unique_vals), inplace=True)
	
	return modified_df

def apply_transformation_on_col(df, pipeline):
	method = pipeline['method']
	column_name = pipeline['column_name']
	try:
		if method == 'Log Transformation':
			df[column_name] = np.log1p(df[column_name])
			return df
		elif method == 'Exponential Transformation':
			df[column_name] = np.exp(df[column_name])
			return df
		elif method == 'Square Root Transformation':
			df[column_name] = np.sqrt(df[column_name])
			return df
		elif method == 'Box-Cox Transformation':
			transformed_data, _ = boxcox(df[column_name] + 1)
			df[column_name] = transformed_data
			return df
		elif method == 'Yeo-Johnson Transformation':
			transformed_data, _ = yeojohnson(df[column_name] + 1)
			df[column_name] = transformed_data
			return df
	except Exception as e:
		print("e7", e)
		return df
	
def apply_encoding_on_df(df, pipeline):
	method = pipeline['method']
	param = pipeline['param']
	try:
		if method == 'One-Hot Encoding':
			cat_columns = df.select_dtypes(include=['object']).columns
			encoded_df = pd.get_dummies(df, columns=cat_columns, drop_first=param)
			return encoded_df
		elif method == 'Label Encoding':
			encoded_df = df.copy()
			label_encoder = LabelEncoder()
			for column in encoded_df.select_dtypes(include=['object']).columns:
				encoded_df[column] = label_encoder.fit_transform(encoded_df[column])
			return encoded_df
	except Exception as e:
		print("e8", e)
		return df
	 
def apply_encoding_on_col(df, pipeline):
	method = pipeline['method']
	column_name = pipeline['column_name']
	param = pipeline['param']
	try:
		if method == 'One-Hot Encoding':
			encoded_df = pd.get_dummies(df[column_name], prefix=column_name, drop_first=param)
			df = pd.concat([df, encoded_df], axis=1)
			df.drop(column_name, axis=1, inplace=True)
			df.replace({(True): 1, (False): 0}, inplace=True)
			return df
		elif method == 'Label Encoding':
			label_encoder = LabelEncoder()
			df[column_name] = label_encoder.fit_transform(df[column_name])
			return df
	except Exception as e:
		print("e9", e)
		return df
	
class LabelEncoderPipelineFriendly(BaseEstimator, TransformerMixin):
	def __init__(self):
		self.le = LabelEncoder()
	
	def fit(self, X, y=None):
		self.le.fit(X)
		return self
	
	def transform(self, X, y=None):
		return self.le.transform(X).reshape(-1, 1)
	
	def fit_transform(self, X, y=None):
		return self.fit(X, y).transform(X)
	
	def inverse_transform(self, X):
		return self.le.inverse_transform(X)

class CustomPipeline(Pipeline):
	def __init__(self, steps, encoding_method=None):
		super().__init__(steps)
		self.encoding_method = encoding_method
	 
def apply_scaling_on_df(df, pipeline):
	method = pipeline['method']
	try:
		if method == 'Standardize Scaling':
			scaler = StandardScaler()
			scaled_data = scaler.fit_transform(df)
			scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
			return scaled_df
		elif method == 'Min-Max Scaling':
			scaler = MinMaxScaler()
			scaled_data = scaler.fit_transform(df)
			scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
			return scaled_df
		elif method == 'Robust Scaling':
			scaler = RobustScaler()
			scaled_data = scaler.fit_transform(df)
			scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
			return scaled_df
	except Exception as e:
		print("e11",e)
		return df
	 
def discretize_output_col(df, pipeline):
	bins = pipeline['bins']
	column_name = pipeline['column_name']
	strategy = pipeline['strategy']
	# column_name, bins, strategy
	# Create a copy to avoid SettingWithCopyWarning

	df_copy = df.copy()
	
	try:
		# Create the discretizer with specified bins and strategy
		discretizer = KBinsDiscretizer(n_bins=int(bins), encode='ordinal', strategy=strategy)
		
		# Use .loc to ensure setting data on a copy
		df_copy.loc[:, column_name] = discretizer.fit_transform(df_copy[[column_name]])
		return df_copy
	except Exception as e:
		print("e12", e)
		return df
	
def column_unique_value_replacement(df, pipeline):
	df_copy = df.copy()
	column_name = pipeline['column_name']
	replacements = pipeline['replacements']
	select = pipeline['select']
  
	df_copy[column_name].replace(replacements, inplace=True)
	if select == 'Convert to int:':
		df_copy[column_name] = df_copy[column_name].astype(pd.Int64Dtype())
	elif select == 'Convert to float:':
		df_copy[column_name] = df_copy[column_name].astype(float)
	return df_copy




cwd = os.getcwd()

# def predict(data, user, pw, db):
def predict(data, model_pipe):
	ttype__ = model_pipe['selected_type']
	model1 = model_pipe['model']
	if ttype__ == "Clustering":
		clean_data = data
		
		model1.fit(clean_data)
		label = model1.labels_
		predictions = pd.DataFrame(label, columns=["Clusters"])

		# Combine predictions with the original data
		final_result = pd.concat([predictions, data], axis=1)
		# final_result.reset_index(inplace=True)

		# final_result = final_result.iloc[:, 1:]
	elif ttype__ == "Clustering by Decomposition":
		clean_data = data
		method_ = model_pipe['decom_method']
		n_comp = model_pipe['decomn_comp']
		if method_ == "PCA":
			decomm_ = PCA(n_components=int(n_comp))
		elif method_ == "NMF":
			decomm_ = NMF(n_components=int(n_comp))
		elif method_ == "FastICA":
			decomm_ = NMF(n_components=int(n_comp))
		elif method_ == "FactorAnalysis":
			decomm_ = NMF(n_components=int(n_comp))
		elif method_ == "DictionaryLearning":
			decomm_ = NMF(n_components=int(n_comp))
		elif method_ == "TruncatedSVD":
			decomm_ = NMF(n_components=int(n_comp))
		else:
			raise ValueError("Invalid model_name argument. Use 'PCA' or 'NMF'.")
		transformed_data = decomm_.fit_transform(clean_data)


		model1.fit(transformed_data)
		label = model1.labels_
		predictions = pd.DataFrame(label, columns=["Clusters"])

		# Combine predictions with the original data
		final_result = pd.concat([predictions, data], axis=1)
			


	return final_result, predictions


def main():  

	# st.title("Prediction")
	st.sidebar.title("Prediction")

	# st.radio('Type of Cab you want to Book', options=['Mini', 'Sedan', 'XL', 'Premium', 'Rental'])
	html_temp = """
	<div style="background-color:tomato;padding:10px">
	<h2 style="color:white;text-align:center;"> Prediction </h2>
	</div>
	
	"""
	st.markdown(html_temp, unsafe_allow_html = True)
	st.text("")
	type_ = st.sidebar.radio("Select Prediction Type:", ["Upload File", "Manual Input"])

	if type_ == 'Upload File':
		choice = st.sidebar.radio("Upload File Or Use Validation Data", ["Upload '.csv' or '.xlsx'", "Use Validation Data"])
		if choice == "Upload '.csv' or '.xlsx'":
			uploadedFile = st.sidebar.file_uploader("Choose a file", type = ['csv', 'xlsx'], accept_multiple_files = False, key = "fileUploader")
			opt_ = st.sidebar.toggle("Output With Pre-Processed Values")
			if opt_ == False:
				opt__ = st.sidebar.toggle("DataFrame View")
			if uploadedFile is not None :
				try:

					data = pd.read_csv(uploadedFile)
					_data__ = data.copy()
				except:
						try:
							data = pd.read_excel(uploadedFile)
							_data__ = data.copy()
						except:      
							data = pd.DataFrame(uploadedFile)
							_data__ = data.copy()
		elif choice == "Use Validation Data":
			opt_ = st.sidebar.toggle("Output With Pre-Processed Values")
			if opt_ == False:
				opt__ = st.sidebar.toggle("DataFrame View")
				
				_data__ = df0.copy()
				_data__ = _data__.reset_index(drop=True)

			data = df0
					
		else:
			st.sidebar.warning("You need to upload a csv or excel file.")
	elif type_ == 'Manual Input':
		print(1)
		# Analyze each column to determine type and range
		# Select a random row to use as default values
		# Sample random row from the DataFrame
		# random_row = dfo.sample(1).iloc[0]

		# # Detect the data type and range of each column
		# column_details = {}
		# for column in dfo.columns:
		#     dtype = dfo[column].dtype
			
		#     if np.issubdtype(dtype, np.number):
		#         min_value = dfo[column].min()
		#         max_value = dfo[column].max()
		#         default_value = random_row[column]
		#         column_details[column] = {
		#             "type": "numeric",
		#             "min": min_value,
		#             "max": max_value,
		#             "default": default_value,
		#         }
			
		#     # Change from np.object to object
		#     elif np.issubdtype(dtype, object):
		#         unique_values = dfo[column].unique()
		#         default_value = random_row[column]
		#         column_details[column] = {
		#             "type": "categorical",
		#             "options": unique_values,
		#             "default": default_value,
		#         }
			
		#     elif np.issubdtype(dtype, np.bool_):
		#         default_value = random_row[column]
		#         column_details[column] = {
		#             "type": "boolean",
		#             "default": default_value,
		#         }

		# # Create Streamlit input elements based on the column details
		# st.write("Dynamic Input Form")
		# user_inputs = {}

		# for column, details in column_details.items():
		#     if details["type"] == "numeric":
		#         user_inputs[column] = st.number_input(
		#             f"Enter a value for {column}",
		#             min_value=details["min"],
		#             max_value=details["max"],
		#             value=details["default"],  # Use the random row's value as the default
		#         )
			
		#     elif details["type"] == "categorical":
		#         user_inputs[column] = st.selectbox(
		#             f"Select a value for {column}",
		#             options=details["options"],
		#             index=list(details["options"]).index(details["default"]),  # Set the default selection
		#         )
			
		#     elif details["type"] == "boolean":
		#         user_inputs[column] = st.checkbox(
		#             f"Toggle {column}",
		#             value=details["default"],  # Use the random row's boolean value
		#         )
		# shuffle = st.button("Shuffle")
		# if shuffle:
		#     st.rerun()
		# # Output the collected inputs
		# st.write("User Inputs:")
		# st.write(user_inputs)
	
	# html_temp = """
	# <div style="background-color:tomato;padding:10px">
	# <p style="color:white;text-align:center;">Add DataBase Credientials </p>
	# </div>
	# """
	# st.sidebar.markdown(html_temp, unsafe_allow_html = True)
			
	# user = st.sidebar.text_input("user", "Type Here")
	# pw = st.sidebar.text_input("password", "Type Here", type="password")
	# db = st.sidebar.text_input("database", "Type Here")
	
	result = ""
	# Get the current working directory
	cwd = os.getcwd()
	try:
		# Find all files in the cwd ending with 'model.pkl'
		reg_models = glob.glob(os.path.join(cwd, "*model_registered.pkl"))
		model_files = glob.glob(os.path.join(cwd, "*model.pkl"))
		all_models = reg_models + model_files
		# Check if any such files exist
		with st.expander("Models Found"):
			# st.write("More Info ⬇")
			tab1, tab2, tab3 = st.tabs(["All Models", "Registered Models (From Custom Model Training)", "Multi Model Trained (From Unsupervised Learning)"])
			with tab1:
				if all_models:
					st.write("All Models found:", all_models)
			with tab2:
				if reg_models:
					st.write("'model_registered.pkl' found:", reg_models)
			with tab3:
				if model_files:
					st.write("'model.pkl' found:", model_files)

		select_model = st.selectbox("Select from trained all model:", all_models)

		model_pipe = joblib.load(select_model)

		st.write(model_pipe)
		st.subheader("Flow:")
		st.write(st.session_state.full_flow)
		Flow = st.session_state.pre_act
	except Exception as e:
		print("e13", e)
	if type_ == 'Upload File':
		if choice == "Upload '.csv' or '.xlsx'":
			for act in Flow:
				if act == 'Drop Column :orange[(For Col)]':
					try:
						# Full path to the file you're looking for
						drop_files = glob.glob(os.path.join(cwd, "*drop_pipeline.pkl"))
						
						st.write(drop_files)

						for drop_file_path in drop_files:
						# Check if the file exists
							if os.path.isfile(drop_file_path):
								st.write(f"File 'drop_pipeline.pkl' found at {drop_file_path}")
								drop_pipe = joblib.load(drop_file_path)
								st.write(drop_pipe)
								
								try:
									data = apply_pipeline_drop(data, drop_pipe)
									st.dataframe(data)
									
								except Exception as e:
									print("e35", e)
							else:
								st.write("No file named 'drop_pipeline.pkl' found in the current directory.")
					except Exception as e:
						print("e_:14=",e)

				elif act == 'Change Data Type :orange[(For Col)]':

					try:
						# datatype_file_path = os.path.join(cwd, "datatype_pipeline.pkl")
						datatype_files = glob.glob(os.path.join(cwd, "*datatype_pipeline.pkl"))
						for datatype_file_path in datatype_files:
							# Check if the file exists
							if os.path.isfile(datatype_file_path):
								st.write(f"File 'drop_pipeline.pkl' found at {datatype_file_path}")
								datatype_pipe = joblib.load(datatype_file_path)
								st.write(datatype_pipe)
								
								try:
									data = apply_pipeline_d_tpye(data, datatype_pipe)
									st.dataframe(data)
								except Exception as e:
									print("e34", e)
							else:
								st.write("No file named 'datatype_pipeline.pkl' found in the current directory.")
					except Exception as e:
						print("e15", e)

				elif act == 'Treat Missing :orange[(For Col)]':

					try:
						# treat_missing_vaues_for_col_file_path = os.path.join(cwd, "treat_missing_vaues_for_col.pkl")
						col_missing_files = glob.glob(os.path.join(cwd, "*treat_missing_vaues_for_col.pkl"))
						for treat_missing_vaues_for_col_file_path in col_missing_files:
							if os.path.isfile(treat_missing_vaues_for_col_file_path):
								st.write(f"File 'treat_missing_vaues_for_col.pkl' found at {treat_missing_vaues_for_col_file_path}")
								treat_missing_vaues_for_col_pipe = joblib.load(treat_missing_vaues_for_col_file_path)
								st.write(treat_missing_vaues_for_col_pipe)
								pipe__ = treat_missing_vaues_for_col_pipe['method']
								selected_column_ = treat_missing_vaues_for_col_pipe['column']
								try:
									# st.dataframe(data)
									# data = treat_missing_vaues_for_col(data, treat_missing_vaues_for_col_pipe)
									# Transform the unseen data
									data[selected_column_] = pipe__.transform(data[[selected_column_]]).ravel()
									st.dataframe(data)
								except Exception as e:
									print("e31", e)
								
							else:
								st.write("No file named 'treat_missing_vaues_for_col.pkl' found in the current directory.")
					except Exception as e:
						print("e19", e)

				elif act == 'Treat Missing :green[(Full DF)]':

					try:
						treat_missing_vaues_in_full_df_file_path_num = os.path.join(cwd, "numeric_treat_missing_vaues_in_full_df.pkl")
						if os.path.isfile(treat_missing_vaues_in_full_df_file_path_num):
							st.write(f"File 'numeric_treat_missing_vaues_in_full_df.pkl' found at {treat_missing_vaues_in_full_df_file_path_num}")
							treat_missing_vaues_in_full_df_pipe = joblib.load(treat_missing_vaues_in_full_df_file_path_num)
							st.write(treat_missing_vaues_in_full_df_pipe)
							# st.session_state.unsup_num_col = numeric_columns
							# st.session_state.unsup_cat_col = categorical_columns
							# numeric_columns = data.select_dtypes(include=np.number).columns.tolist()
							numeric_columns = st.session_state.unsup_num_col
							st.write(numeric_columns)
							# Transform numerical columns
							data[numeric_columns] = treat_missing_vaues_in_full_df_pipe.transform(data[numeric_columns])
							# data = treat_missing_values_in_full_df(data, treat_missing_vaues_in_full_df_pipe)
							st.dataframe(data)
						else:
							st.write("No file named 'numeric_treat_missing_vaues_in_full_df.pkl' found in the current directory.")
					except Exception as e:
						print("e20e20", e)

					try:
						treat_missing_vaues_in_full_df_file_path_obj = os.path.join(cwd, "categorical_treat_missing_vaues_in_full_df.pkl")
						if os.path.isfile(treat_missing_vaues_in_full_df_file_path_obj):
							st.write(f"File 'categorical_treat_missing_vaues_in_full_df.pkl' found at {treat_missing_vaues_in_full_df_file_path_obj}")
							treat_missing_vaues_in_full_df_pipe = joblib.load(treat_missing_vaues_in_full_df_file_path_obj)
							st.write(treat_missing_vaues_in_full_df_pipe)
							# categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
							categorical_columns = st.session_state.unsup_cat_col
							# Transform categorical columns
							data[categorical_columns] = treat_missing_vaues_in_full_df_pipe.transform(data[categorical_columns])
							# data = treat_missing_values_in_full_df(data, treat_missing_vaues_in_full_df_pipe)
							st.dataframe(data)
						else:
							st.write("No file named 'categorical_treat_missing_vaues_in_full_df.pkl' found in the current directory.")
					except Exception as e:
						print("e20", e)

				elif act == 'Drop Duplicates :green[(Full DF)]':
				
					try:
						drop_dup_file_path = os.path.join(cwd, "drop_dup_pipeline.pkl")
						if os.path.isfile(drop_dup_file_path):
							st.write(f"File 'drop_dup_pipeline.pkl' found at {drop_dup_file_path}")
							drop_dup_pipe = joblib.load(drop_dup_file_path)
							st.write(drop_dup_pipe)
							data = apply_pipeline_drop_dup(data, drop_dup_pipe)
							st.dataframe(data)
						else:
							st.write("No file named 'drop_dup_pipeline.pkl' found in the current directory.")
					except Exception as e:
						print("e16", e)

				elif act == 'Treat Outliers :orange[(For Col)]':

					try:
						# outliers_for_col_file_path = os.path.join(cwd, "outliers_for_col_pipeline.pkl")
						col_outliers_files = glob.glob(os.path.join(cwd, "*outliers_for_col_pipeline.pkl"))

						for outliers_for_col_file_path in col_outliers_files:
							if os.path.isfile(outliers_for_col_file_path):
								# st.write("jnnkj")
								st.write(f"File 'outliers_for_col_pipeline.pkl' found at {outliers_for_col_file_path}")
								outliers_for_col_pipe = joblib.load(outliers_for_col_file_path)
								st.code(outliers_for_col_pipe)
								try:
									# st.dataframe(data)
									# data = apply_outliers_for_col(data, outliers_for_col_pipe)
									# Transforming the unseen data
									# Transforming the unseen data
									data_ = outliers_for_col_pipe.transform(data)

									# Convert the transformed numpy array back to a DataFrame
									transformed_columns = outliers_for_col_pipe.transformers_[0][2] + [col for col in data.columns if col not in outliers_for_col_pipe.transformers_[0][2]]
									data = pd.DataFrame(data, columns=transformed_columns)
									st.dataframe(data)

									# Reapply the original data types
									for col in data.columns:
										data[col] = data[col].astype(data[col].dtype)

									# Reorder columns to match the original DataFrame
									data = data[data.columns]

									# df_unseen_winsorized = outliers_for_col_pipe.transform(data)

									# data = pd.DataFrame(df_unseen_winsorized, columns=data.columns)
									# st.dataframe(data)
								except Exception as e:
									print("e32g", e)
							else:
								st.write("No file named 'outliers_for_col_pipeline.pkl' found in the current directory.")
					except Exception as e:
						print("e17", e)

				elif act == 'Treat Outliers :green[(Full DF)]':
				
					def apply_winsorization(df, pipe):
						# Make a copy of the DataFrame to preserve original data and column order
						df_copy = df.copy()
						numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
						# Fit and transform the winsorization on the specified column
						df_transformed = pipe.transform(df[numeric_columns])
						
						# Convert the transformed array back to a DataFrame
						df_transformed = pd.DataFrame(df_transformed, columns=numeric_columns)

						# Replace original numeric columns with transformed ones
						df_copy[numeric_columns] = df_transformed

						return df_copy
					try:
						outliers_for_full_df_file_path = os.path.join(cwd, "outliers_for_full_df_pipeline.pkl")
						if os.path.isfile(outliers_for_full_df_file_path):
							st.write(f"File 'outliers_for_full_df_pipeline.pkl' found at {outliers_for_full_df_file_path}")
							outliers_for_full_df_pipe = joblib.load(outliers_for_full_df_file_path)
							st.write("l")
							st.code(outliers_for_full_df_pipe)
							
							data = apply_winsorization(data, outliers_for_full_df_pipe)
							# st.write(data_)
							st.dataframe(data)
							# Convert the transformed array back to a DataFrame
							# data = pd.DataFrame(data_, columns=data.columns)

							# data = apply_outliers_full_df(data, outliers_for_full_df_pipe)
						else:
							st.write("No file named 'outliers_for_full_df_pipeline.pkl' found in the current directory.")
					except Exception as e:
						print("e18", e)
				elif act == 'Apply Transformation :orange[(For Col)]':	

				# Missing Vaalues


					try:
						# apply_transformation_on_col_file_path = os.path.join(cwd, "apply_transformation_on_col.pkl")
						tran_files = glob.glob(os.path.join(cwd, "*apply_transformation_on_col.pkl"))
						for apply_transformation_on_col_file_path in tran_files:
							if os.path.isfile(apply_transformation_on_col_file_path):
								st.write(f"File 'treat_missing_vaues_in_full_df.pkl' found at {apply_transformation_on_col_file_path}")
								apply_transformation_on_col_pipe = joblib.load(apply_transformation_on_col_file_path)
								st.write(apply_transformation_on_col_pipe)
								try:
									data = apply_transformation_on_col(data, apply_transformation_on_col_pipe)
									st.dataframe(data)
								except Exception as e:
									print("e30", e)
							else:
								st.write("No file named 'apply_transformation_on_col.pkl' found in the current directory.")
					except Exception as e:
						print("e21", e)

				elif act == 'Column Unique Value Replacement :orange[(For Col)]':
				
				
					# apply_transformation_on_col_file_path = os.path.join(cwd, "apply_transformation_on_col.pkl")
					# if os.path.isfile(apply_transformation_on_col_file_path):
					#     st.write(f"File 'apply_transformation_on_col.pkl' found at {apply_transformation_on_col_file_path}")
					#     apply_transformation_on_col_pipe = joblib.load(apply_transformation_on_col_file_path)
					#     st.write(apply_transformation_on_col_pipe)
					# else:
					#     st.write("No file named 'apply_transformation_on_col.pkl' found in the current directory.")
					try:
						# column_unique_value_replacement_file_path = os.path.join(cwd, "column_unique_value_replacement.pkl")
						out_rplac_files = glob.glob(os.path.join(cwd, "*column_unique_value_replacement.pkl"))
						for column_unique_value_replacement_file_path in out_rplac_files:
							if os.path.isfile(column_unique_value_replacement_file_path):
								st.write(f"File 'column_unique_value_replacement.pkl' found at {column_unique_value_replacement_file_path}")
								column_unique_value_replacement_pipe = joblib.load(column_unique_value_replacement_file_path)
								st.write(column_unique_value_replacement_pipe)
								try:
									data = column_unique_value_replacement(data, column_unique_value_replacement_pipe)
									st.dataframe(data)
								except Exception as e:
									print("e29", e)
							else:
								st.write("No file named 'column_unique_value_replacement.pkl' found in the current directory.")
					except Exception as e:
						print("e22", e)

				elif act == 'Discretize Variable :orange[(For Col)]':

					try:    
						# discretize_output_col_file_path = os.path.join(cwd, "discretize_output_col.pkl")
						discretize_output_col_files = glob.glob(os.path.join(cwd, "*discretize_output_col.pkl"))
						for discretize_output_col_file_path in discretize_output_col_files:
							if os.path.isfile(discretize_output_col_file_path):
								st.write(f"File 'discretize_output_col.pkl' found at {discretize_output_col_file_path}")
								discretize_output_col_pipe = joblib.load(discretize_output_col_file_path)
								st.code(discretize_output_col_pipe)
								try:
									# data = discretize_output_col(data, discretize_output_col_pipe)
									# Transforming new data
									# st.dataframe(data)
									column_name = discretize_output_col_pipe['column_name']
									pipe_discretize_output_col_pipe = discretize_output_col_pipe['strategy']
									data[column_name] = pipe_discretize_output_col_pipe.transform(data[[column_name]])
									st.dataframe(data)
								except Exception as e:
									print("e28", e)
							else:
								st.write("No file named 'discretize_output_col.pkl' found in the current directory.")
					except Exception as e:
						print("e23", e)

				elif act == 'Dummy Variables :green[(Full DF)]':
					
					try:
						apply_encoding_on_df_file_path = os.path.join(cwd, "apply_encoding_on_df.pkl")
						if os.path.isfile(apply_encoding_on_df_file_path):
							st.write(f"File 'apply_encoding_on_df.pkl' found at {apply_encoding_on_df_file_path}")
							# apply_encoding_on_df_pipe = joblib.load(apply_encoding_on_df_file_path)
							# st.code(apply_encoding_on_df_pipe)
							# df_new_encoded = apply_encoding_on_df_pipe.transform(data)
							
							# # Create a DataFrame with the encoded columns for new data
							# encoded_categorical_columns_new = apply_encoding_on_df_pipe.named_steps['encoder'].transformers_[0][1].get_feature_names_out(data.select_dtypes(include=['object']).columns)
							# numerical_columns_new = data.columns.difference(data.select_dtypes(include=['object']).columns)
							# encoded_feature_names_new = list(encoded_categorical_columns_new) + list(numerical_columns_new)
							# # st.dataframe(data)
							# df_new_encoded = pd.DataFrame(df_new_encoded, columns=encoded_feature_names_new, index=data.index)

							# # Drop the original columns used for encoding in new data
							# data = data.drop(columns=data.select_dtypes(include=['object']).columns).join(df_new_encoded[encoded_categorical_columns_new])
							# st.dataframe(data)


							# Load the pipeline
							apply_encoding_on_df_pipe = joblib.load(apply_encoding_on_df_file_path)
							st.code(apply_encoding_on_df_pipe)
							
							# Retrieve the original categorical columns and numerical columns used during fitting
							encoder = apply_encoding_on_df_pipe.named_steps['encoder']
							original_categorical_columns = encoder.transformers_[0][2]  # Get the original categorical columns
							original_feature_names = encoder.transformers_[0][1].get_feature_names_out(original_categorical_columns)
							
							# Get the categorical columns from the current data
							current_categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
							
							# Align current data with the original columns
							missing_cols = set(original_categorical_columns) - set(current_categorical_columns)
							additional_cols = set(current_categorical_columns) - set(original_categorical_columns)
							
							# Add missing columns with default values (e.g., NaN or a new category)
							for col in missing_cols:
								data[col] = np.nan  # or a default category
							
							# Remove any additional columns that were not present during fitting
							data = data.drop(columns=list(additional_cols))
							
							# Ensure the column order matches the original fitting
							data = data[original_categorical_columns + list(data.columns.difference(original_categorical_columns))]
							
							# Transform the data using the pipeline
							df_new_encoded = apply_encoding_on_df_pipe.transform(data)
							
							# Create a DataFrame with the encoded columns for new data
							numerical_columns_new = data.columns.difference(original_categorical_columns)
							encoded_feature_names_new = list(original_feature_names) + list(numerical_columns_new)
							
							# Convert the transformed array back to a DataFrame with the correct column names
							df_new_encoded = pd.DataFrame(df_new_encoded, columns=encoded_feature_names_new, index=data.index)
							
							# Drop the original categorical columns from the current DataFrame
							data = data.drop(columns=original_categorical_columns)
							
							# Now join the encoded DataFrame with the original DataFrame
							data = data.join(df_new_encoded, rsuffix='_encoded')

							# Display the transformed dataframe
							st.dataframe(data)
							# data = apply_encoding_on_df(data, apply_encoding_on_df_pipe)
						else:
							st.write("No file named 'apply_encoding_on_df.pkl' found in the current directory.")
					except Exception as e:
						print("e24", e)

				elif act == 'Dummy Variable :orange[(For Col)]':

					try:
						col_dum_files = glob.glob(os.path.join(cwd, "*_apply_encoding_on_col.pkl"))
						# st.write(col_dum_files)
						for apply_encoding_on_col_file_path in col_dum_files:
							if os.path.isfile(apply_encoding_on_col_file_path):
								st.write(f"File 'apply_encoding_on_col.pkl' found at {apply_encoding_on_col_file_path}")
								try:
									apply_encoding_on_col_pipe = joblib.load(apply_encoding_on_col_file_path)
								except Exception as e:
									print("g:07", e)
								# st.text("hi")
								st.write(apply_encoding_on_col_pipe)
								encoder_pipeline_loaded = apply_encoding_on_col_pipe['method']
								column_name = apply_encoding_on_col_pipe['column_name']
								try:
									# data = apply_encoding_on_col(data, apply_encoding_on_col_pipe)

									data__ = encoder_pipeline_loaded.transform(data)

									# Create a DataFrame with the encoded columns for new data
									if encoder_pipeline_loaded.encoding_method == 'onehot':
										encoded_categorical_columns_new = encoder_pipeline_loaded.named_steps['encoder'].transformers_[0][1].get_feature_names_out([column_name])
									else:
										encoded_categorical_columns_new = [column_name]

									# Combine feature names for the encoded categorical columns and numerical columns
									numerical_columns_new = data.columns.difference([column_name])
									encoded_feature_names_new = list(encoded_categorical_columns_new) + list(numerical_columns_new)

									data__ = pd.DataFrame(data__, columns=encoded_feature_names_new, index=data.index)

									# Drop the original column used for encoding in new data
									data = data.drop(columns=[column_name]).join(data__[encoded_categorical_columns_new])
									st.dataframe(data)
									# print("\nEncoded DataFrame for new data:")
									# print(df_new_final)
								except Exception as e:
									print("e27", e)
								# st.dataframe(data)
							else:
								st.write("No file named 'apply_encoding_on_col.pkl' found in the current directory.")
					except Exception as e:
						print("e25", e)

				elif act == 'Apply Scaling :green[(Full DF)]':

					try:
						apply_scaling_on_df_file_path = os.path.join(cwd, "apply_scaling_on_df.pkl")
						if os.path.isfile(apply_scaling_on_df_file_path):
							st.write(f"File 'apply_scaling_on_df.pkl' found at {apply_scaling_on_df_file_path}")
							apply_scaling_on_df_pipe = joblib.load(apply_scaling_on_df_file_path)
							st.code(apply_scaling_on_df_pipe)

							try:
								data = apply_scaling_on_df(data, apply_scaling_on_df_pipe)
								st.dataframe(data)
							except Exception as e:
								print("e26", e)
						else:
							st.write("No file named 'apply_scaling_on_df.pkl' found in the current directory.")
					except Exception as e:
						print("e26", e)
				try:
					data = data.reset_index(drop=True)
				except:
					pass
		elif choice == "Use Validation Data":
			data = df0
			data = data.reset_index(drop=True)
			# st.dataframe(data)
		

	elif type_ == 'Manual Input':
		# st.warning("hi1")
		random_row = df.sample(1).iloc[0]

		# Detect the data type and range of each column
		column_details = {}
		for column in df.columns:
			dtype = df[column].dtype
			
			if np.issubdtype(dtype, np.number):
				min_value = df[column].min()
				max_value = df[column].max()
				default_value = random_row[column]
				column_details[column] = {
					"type": "numeric",
					"min": min_value,
					"max": max_value,
					"default": default_value,
				}
			
			# Change from np.object to object
			elif np.issubdtype(dtype, object):
				unique_values = df[column].unique()
				default_value = random_row[column]
				column_details[column] = {
					"type": "categorical",
					"options": unique_values,
					"default": default_value,
				}
			
			elif np.issubdtype(dtype, np.bool_):
				default_value = random_row[column]
				column_details[column] = {
					"type": "boolean",
					"default": default_value,
				}

		# Create Streamlit input elements based on the column details
		st.write("Dynamic Input Form")
		user_inputs = {}

		for column, details in column_details.items():
			if details["type"] == "numeric":
				user_inputs[column] = st.number_input(
					f"Enter a value for {column}",
					min_value=details["min"],
					max_value=details["max"],
					value=details["default"],  # Use the random row's value as the default
				)
			
			elif details["type"] == "categorical":
				user_inputs[column] = st.selectbox(
					f"Select a value for {column}",
					options=details["options"],
					index=list(details["options"]).index(details["default"]),  # Set the default selection
				)
			
			elif details["type"] == "boolean":
				user_inputs[column] = st.checkbox(
					f"Toggle {column}",
					value=details["default"],  # Use the random row's boolean value
				)
		shuffle = st.button("Shuffle")
		if shuffle:
			st.rerun()
		# Output the collected inputs
		st.write("Your Inputs:")
		st.write(user_inputs)
		dota = pd.DataFrame([user_inputs])
		st.dataframe(dota)
		data = dota
		for act in Flow:
			if act == 'Drop Column :orange[(For Col)]':
				try:
					# Full path to the file you're looking for
					drop_files = glob.glob(os.path.join(cwd, "*drop_pipeline.pkl"))
					
					st.write(drop_files)

					for drop_file_path in drop_files:
					# Check if the file exists
						if os.path.isfile(drop_file_path):
							st.write(f"File 'drop_pipeline.pkl' found at {drop_file_path}")
							drop_pipe = joblib.load(drop_file_path)
							st.write(drop_pipe)
							
							try:
								data = apply_pipeline_drop(data, drop_pipe)
								
							except Exception as e:
								print("e35", e)
						else:
							st.write("No file named 'drop_pipeline.pkl' found in the current directory.")
				except Exception as e:
					print("e_:14=",e)
			elif act == 'Change Data Type :orange[(For Col)]':
				try:
					# datatype_file_path = os.path.join(cwd, "datatype_pipeline.pkl")
					datatype_files = glob.glob(os.path.join(cwd, "*datatype_pipeline.pkl"))
					for datatype_file_path in datatype_files:
						# Check if the file exists
						if os.path.isfile(datatype_file_path):
							st.write(f"File 'drop_pipeline.pkl' found at {datatype_file_path}")
							datatype_pipe = joblib.load(datatype_file_path)
							st.write(datatype_pipe)
							
							try:
								data = apply_pipeline_d_tpye(data, datatype_pipe)
							except Exception as e:
								print("e34", e)
						else:
							st.write("No file named 'datatype_pipeline.pkl' found in the current directory.")
				except Exception as e:
					print("e15", e)

			elif act == 'Treat Missing :orange[(For Col)]':

				try:
					# treat_missing_vaues_for_col_file_path = os.path.join(cwd, "treat_missing_vaues_for_col.pkl")
					col_missing_files = glob.glob(os.path.join(cwd, "*treat_missing_vaues_for_col.pkl"))
					for treat_missing_vaues_for_col_file_path in col_missing_files:
						if os.path.isfile(treat_missing_vaues_for_col_file_path):
							st.write(f"File 'treat_missing_vaues_for_col.pkl' found at {treat_missing_vaues_for_col_file_path}")
							treat_missing_vaues_for_col_pipe = joblib.load(treat_missing_vaues_for_col_file_path)
							st.write(treat_missing_vaues_for_col_pipe)
							pipe__ = treat_missing_vaues_for_col_pipe['method']
							selected_column_ = treat_missing_vaues_for_col_pipe['column']
							try:
								# st.dataframe(data)
								# data = treat_missing_vaues_for_col(data, treat_missing_vaues_for_col_pipe)
								# Transform the unseen data
								data[selected_column_] = pipe__.transform(data[[selected_column_]]).ravel()
								# st.dataframe(data)
							except Exception as e:
								print("e31", e)
							
						else:
							st.write("No file named 'treat_missing_vaues_for_col.pkl' found in the current directory.")
				except Exception as e:
					print("e19", e)

			elif act == 'Treat Missing :green[(Full DF)]':

				try:
					treat_missing_vaues_in_full_df_file_path_num = os.path.join(cwd, "numeric_treat_missing_vaues_in_full_df.pkl")
					if os.path.isfile(treat_missing_vaues_in_full_df_file_path_num):
						st.write(f"File 'numeric_treat_missing_vaues_in_full_df.pkl' found at {treat_missing_vaues_in_full_df_file_path_num}")
						treat_missing_vaues_in_full_df_pipe = joblib.load(treat_missing_vaues_in_full_df_file_path_num)
						st.write(treat_missing_vaues_in_full_df_pipe)
						numeric_columns = data.select_dtypes(include=np.number).columns.tolist()
						# Transform numerical columns
						data[numeric_columns] = treat_missing_vaues_in_full_df_pipe.transform(data[numeric_columns])
						# data = treat_missing_values_in_full_df(data, treat_missing_vaues_in_full_df_pipe)
						st.dataframe(data)
					else:
						st.write("No file named 'numeric_treat_missing_vaues_in_full_df.pkl' found in the current directory.")
				except Exception as e:
					print("e20", e)

				

				try:
					treat_missing_vaues_in_full_df_file_path_obj = os.path.join(cwd, "categorical_treat_missing_vaues_in_full_df.pkl")
					if os.path.isfile(treat_missing_vaues_in_full_df_file_path_obj):
						st.write(f"File 'categorical_treat_missing_vaues_in_full_df.pkl' found at {treat_missing_vaues_in_full_df_file_path_obj}")
						treat_missing_vaues_in_full_df_pipe = joblib.load(treat_missing_vaues_in_full_df_file_path_obj)
						st.write(treat_missing_vaues_in_full_df_pipe)
						categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
						# Transform categorical columns
						data[categorical_columns] = treat_missing_vaues_in_full_df_pipe.transform(data[categorical_columns])
						# data = treat_missing_values_in_full_df(data, treat_missing_vaues_in_full_df_pipe)
					else:
						st.write("No file named 'categorical_treat_missing_vaues_in_full_df.pkl' found in the current directory.")
				except Exception as e:
					print("e20", e)

			elif act == 'Drop Duplicates :green[(Full DF)]':	

				try:
					drop_dup_file_path = os.path.join(cwd, "drop_dup_pipeline.pkl")
					if os.path.isfile(drop_dup_file_path):
						st.write(f"File 'drop_dup_pipeline.pkl' found at {drop_dup_file_path}")
						drop_dup_pipe = joblib.load(drop_dup_file_path)
						st.write(drop_dup_pipe)
						data = apply_pipeline_drop_dup(data, drop_dup_pipe)
					else:
						st.write("No file named 'drop_dup_pipeline.pkl' found in the current directory.")
				except Exception as e:
					print("e16", e)

			elif act == 'Treat Outliers :orange[(For Col)]':

				try:
					# outliers_for_col_file_path = os.path.join(cwd, "outliers_for_col_pipeline.pkl")
					col_outliers_files = glob.glob(os.path.join(cwd, "*outliers_for_col_pipeline.pkl"))

					for outliers_for_col_file_path in col_outliers_files:
						if os.path.isfile(outliers_for_col_file_path):
							# st.write("jnnkj")
							st.write(f"File 'outliers_for_col_pipeline.pkl' found at {outliers_for_col_file_path}")
							outliers_for_col_pipe = joblib.load(outliers_for_col_file_path)
							st.code(outliers_for_col_pipe)
							try:
								# st.dataframe(data)
								# data = apply_outliers_for_col(data, outliers_for_col_pipe)
								# Transforming the unseen data
								# Transforming the unseen data
								data_ = outliers_for_col_pipe.transform(data)

								# Convert the transformed numpy array back to a DataFrame
								transformed_columns = outliers_for_col_pipe.transformers_[0][2] + [col for col in data.columns if col not in outliers_for_col_pipe.transformers_[0][2]]
								data = pd.DataFrame(data, columns=transformed_columns)

								# Reapply the original data types
								for col in data.columns:
									data[col] = data[col].astype(data[col].dtype)

								# Reorder columns to match the original DataFrame
								data = data[data.columns]

								# df_unseen_winsorized = outliers_for_col_pipe.transform(data)

								# data = pd.DataFrame(df_unseen_winsorized, columns=data.columns)
								# st.dataframe(data)
							except Exception as e:
								print("e32g", e)
						else:
							st.write("No file named 'outliers_for_col_pipeline.pkl' found in the current directory.")
				except Exception as e:
					print("e17", e)

			elif act == 'Treat Outliers :green[(Full DF)]':
			
				def apply_winsorization(df, pipe):
					# Make a copy of the DataFrame to preserve original data and column order
					df_copy = df.copy()
					numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
					# Fit and transform the winsorization on the specified column
					df_transformed = pipe.transform(df[numeric_columns])
					
					# Convert the transformed array back to a DataFrame
					df_transformed = pd.DataFrame(df_transformed, columns=numeric_columns)

					# Replace original numeric columns with transformed ones
					df_copy[numeric_columns] = df_transformed

					return df_copy
				try:
					outliers_for_full_df_file_path = os.path.join(cwd, "outliers_for_full_df_pipeline.pkl")
					if os.path.isfile(outliers_for_full_df_file_path):
						st.write(f"File 'outliers_for_full_df_pipeline.pkl' found at {outliers_for_full_df_file_path}")
						outliers_for_full_df_pipe = joblib.load(outliers_for_full_df_file_path)
						st.write("l")
						st.code(outliers_for_full_df_pipe)
						
						data = apply_winsorization(data, outliers_for_full_df_pipe)
						# st.write(data_)
						# st.dataframe(data)
						# Convert the transformed array back to a DataFrame
						# data = pd.DataFrame(data_, columns=data.columns)

						# data = apply_outliers_full_df(data, outliers_for_full_df_pipe)
					else:
						st.write("No file named 'outliers_for_full_df_pipeline.pkl' found in the current directory.")
				except Exception as e:
					print("e18", e)
			
			elif act == 'Apply Transformation :orange[(For Col)]':

				try:
					# apply_transformation_on_col_file_path = os.path.join(cwd, "apply_transformation_on_col.pkl")
					tran_files = glob.glob(os.path.join(cwd, "*apply_transformation_on_col.pkl"))
					for apply_transformation_on_col_file_path in tran_files:
						if os.path.isfile(apply_transformation_on_col_file_path):
							st.write(f"File 'treat_missing_vaues_in_full_df.pkl' found at {apply_transformation_on_col_file_path}")
							apply_transformation_on_col_pipe = joblib.load(apply_transformation_on_col_file_path)
							st.write(apply_transformation_on_col_pipe)
							try:
								data = apply_transformation_on_col(data, apply_transformation_on_col_pipe)
								# st.dataframe(data)
							except Exception as e:
								print("e30", e)
						else:
							st.write("No file named 'apply_transformation_on_col.pkl' found in the current directory.")
				except Exception as e:
					print("e21", e)
			
			elif act == 'Column Unique Value Replacement :orange[(For Col)]':
				# apply_transformation_on_col_file_path = os.path.join(cwd, "apply_transformation_on_col.pkl")
				# if os.path.isfile(apply_transformation_on_col_file_path):
				#     st.write(f"File 'apply_transformation_on_col.pkl' found at {apply_transformation_on_col_file_path}")
				#     apply_transformation_on_col_pipe = joblib.load(apply_transformation_on_col_file_path)
				#     st.write(apply_transformation_on_col_pipe)
				# else:
				#     st.write("No file named 'apply_transformation_on_col.pkl' found in the current directory.")
				try:
					# column_unique_value_replacement_file_path = os.path.join(cwd, "column_unique_value_replacement.pkl")
					out_rplac_files = glob.glob(os.path.join(cwd, "*column_unique_value_replacement.pkl"))
					for column_unique_value_replacement_file_path in out_rplac_files:
						if os.path.isfile(column_unique_value_replacement_file_path):
							st.write(f"File 'column_unique_value_replacement.pkl' found at {column_unique_value_replacement_file_path}")
							column_unique_value_replacement_pipe = joblib.load(column_unique_value_replacement_file_path)
							st.write(column_unique_value_replacement_pipe)
							try:
								data = column_unique_value_replacement(data, column_unique_value_replacement_pipe)
								# st.dataframe(data)
							except Exception as e:
								print("e29", e)
						else:
							st.write("No file named 'column_unique_value_replacement.pkl' found in the current directory.")
				except Exception as e:
					print("e22", e)

			elif act == 'Discretize Variable :orange[(For Col)]':

				try:    
					# discretize_output_col_file_path = os.path.join(cwd, "discretize_output_col.pkl")
					discretize_output_col_files = glob.glob(os.path.join(cwd, "*discretize_output_col.pkl"))
					for discretize_output_col_file_path in discretize_output_col_files:
						if os.path.isfile(discretize_output_col_file_path):
							st.write(f"File 'discretize_output_col.pkl' found at {discretize_output_col_file_path}")
							discretize_output_col_pipe = joblib.load(discretize_output_col_file_path)
							st.code(discretize_output_col_pipe)
							try:
								# data = discretize_output_col(data, discretize_output_col_pipe)
								# Transforming new data
								# st.dataframe(data)
								column_name = discretize_output_col_pipe['column_name']
								pipe_discretize_output_col_pipe = discretize_output_col_pipe['strategy']
								data[column_name] = pipe_discretize_output_col_pipe.transform(data[[column_name]])
							except Exception as e:
								print("e28", e)
						else:
							st.write("No file named 'discretize_output_col.pkl' found in the current directory.")
				except Exception as e:
					print("e23", e)
				
			elif act == 'Dummy Variables :green[(Full DF)]':

				try:
					apply_encoding_on_df_file_path = os.path.join(cwd, "apply_encoding_on_df.pkl")
					if os.path.isfile(apply_encoding_on_df_file_path):
						st.write(f"File 'apply_encoding_on_df.pkl' found at {apply_encoding_on_df_file_path}")
						apply_encoding_on_df_pipe = joblib.load(apply_encoding_on_df_file_path)
						st.code(apply_encoding_on_df_pipe)
						df_new_encoded = apply_encoding_on_df_pipe.transform(data)

						# Create a DataFrame with the encoded columns for new data
						encoded_categorical_columns_new = apply_encoding_on_df_pipe.named_steps['encoder'].transformers_[0][1].get_feature_names_out(data.select_dtypes(include=['object']).columns)
						numerical_columns_new = data.columns.difference(data.select_dtypes(include=['object']).columns)
						encoded_feature_names_new = list(encoded_categorical_columns_new) + list(numerical_columns_new)

						df_new_encoded = pd.DataFrame(df_new_encoded, columns=encoded_feature_names_new, index=data.index)

						# Drop the original columns used for encoding in new data
						data = data.drop(columns=data.select_dtypes(include=['object']).columns).join(df_new_encoded[encoded_categorical_columns_new])

						# data = apply_encoding_on_df(data, apply_encoding_on_df_pipe)
						st.dataframe(data)
					else:
						st.write("No file named 'apply_encoding_on_df.pkl' found in the current directory.")
				except Exception as e:
					print("e24", e)

			elif act == 'Dummy Variable :orange[(For Col)]':

				try:
					col_dum_files = glob.glob(os.path.join(cwd, "*_apply_encoding_on_col.pkl"))
					# st.write(col_dum_files)
					for apply_encoding_on_col_file_path in col_dum_files:
						if os.path.isfile(apply_encoding_on_col_file_path):
							st.write(f"File 'apply_encoding_on_col.pkl' found at {apply_encoding_on_col_file_path}")
							try:
								apply_encoding_on_col_pipe = joblib.load(apply_encoding_on_col_file_path)
							except Exception as e:
								print("g:07", e)
							# st.text("hi")
							st.write(apply_encoding_on_col_pipe)
							encoder_pipeline_loaded = apply_encoding_on_col_pipe['method']
							column_name = apply_encoding_on_col_pipe['column_name']
							try:
								# data = apply_encoding_on_col(data, apply_encoding_on_col_pipe)

								data__ = encoder_pipeline_loaded.transform(data)

								# Create a DataFrame with the encoded columns for new data
								if encoder_pipeline_loaded.encoding_method == 'onehot':
									encoded_categorical_columns_new = encoder_pipeline_loaded.named_steps['encoder'].transformers_[0][1].get_feature_names_out([column_name])
								else:
									encoded_categorical_columns_new = [column_name]

								# Combine feature names for the encoded categorical columns and numerical columns
								numerical_columns_new = data.columns.difference([column_name])
								encoded_feature_names_new = list(encoded_categorical_columns_new) + list(numerical_columns_new)

								data__ = pd.DataFrame(data__, columns=encoded_feature_names_new, index=data.index)

								# Drop the original column used for encoding in new data
								data = data.drop(columns=[column_name]).join(data__[encoded_categorical_columns_new])

								# print("\nEncoded DataFrame for new data:")
								# print(df_new_final)
							except Exception as e:
								print("e27", e)
							# st.dataframe(data)
						else:
							st.write("No file named 'apply_encoding_on_col.pkl' found in the current directory.")
				except Exception as e:
					print("e25", e)

			elif act == 'Apply Scaling :green[(Full DF)]':

				try:
					apply_scaling_on_df_file_path = os.path.join(cwd, "apply_scaling_on_df.pkl")
					if os.path.isfile(apply_scaling_on_df_file_path):
						st.write(f"File 'apply_scaling_on_df.pkl' found at {apply_scaling_on_df_file_path}")
						apply_scaling_on_df_pipe = joblib.load(apply_scaling_on_df_file_path)
						st.code(apply_scaling_on_df_pipe)

						try:
							data = apply_scaling_on_df(data, apply_scaling_on_df_pipe)
						except Exception as e:
							print("e26", e)
					else:
						st.write("No file named 'apply_scaling_on_df.pkl' found in the current directory.")
				except Exception as e:
					print("e26", e)
		dota = data
		try:
			dota = dota.reset_index(drop=True)
		except:
			pass

	

	

	
	if st.button("Predict", type="primary", use_container_width=True):
		# result = predict(dota, user, pw, db)
		try:
			if type_ == 'Upload File':
				if choice == "Upload '.csv' or '.xlsx'":
					result, predictions = predict(data, model_pipe)
					# st.dataframe(result)
					# Combine predictions with the original data 
					final_result = pd.concat([predictions, _data__], axis=1)
					
					# try:
					if opt_ == True:
						try:
							# cm = sns.light_palette("tomato", as_cmap = True)
							
							# styled_result = result.style

							# st.table(styled_result.background_gradient(cmap = cm).format(precision=2))
				
							
							# # Function to create a color map for the 'Clusters' column based on its value
							# def generate_color(val, max_cluster):
							# 	if isinstance(val, int):
							# 		# Normalize the cluster value to get a value between 0 and 1
							# 		norm_val = val / max_cluster
							# 		# Use a linear gradient from white to red
							# 		color = sns.light_palette("red", as_cmap=True)(norm_val)  # Get the corresponding color in RGB
							# 		# Convert RGB to hexadecimal for background color
							# 		color_hex = mcolors.to_hex(color)
							# 		return f'background-color: {color_hex}'
							# 	else:
							# 		return ''  # No color for non-integer values

							# 			# Get the maximum cluster value to determine the range
							# max_cluster = result["Clusters"].max()

							# # Define a color map for other columns
							# cm = sns.light_palette("tomato", as_cmap=True)

							# # Style the DataFrame
							# styled_result = result.style

							# # Apply the gradient to all columns except 'Clusters'
							# styled_result = styled_result.background_gradient(cmap=cm, subset=data.columns.difference(["Clusters"]))

							# # Apply the custom color map to the 'Clusters' column
							# styled_result = styled_result.applymap(lambda val: generate_color(val, max_cluster), subset=["Clusters"])

							# # Display the styled table in Streamlit
							# st.table(styled_result.format(precision=2))
				
							# Change the Pandas option to allow more elements to be styled
							pd.set_option("styler.render.max_elements", 5000000000000000000000000000)  # Adjust this to a value greater than your cell count

							# Define color maps for the gradients
							tomato_cm = sns.light_palette("tomato", as_cmap=True)
							gold_cm = sns.light_palette("gold", as_cmap=True)

							# Function to create a color map for the 'Clusters' column based on its value
							def generate_color(val, max_cluster):
								if isinstance(val, int):
									norm_val = val / max_cluster
									color = sns.light_palette("lightblue", as_cmap=True)(norm_val)
									color_hex = mcolors.to_hex(color)
									return f'background-color: {color_hex}'
								else:
									return ''  # No color for non-integer values

							# Maximum cluster value for the color map
							max_cluster = result["Clusters"].max()

							# Apply gradients to columns, using different color maps
							styled_result = result.style

							# Tomato color map for 'Feature1' and 'Feature2'
							styled_result = styled_result.background_gradient(cmap=tomato_cm)

							# Gold color map for 'Feature3'
							# styled_result = styled_result.background_gradient(cmap=gold_cm)

							# Custom color map for the 'Clusters' column
							styled_result = styled_result.applymap(lambda val: generate_color(val, max_cluster), subset=["Clusters"])

							# Display the styled table in Streamlit
							st.table(styled_result.format(precision=2))
						except Exception as e:
							st.dataframe(final_result)
							pass


						st.success("Success ✅")
					else:

						final_result = pd.concat([predictions, _data__], axis=1)
						# try:
						cm = sns.light_palette("tomato", as_cmap=True)

						# Apply the color map to the DataFrame
						# styled_df = final_result.style.background_gradient(cmap=cm)
						
						styled_df_info1 = (
							final_result.style.set_table_styles(
								[
									{'selector': 'th', 'props': 'background-color: #15296B; color: white;'},  # Table headers
									{'selector': 'td', 'props': 'background-color: #FF4B4B; border: 1px solid black;'},  # Table cells
									{'selector': 'tr', 'props': 'border: 2px solid black;'}  # Table rows
								]
							)
							.highlight_max(axis=0, props='background-color: tomato; color: black;')  # Highlight max values with tomato color
							.highlight_min(axis=0, props='background-color: lightcoral; color: white;')  # Highlight min values with lightcoral color
							# .format("{:.2f}")
						)

						# Display the styled DataFrame in Streamlit
						if opt__ == False:
							try:
								st.table(styled_df_info1)
							except:
								st.dataframe(final_result)
						else:
							try:
								st.dataframe(styled_df_info1)
							except:
								st.dataframe(final_result)

				else:     # Use Validation Data
					result, predictions = predict(data, model_pipe)
					# st.dataframe(result)
					# Combine predictions with the original data 
					final_result = pd.concat([predictions, data], axis=1)
					# st.dataframe(final_result)

					if opt_ == True:
						
						# cm = sns.light_palette("tomato", as_cmap = True)
						
						# styled_result = result.style

						# st.table(styled_result.background_gradient(cmap = cm).format(precision=2))
			
						
						# # Function to create a color map for the 'Clusters' column based on its value
						# def generate_color(val, max_cluster):
						# 	if isinstance(val, int):
						# 		# Normalize the cluster value to get a value between 0 and 1
						# 		norm_val = val / max_cluster
						# 		# Use a linear gradient from white to red
						# 		color = sns.light_palette("red", as_cmap=True)(norm_val)  # Get the corresponding color in RGB
						# 		# Convert RGB to hexadecimal for background color
						# 		color_hex = mcolors.to_hex(color)
						# 		return f'background-color: {color_hex}'
						# 	else:
						# 		return ''  # No color for non-integer values

						# 			# Get the maximum cluster value to determine the range
						# max_cluster = result["Clusters"].max()

						# # Define a color map for other columns
						# cm = sns.light_palette("tomato", as_cmap=True)

						# # Style the DataFrame
						# styled_result = result.style

						# # Apply the gradient to all columns except 'Clusters'
						# styled_result = styled_result.background_gradient(cmap=cm, subset=data.columns.difference(["Clusters"]))

						# # Apply the custom color map to the 'Clusters' column
						# styled_result = styled_result.applymap(lambda val: generate_color(val, max_cluster), subset=["Clusters"])

						# # Display the styled table in Streamlit
						# st.table(styled_result.format(precision=2))
			
						# Change the Pandas option to allow more elements to be styled
						pd.set_option("styler.render.max_elements", 5000000000000000000000000000)  # Adjust this to a value greater than your cell count

						# Define color maps for the gradients
						tomato_cm = sns.light_palette("tomato", as_cmap=True)
						gold_cm = sns.light_palette("gold", as_cmap=True)

						# Function to create a color map for the 'Clusters' column based on its value
						def generate_color(val, max_cluster):
							if isinstance(val, int):
								norm_val = val / max_cluster
								color = sns.light_palette("lightblue", as_cmap=True)(norm_val)
								color_hex = mcolors.to_hex(color)
								return f'background-color: {color_hex}'
							else:
								return ''  # No color for non-integer values

						# Maximum cluster value for the color map
						max_cluster = result["Clusters"].max()

						# Apply gradients to columns, using different color maps
						styled_result = result.style

						# Tomato color map for 'Feature1' and 'Feature2'
						styled_result = styled_result.background_gradient(cmap=tomato_cm)

						# Gold color map for 'Feature3'
						# styled_result = styled_result.background_gradient(cmap=gold_cm)

						# Custom color map for the 'Clusters' column
						styled_result = styled_result.applymap(lambda val: generate_color(val, max_cluster), subset=["Clusters"])

						# Display the styled table in Streamlit
						st.table(styled_result.format(precision=2))


						st.success("Success ✅")
					else:

						final_result = pd.concat([predictions, _data__], axis=1)
						cm = sns.light_palette("tomato", as_cmap=True)

						# Apply the color map to the DataFrame
						styled_df = final_result.style.background_gradient(cmap=cm)
						
						styled_df_info1 = (
							final_result.style.set_table_styles(
								[
									{'selector': 'th', 'props': 'background-color: #15296B; color: white;'},  # Table headers
									{'selector': 'td', 'props': 'background-color: #FF4B4B; border: 1px solid black;'},  # Table cells
									{'selector': 'tr', 'props': 'border: 2px solid black;'}  # Table rows
								]
							)
							.highlight_max(axis=0, props='background-color: tomato; color: black;')  # Highlight max values with tomato color
							.highlight_min(axis=0, props='background-color: lightcoral; color: white;')  # Highlight min values with lightcoral color
							# .format("{:.2f}")
						)

						# Display the styled DataFrame in Streamlit
						if opt__ == False:
							st.table(styled_df_info1)
						else:
							st.dataframe(styled_df_info1)
			elif type_ == 'Manual Input':
				result, predictions = predict(dota, model_pipe)

				# Combine predictions with the original data 
				final_result = pd.concat([predictions, _data__], axis=1)
				


				pd.set_option("styler.render.max_elements", 5000000000000000000000000000)  # Adjust this to a value greater than your cell count

				# Define color maps for the gradients
				tomato_cm = sns.light_palette("tomato", as_cmap=True)
				gold_cm = sns.light_palette("gold", as_cmap=True)

				# Function to create a color map for the 'Clusters' column based on its value
				def generate_color(val, max_cluster):
					if isinstance(val, int):
						norm_val = val / max_cluster
						color = sns.light_palette("lightblue", as_cmap=True)(norm_val)
						color_hex = mcolors.to_hex(color)
						return f'background-color: {color_hex}'
					else:
						return ''  # No color for non-integer values

				# Maximum cluster value for the color map
				max_cluster = result["Clusters"].max()

				# Apply gradients to columns, using different color maps
				styled_result = result.style

				# Tomato color map for 'Feature1' and 'Feature2'
				styled_result = styled_result.background_gradient(cmap=tomato_cm)

				# Gold color map for 'Feature3'
				# styled_result = styled_result.background_gradient(cmap=gold_cm)

				# Custom color map for the 'Clusters' column
				styled_result = styled_result.applymap(lambda val: generate_color(val, max_cluster), subset=["Clusters"])

				# Display the styled table in Streamlit
				st.table(styled_result.format(precision=2))


				st.success("Success ✅")
				st.warning("Not Yet Released")
				st.success("Comming Soon 🔜 @ https://pypi.org/project/glook/")
		except UnboundLocalError:
			st.warning("Upload .csv or .xlsx")
		except Exception as e:
			st.error(e)
			print(e)
		
if __name__=='__main__':
	main()


