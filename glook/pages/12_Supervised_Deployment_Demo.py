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
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

if "X_test" in st.session_state:
	
	X_test0 = st.session_state.X_test0
	X_test = st.session_state.X_test
	y_name = st.session_state.y_name
try:
	# dfo = st.session_state.X_test0
	dfo = st.session_state.X_val1
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
		elif method == 'IQR':
		# Apply IQR method to all numerical columns
			for column_name in numerical_columns:
				# Calculate IQR
				Q1 = df[column_name].quantile(0.25)  # First quartile
				Q3 = df[column_name].quantile(0.75)  # Third quartile
				IQR = Q3 - Q1  # Interquartile range
				
				# Define lower and upper bounds for outliers
				lower_bound = Q1 - 1.5 * IQR
				upper_bound = Q3 + 1.5 * IQR
				
				# Cap values outside these bounds
				df[column_name] = np.where(df[column_name] < lower_bound, lower_bound, df[column_name])
				df[column_name] = np.where(df[column_name] > upper_bound, upper_bound, df[column_name])
			
			st.warning("Outliers have been treated using the IQR method, but changes will not be saved until the 'Confirm Changes' button is pressed.")
			print(19)
	
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
def predict(data, model_pipe, y_name):
	# engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")
	clean_data = data
	model1 = model_pipe['model']
	# hi = model1.predict(clean_data)
	# st.write(hi)
	# print(hi)
	# Make predictions using the loaded model
	y_name = y_name + '_Predictions'
	predictions = pd.DataFrame(model1.predict(clean_data), columns=[y_name])

	# Combine predictions with the original data
	final_result = pd.concat([predictions, data], axis=1)

	# Display the final result
	# print(final_result)
	# final.to_sql('cancer_predictions', con = engine, if_exists = 'replace', chunksize = 1000, index = False)

	return final_result, predictions


def main():  

	st.title("Prediction")
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
		choice = st.sidebar.radio("Upload File Or use Validation Data", ["Upload '.csv' or '.xlsx'", "Use Validation Data"])
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
			opt__ = st.sidebar.toggle("DataFrame View")
			data = X_test0
			# st.write(data)
					
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
		select_model = st.selectbox("Select trained model:", model_files)

		model_pipe = joblib.load(select_model)
		st.write(model_pipe)
		st.subheader("Flow:")
		# st.write(st.session_state.pre_act)
		st.write(st.session_state.full_flow)
		Flow = st.session_state.pre_act
		# st.dataframe(data)
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
							y_var = st.session_state.y_var
							numeric_columns = data.select_dtypes(include=np.number).columns.difference([y_var]).tolist()
							            # Get the feature names that were used during the fitting
							fitted_features = treat_missing_vaues_in_full_df_pipe.feature_names_in_

							# Drop columns that were not used during fitting
							numeric_columns = [col for col in numeric_columns if col in fitted_features]
							# Transform numerical columns
							data[numeric_columns] = treat_missing_vaues_in_full_df_pipe.transform(data[numeric_columns])
							# data = df_ccopy
							st.dataframe(data)
							# data = treat_missing_values_in_full_df(data, treat_missing_vaues_in_full_df_pipe)
						else:
							st.write("No file named 'numeric_treat_missing_vaues_in_full_df.pkl' found in the current directory.")
					except Exception as e:
						print("e201", e)

					try:
						treat_missing_vaues_in_full_df_file_path_obj = os.path.join(cwd, "categorical_treat_missing_vaues_in_full_df.pkl")
						if os.path.isfile(treat_missing_vaues_in_full_df_file_path_obj):
							st.write(f"File 'categorical_treat_missing_vaues_in_full_df.pkl' found at {treat_missing_vaues_in_full_df_file_path_obj}")
							treat_missing_vaues_in_full_df_pipe = joblib.load(treat_missing_vaues_in_full_df_file_path_obj)
							st.write(treat_missing_vaues_in_full_df_pipe)

							y_var = st.session_state.y_var
							# categorical_columns = data.select_dtypes(include=['object']).columns.difference([y_var]).tolist()



							# Identify categorical columns in the data
							categorical_columns = data.select_dtypes(include=['object']).columns.difference([y_var]).tolist()
							
							# Get the features that were used during the fit
							fitted_features = treat_missing_vaues_in_full_df_pipe.feature_names_in_
							
							# Add missing features to the dataset with NaN values
							for feature in fitted_features:
								if feature not in data.columns:
									data[feature] = np.nan
									st.write(f"Added missing feature '{feature}' with NaN values to the dataset.")
							
							# Transform the data using the pipeline
							data[fitted_features] = treat_missing_vaues_in_full_df_pipe.transform(data[fitted_features])
							st.dataframe(data)

							#### COMMENTED
							# # Transform numerical columns
							# data[categorical_columns] = treat_missing_vaues_in_full_df_pipe.transform(data[categorical_columns])					
							# st.dataframe(data)
							#### COMMENTED



							# numeric_columns = df_ccopy.select_dtypes(include=np.number).columns.difference([y_var]).tolist()
							# # Transform numerical columns
							# df_ccopy[numeric_columns] = treat_missing_vaues_in_full_df_pipe.transform(df_ccopy[numeric_columns])
							# data = df_ccopy
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

									# Reapply the original data types
									for col in data.columns:
										data[col] = data[col].astype(data[col].dtype)

									# Reorder columns to match the original DataFrame
									data = data[data.columns]

									# df_unseen_winsorized = outliers_for_col_pipe.transform(data)

									# data = pd.DataFrame(df_unseen_winsorized, columns=data.columns)
									st.dataframe(data)
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
						y_var = st.session_state.y_var

						numeric_columns = df.select_dtypes(include=['number']).columns.difference([y_var]).tolist()
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
									# st.dataframe(data)
								except Exception as e:
									print("e30", e)
							else:
								st.write("No file named 'apply_transformation_on_col.pkl' found in the current directory.")
					except Exception as e:
						print("e21", e)
				
				
				# apply_transformation_on_col_file_path = os.path.join(cwd, "apply_transformation_on_col.pkl")
				# if os.path.isfile(apply_transformation_on_col_file_path):
				#     st.write(f"File 'apply_transformation_on_col.pkl' found at {apply_transformation_on_col_file_path}")
				#     apply_transformation_on_col_pipe = joblib.load(apply_transformation_on_col_file_path)
				#     st.write(apply_transformation_on_col_pipe)
				# else:
				#     st.write("No file named 'apply_transformation_on_col.pkl' found in the current directory.")
				elif act == 'Column Unique Value Replacement :orange[(For Col)]':
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
								st.write(discretize_output_col_pipe)
								try:
									# data = discretize_output_col(data, discretize_output_col_pipe)
									# Transforming new data
									
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

				elif act == 'Dummy Variables :green[(Full DF)]':
					
					try:
						apply_encoding_on_df_file_path = os.path.join(cwd, "apply_encoding_on_df.pkl")
						if os.path.isfile(apply_encoding_on_df_file_path):
							st.write(f"File 'apply_encoding_on_df.pkl' found at {apply_encoding_on_df_file_path}")
							apply_encoding_on_df_pipe = joblib.load(apply_encoding_on_df_file_path)
							st.code(apply_encoding_on_df_pipe)
							y_var = st.session_state.y_var
							df_new_encoded = apply_encoding_on_df_pipe.transform(data)
							data_ = data.copy()
							# Create a DataFrame with the encoded columns for new data
							encoded_categorical_columns_new = apply_encoding_on_df_pipe.named_steps['encoder'].transformers_[0][1].get_feature_names_out(data.select_dtypes(include=['object']).columns.difference([y_var]))
							
							numerical_columns_new = data.columns.difference(data.select_dtypes(include=['object']).columns)
							
							encoded_feature_names_new = list(encoded_categorical_columns_new) + list(numerical_columns_new)
							

							df_new_encoded = pd.DataFrame(df_new_encoded, columns=encoded_feature_names_new, index=data.index)
							
							# df_new_encoded[data] = data[data]
							# data = df_new_encoded
							# Drop the original columns used for encoding in new data
							data = data.drop(columns=data.select_dtypes(include=['object']).columns).join(df_new_encoded[encoded_categorical_columns_new])
							st.dataframe(data)

						else:
							st.write("No file named 'apply_encoding_on_df.pkl' found in the current directory.")
					except Exception as e:
						print("e24", e)

				elif act == 'Apply Scaling :green[(Full DF)]':

					try:
						apply_scaling_on_df_file_path = os.path.join(cwd, "apply_scaling_on_df.pkl")
						if os.path.isfile(apply_scaling_on_df_file_path):
							st.write(f"File 'apply_scaling_on_df.pkl' found at {apply_scaling_on_df_file_path}")
							apply_scaling_on_df_pipe = joblib.load(apply_scaling_on_df_file_path)
							st.code(apply_scaling_on_df_pipe)

							try:
								# print(data.info())
								data = apply_scaling_on_df(data, apply_scaling_on_df_pipe)
								st.dataframe(data)
								# gg = st.session_state.df_pre
								# gg_copy = gg.copy()
								# y_var = st.session_state.y_var
								# gg_copy = gg_copy.drop(columns=[y_var])
								# data = data[gg_copy.columns]
								# st.dataframe(data)
							except Exception as e:
								print("e26", e)
						else:
							st.write("No file named 'apply_scaling_on_df.pkl' found in the current directory.")
					except Exception as e:
						print("e26", e)
					try:
						gg = st.session_state.df_pre
						gg_copy = gg.copy()
						y_var = st.session_state.y_var
						gg_copy = gg_copy.drop(columns=[y_var])
						data = data[gg_copy.columns]
						data = data.reset_index(drop=True)
					except:
						pass
			# st.write(data)
		elif choice == "Use Validation Data":
			data = X_test0
			data = data.reset_index(drop=True)
			# st.write(data)

	elif type_ == 'Manual Input':
		# st.warning("hi1")
		if "random_row" in st.session_state:
			random_row = st.session_state.random_row
			# st.warning("hi1")
		# if random_row is None:
		else:
			random_row = dfo.sample(1).iloc[0]
			st.session_state.random_row = random_row
			# st.session_state.random_row = random_row

		# Detect the data type and range of each column
		column_details = {}
		for column in dfo.columns:
			dtype = dfo[column].dtype
			
			if np.issubdtype(dtype, np.number):
				min_value = dfo[column].min()
				max_value = dfo[column].max()
				default_value = random_row[column]
				column_details[column] = {
					"type": "numeric",
					"min": min_value,
					"max": max_value,
					"default": default_value,
				}
			
			# Change from np.object to object
			elif np.issubdtype(dtype, object):
				unique_values = dfo[column].unique()
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
		if "input_keys" not in st.session_state:
			st.session_state.input_keys = {}

		st.write("Dynamic Input Form")
		user_inputs = {}

		for column, details in column_details.items():
			if column not in st.session_state.input_keys:
				unique_key = f"{column}_{np.random.uniform(-1e10, 1e10)}"
				st.session_state.input_keys[column] = unique_key
			else:
				unique_key = st.session_state.input_keys[column]
			
			if details["type"] == "numeric":
				user_inputs[column] = st.number_input(
					f"Enter a value for {column}",
					min_value=details["min"],
					max_value=details["max"],
					key=unique_key,
					value=details["default"]
				)
			elif details["type"] == "categorical":
				user_inputs[column] = st.selectbox(
					f"Select a value for {column}",
					options=details["options"],
					index=list(details["options"]).index(details["default"]),
					key=unique_key
				)
			elif details["type"] == "boolean":
				user_inputs[column] = st.checkbox(
					f"Toggle {column}",
					value=details["default"],
					key=unique_key
				)

		# shuffle_key = f"shuffle_button_{np.random.uniform(-1e10, 1e10)}"
		# shuffle = st.button("Shuffle", key=shuffle_key)

		# if shuffle:
		# 	# Clear input keys to regenerate unique keys
		# 	# if "random_row" in st.session_state:
		# 	del st.session_state.random_row
		# 	del st.session_state.input_keys
			# st.experimental_rerun()
		# Output the collected inputs
		st.write("Your Inputs:")
		st.write(user_inputs)
		dota = pd.DataFrame([user_inputs])
		st.write("lk")
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
						y_var = st.session_state.y_var
						numeric_columns = data.select_dtypes(include=np.number).columns.difference([y_var]).tolist()
						# Transform numerical columns
						data[numeric_columns] = treat_missing_vaues_in_full_df_pipe.transform(data[numeric_columns])
						# data = df_ccopy
						st.dataframe(data)
						# data = treat_missing_values_in_full_df(data, treat_missing_vaues_in_full_df_pipe)
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

						y_var = st.session_state.y_var
						categorical_columns = data.select_dtypes(include=['object']).columns.difference([y_var]).tolist()
						# Transform numerical columns
						data[categorical_columns] = treat_missing_vaues_in_full_df_pipe.transform(data[categorical_columns])					
						st.dataframe(data)
						# numeric_columns = df_ccopy.select_dtypes(include=np.number).columns.difference([y_var]).tolist()
						# # Transform numerical columns
						# df_ccopy[numeric_columns] = treat_missing_vaues_in_full_df_pipe.transform(df_ccopy[numeric_columns])
						# data = df_ccopy
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

								# Reapply the original data types
								for col in data.columns:
									data[col] = data[col].astype(data[col].dtype)

								# Reorder columns to match the original DataFrame
								data = data[data.columns]

								# df_unseen_winsorized = outliers_for_col_pipe.transform(data)

								# data = pd.DataFrame(df_unseen_winsorized, columns=data.columns)
								st.dataframe(data)
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
					y_var = st.session_state.y_var

					numeric_columns = df.select_dtypes(include=['number']).columns.difference([y_var]).tolist()
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
			
			
			# apply_transformation_on_col_file_path = os.path.join(cwd, "apply_transformation_on_col.pkl")
			# if os.path.isfile(apply_transformation_on_col_file_path):
			#     st.write(f"File 'apply_transformation_on_col.pkl' found at {apply_transformation_on_col_file_path}")
			#     apply_transformation_on_col_pipe = joblib.load(apply_transformation_on_col_file_path)
			#     st.write(apply_transformation_on_col_pipe)
			# else:
			#     st.write("No file named 'apply_transformation_on_col.pkl' found in the current directory.")
			elif act == 'Column Unique Value Replacement :orange[(For Col)]':
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
								st.dataframe(data)
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
						y_var = st.session_state.y_var
						df_new_encoded = apply_encoding_on_df_pipe.transform(data)
						data_ = data.copy()
						# Create a DataFrame with the encoded columns for new data
						encoded_categorical_columns_new = apply_encoding_on_df_pipe.named_steps['encoder'].transformers_[0][1].get_feature_names_out(data.select_dtypes(include=['object']).columns.difference([y_var]))
						
						numerical_columns_new = data.columns.difference(data.select_dtypes(include=['object']).columns)
						
						encoded_feature_names_new = list(encoded_categorical_columns_new) + list(numerical_columns_new)
						

						df_new_encoded = pd.DataFrame(df_new_encoded, columns=encoded_feature_names_new, index=data.index)
						
						# df_new_encoded[data] = data[data]
						# data = df_new_encoded
						# Drop the original columns used for encoding in new data
						data = data.drop(columns=data.select_dtypes(include=['object']).columns).join(df_new_encoded[encoded_categorical_columns_new])
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
								st.dataframe(data)
							except Exception as e:
								print("e27", e)
							st.dataframe(data)
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
		dota = data
		try:
			st.warning("This feature is in the beta phase and has not yet been fully released.")
			dota = dota.reset_index(drop=True)
		except:
			pass
	

	
	# s_ = st.color_picker("Pick_")
	# s = st.color_picker("Pick")
	
	if st.button("Predict", type="primary", use_container_width=True):
		# result = predict(data, user, pw, db)
		try:
			if type_ == 'Upload File':
				
				y_var = st.session_state.y_var
				result, predictions = predict(data, model_pipe, y_var)

				if choice == "Upload '.csv' or '.xlsx'":
					try:
						if opt_ == True:
							

							cm = sns.light_palette("tomato", as_cmap = True)

							# # Function to apply background color to specific cells
							# def highlight_machine_failure(val):
							#     color = "#FF774B" if val == "Machine_Failure" else ""
							#     return f"background-color: {color}"
							# styled_result = result.style.applymap(highlight_machine_failure)

							# Apply the custom styling to the DataFrame
							styled_result = result.style
							# st.table(result.style.background_gradient(cmap = cm).set_precision(2))
							# Display the styled DataFrame using streamlit
							st.table(styled_result.background_gradient(cmap = cm).format(precision=2))
						else:
							# Combine predictions with the original data 
							final_result = pd.concat([predictions, _data__], axis=1)
							# Create a color map using Seaborn
							cm = sns.light_palette("tomato", as_cmap=True)

							# Apply the color map to the DataFrame
							styled_df = final_result.style.background_gradient(cmap=cm)

							# Display the styled DataFrame using st.dataframe
							# st.dataframe(styled_df)                                 # IM
							# Define styles
							# Apply styles and highlight
							# s = st.color_picker("Pick")
							
							styled_df_info1 = (
								final_result.style.set_table_styles(
									[
										{'selector': 'th', 'props': 'background-color: #15296B; color: white;'},  # Table headers
										{'selector': 'td', 'props': f'background-color: #FF4B4B; border: 2px solid black;'},  # Table cells
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
					except Exception as e:
						st.dataframe(result)
				else:

					# Combine predictions with the original data 
					final_result = pd.concat([predictions, data], axis=1)
					# Create a color map using Seaborn
					cm = sns.light_palette("tomato", as_cmap=True)

					# Apply the color map to the DataFrame
					styled_df = final_result.style.background_gradient(cmap=cm)

					# Display the styled DataFrame using st.dataframe
					# st.dataframe(styled_df)                                 # IM
					# Define styles
					# Apply styles and highlight
					# s = st.color_picker("Pick")
					
					styled_df_info1 = (
						final_result.style.set_table_styles(
							[
								{'selector': 'th', 'props': 'background-color: #15296B; color: white;'},  # Table headers
								{'selector': 'td', 'props': f'background-color: #FF4B4B; border: px solid black;'},  # Table cells
								{'selector': 'tr', 'props': 'border: 1px solid black;'}  # Table rows
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
				y_var = st.session_state.y_var
				st.dataframe(dota)
				result, predictions = predict(dota, model_pipe, y_var)
				
				# Combine predictions with the original data 
				final_result = pd.concat([predictions, dota], axis=1)

				cm = sns.light_palette("tomato", as_cmap = True)
				pd.set_option("styler.render.max_elements", 5000000000000000000000000000)  # Adjust this to a value greater than your cell count
				styled_result = result.style
				# Define color maps for the gradients
				tomato_cm = sns.light_palette("tomato", as_cmap=True)
				gold_cm = sns.light_palette("gold", as_cmap=True)

				# Display the styled table in Streamlit
				# st.table(styled_result.format(precision=2))
				st.table(styled_result.background_gradient(cmap = cm).format(precision=2))


				st.success("Success ✅")
				st.warning("This feature is in the beta phase and has not yet been fully released.")
				st.success("Comming Soon 🔜 @ https://pypi.org/project/glook/")
		except UnboundLocalError:
			st.warning("Upload .csv or .xlsx")
		except Exception as e:
			print(e)
			# st.error(e)
		
if __name__=='__main__':
	main()

