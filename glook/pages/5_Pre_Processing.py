import streamlit as st
st.set_page_config(page_title='ML-Automation', page_icon='🏗️', layout=
	'wide', initial_sidebar_state='expanded')
print(1)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import statsmodels.api as sm
# st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_option('deprecation.showPyplotGlobalUse', False)
print(2)
from scipy.stats.mstats import winsorize
from sklearn.impute import SimpleImputer
from scipy.stats import boxcox, yeojohnson
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, KBinsDiscretizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import joblib
from sklearn.pipeline import Pipeline
from feature_engine.outliers import Winsorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import traceback

Confirm_Change = False


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


def change_dtype(d, column_name, new_dtype):
	try:
		df = d.copy()
		df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
		df[column_name] = df[column_name].astype(new_dtype)
		st.warning(f"Data type of column '{column_name}' changed to {new_dtype}, but these changes won't be saved until you press the 'Confirm Changes' button below.")
		print(3)
		return df
	except Exception as e:
		st.warning('Select Proper Column')
		print(4)
		return d
		# st.error(f'Error occurred: {e}')
		print(5)


def drop_column(df, column_name):
	try:
		df.drop(column_name, axis=1, inplace=True)
		print(6)
		st.warning(
			f"Column '{column_name}' has been dropped, but changes won't be saved until you press the 'Confirm Changes' button below."
			)
		print(7)
	except Exception as e:
		st.error(f'Error occurred: {e}')
		print(8)


def drop_duplicates(df):
	try:
		original_rows = len(df)
		df.drop_duplicates(inplace=True)
		print(9)
		new_rows = len(df)
		st.warning(
			f"Dropped {original_rows - new_rows} duplicate rows, but changes will not be saved until you press the 'Confirm Changes' button below."
			)
		print(10)
	except Exception as e:
		st.error(f'Error occurred: {e}')
		print(11)


def create_full_df_impute_pipelines(modified_df, numeric_treatment, categorical_treatment):
	try:
		# Define numerical and categorical columns
		numeric_columns = modified_df.select_dtypes(include=np.number).columns.tolist()
		categorical_columns = modified_df.select_dtypes(include=['object']).columns.tolist()
		st.session_state.unsup_num_col = numeric_columns
		st.session_state.unsup_cat_col = categorical_columns
		# Numerical pipeline
		numeric_pipeline = None
		if numeric_treatment == 'Mean':
			numeric_pipeline = Pipeline(steps=[
				('imputer', SimpleImputer(strategy='mean'))
			])
		elif numeric_treatment == 'Median':
			numeric_pipeline = Pipeline(steps=[
				('imputer', SimpleImputer(strategy='median'))
			])
		elif numeric_treatment == 'Mode':
			numeric_pipeline = Pipeline(steps=[
				('imputer', SimpleImputer(strategy='most_frequent'))
			])
		elif numeric_treatment == 'Random':
			class RandomImputer:
				def __init__(self):
					pass
				
				def fit(self, X, y=None):
					return self
				
				def transform(self, X):
					X = X.copy()
					for col in X.columns:
						nan_mask = X[col].isnull()
						X.loc[nan_mask, col] = np.random.choice(X[col].dropna(), nan_mask.sum())
					return X
			
			numeric_pipeline = Pipeline(steps=[
				('imputer', RandomImputer())
			])

		# Apply numerical pipeline
		if numeric_pipeline:
			modified_df[numeric_columns] = numeric_pipeline.fit_transform(modified_df[numeric_columns])

		# Categorical pipeline
		categorical_pipeline = None
		if categorical_treatment == 'Mode':
			categorical_pipeline = Pipeline(steps=[
				('imputer', SimpleImputer(strategy='most_frequent'))
			])
		elif categorical_treatment == 'Random':
			class RandomImputer:
				def __init__(self):
					pass
				
				def fit(self, X, y=None):
					return self
				
				def transform(self, X):
					X = X.copy()
					for col in X.columns:
						nan_mask = X[col].isnull()
						X.loc[nan_mask, col] = np.random.choice(X[col].dropna(), nan_mask.sum())
					return X
			
			categorical_pipeline = Pipeline(steps=[
				('imputer', RandomImputer())
			])

		# Apply categorical pipeline
		if categorical_pipeline:
			modified_df[categorical_columns] = categorical_pipeline.fit_transform(modified_df[categorical_columns])
	except Exception as e:
		print("e_+;", e)
		return modified_df, None

	return modified_df, (numeric_pipeline, categorical_pipeline)

def create_full_df_impute_pipelines_for_sup(modified_df, y_col, numeric_treatment, categorical_treatment):
	try:
	# Define numerical and categorical columns excluding the Y column
		numeric_columns = modified_df.select_dtypes(include=np.number).columns.difference([y_col]).tolist()
		categorical_columns = modified_df.select_dtypes(include=['object']).columns.difference([y_col]).tolist()

		# Numerical pipeline
		numeric_pipeline = None
		categorical_pipeline = None
		if numeric_treatment == 'Mean':
			numeric_pipeline = Pipeline(steps=[
				('imputer', SimpleImputer(strategy='mean'))
			])
		elif numeric_treatment == 'Median':
			numeric_pipeline = Pipeline(steps=[
				('imputer', SimpleImputer(strategy='median'))
			])
		elif numeric_treatment == 'Mode':
			numeric_pipeline = Pipeline(steps=[
				('imputer', SimpleImputer(strategy='most_frequent'))
			])
		elif numeric_treatment == 'Random':
			class RandomImputer:
				def __init__(self):
					pass

				def fit(self, X, y=None):
					return self

				def transform(self, X):
					X = X.copy()
					for col in X.columns:
						nan_mask = X[col].isnull()
						X.loc[nan_mask, col] = np.random.choice(X[col].dropna(), nan_mask.sum())
					return X

			numeric_pipeline = Pipeline(steps=[
				('imputer', RandomImputer())
			])

		# Apply numerical pipeline
		if numeric_pipeline:
			modified_df[numeric_columns] = numeric_pipeline.fit_transform(modified_df[numeric_columns])

		# Categorical pipeline
		categorical_pipeline = None
		if categorical_treatment == 'Mode':
			categorical_pipeline = Pipeline(steps=[
				('imputer', SimpleImputer(strategy='most_frequent'))
			])
		elif categorical_treatment == 'Random':
			class RandomImputer:
				def __init__(self):
					pass

				def fit(self, X, y=None):
					return self

				def transform(self, X):
					X = X.copy()
					for col in X.columns:
						nan_mask = X[col].isnull()
						X.loc[nan_mask, col] = np.random.choice(X[col].dropna(), nan_mask.sum())
					return X

			categorical_pipeline = Pipeline(steps=[
				('imputer', RandomImputer())
			])

		# Apply categorical pipeline
		if categorical_pipeline:
			try:
				modified_df[categorical_columns] = categorical_pipeline.fit_transform(modified_df[categorical_columns])
			except Exception as e:
				print("e:_-:", e)
				return modified_df, numeric_pipeline, categorical_pipeline
	except Exception as e:
		print("e:_-+:", e)
		return modified_df, numeric_pipeline, categorical_pipeline
	return modified_df, numeric_pipeline, categorical_pipeline

# def treat_outliers(df, column_name, method, z_score_threshold, lower_limit, upper_limit):
# 	try:
# 		if method == 'Delete Outliers':
# 			z_scores = np.abs((df[column_name] - df[column_name].mean()) /
# 				df[column_name].std())
# 			df = df[z_scores < float(z_score_threshold)]
# 			st.warning(
# 				"Outliers deleted. Changes will not be saved until you press the 'Confirm Changes' button below."
# 				)
# 			print(12)
# 			return df
# 		elif method == 'Winsorization':
# 			modified_column = winsorize(df[column_name], limits=[
# 				lower_limit, upper_limit])
# 			df[column_name] = modified_column
# 			st.warning(
# 				"Outliers have been treated using Winsorization, but changes will not be saved until the 'Confirm Changes' button is pressed."
# 				)
# 			print(13)
# 			return df
# 	except Exception as e:
# 		print(e)
# 		print(14)
# 		st.warning('Select Proper Numerical Column')
# 		print(15)
# 		return df

# def treat_outliers()
# def apply_winsorization_col(df, column_name):
# 	# Apply Winsorization to the column
# 	numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
# 	df_copy = df
# 	winsor = Winsorizer(capping_method='iqr', 
# 						tail='both',
# 						fold=1.5,
# 						variables=[column_name])

# 	# Creating pipelines for winsor
# 	outlier_pipeline = Pipeline(steps=[('winsor', winsor)])

# 	# Creating ColumnTransformer for winsorization
# 	# Pass the column index or name
# 	preprocessor = ColumnTransformer(transformers=[('wins', outlier_pipeline, [column_name])], remainder='passthrough')

# 	# Fitting and transforming the winsorization
# 	df_winsorized = preprocessor.fit_transform(df_copy)

# 	# Convert back to DataFrame
# 	df_winsorized = pd.DataFrame(df_winsorized, columns=[column_name] + [col for col in df_copy.columns if col != column_name])

# 	df_winsorized[column_name] = pd.to_numeric(df_winsorized[column_name])

# 	# Reorder columns to match original DataFrame
# 	df_winsorized = df_winsorized[df_copy.columns]
# 	# Convert all numeric columns to numeric (in case of any changes)
# 	df_winsorized[numeric_columns] = df_winsorized[numeric_columns].apply(pd.to_numeric)
# 	return preprocessor, df_winsorized

# def apply_winsorization_col(df, column_name):
# 	preprocessor = None 
# 	try:
# 		# numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
# 		df_copy = df
# 		# Apply Winsorization to the column
# 		winsor = Winsorizer(capping_method='iqr', 
# 							tail='both',
# 							fold=1.5,
# 							variables=[column_name])

# 		# Creating pipelines for winsor
# 		outlier_pipeline = Pipeline(steps=[('winsor', winsor)])

# 		# Creating ColumnTransformer for winsorization
# 		# Pass the column index or name
# 		preprocessor = ColumnTransformer(transformers=[('wins', outlier_pipeline, [column_name])], remainder='passthrough')

# 		# Fitting and transforming the winsorization
# 		df_winsorized = preprocessor.fit_transform(df_copy)

# 		# Convert back to DataFrame
# 		df_winsorized = pd.DataFrame(df_winsorized, columns=df_copy.columns)

# 		# Reorder columns to match original DataFrame
# 		df_winsorized = df_winsorized[df_copy.columns]
# 		df_winsorized[column_name] = pd.to_numeric(df_winsorized[column_name])
# 		# Convert all numeric columns to numeric (in case of any changes)
# 		# df_winsorized[numeric_columns] = df_winsorized[numeric_columns].apply(pd.to_numeric)
# 	except Exception as e:
# 		print("e:9:", e)
# 		return preprocessor, df
# 	return preprocessor, df_winsorized
def apply_winsorization_col(df, column_name, capping_method, tail, fold):
	preprocessor = None
	try:
		# Store the original data types
		original_dtypes = df.dtypes

		# Apply Winsorization to the column
		winsor = Winsorizer(capping_method=capping_method, 
							tail=tail,
							fold=fold,
							variables=[column_name])

		# Creating pipelines for winsor
		outlier_pipeline = Pipeline(steps=[('winsor', winsor)])

		# Creating ColumnTransformer for winsorization
		preprocessor = ColumnTransformer(transformers=[('wins', outlier_pipeline, [column_name])], remainder='passthrough')

		# Fitting and transforming the winsorization
		df_winsorized = preprocessor.fit_transform(df)

		# Convert back to DataFrame
		transformed_columns = preprocessor.transformers_[0][2] + [col for col in df.columns if col not in column_name]
		df_winsorized = pd.DataFrame(df_winsorized, columns=transformed_columns)

		# Reapply the original data types
		for col in df_winsorized.columns:
			df_winsorized[col] = df_winsorized[col].astype(original_dtypes[col])

		# Reorder columns to match the original DataFrame
		df_winsorized = df_winsorized[df.columns]
		st.warning(f"Winsorization ({capping_method}) applied on `{column_name}`, but changes will not be saved until you press the 'Confirm Changes' button below.")
	except Exception as e:
		print("e:9:", e)
		st.error(e)
		return preprocessor, df

	return preprocessor, df_winsorized


def apply_winsorization_col_sup(df, column_name, y_col, capping_method, tail, fold):
	preprocessor = None
	try:
		# Store the original data types
		original_dtypes = df.dtypes

		# Apply Winsorization to the specified column excluding the y_col
		if column_name == y_col:
			raise ValueError(f"Column '{column_name}' should not be the same as the Y column '{y_col}'.")

		winsor = Winsorizer(capping_method=capping_method, 
							tail=tail,
							fold=fold,
							variables=[column_name])

		# Creating pipelines for winsor
		outlier_pipeline = Pipeline(steps=[('winsor', winsor)])

		# Creating ColumnTransformer for winsorization
		preprocessor = ColumnTransformer(transformers=[('wins', outlier_pipeline, [column_name])], remainder='passthrough')

		# Fitting and transforming the winsorization
		df_winsorized = preprocessor.fit_transform(df.drop(columns=[y_col]))
		st.warning(f"Winsorization ({capping_method}) applied on full df, but changes will not be saved until you press the 'Confirm Changes' button below.")
		# Convert back to DataFrame with proper column names and order
		winsorized_columns = [column_name] + [col for col in df.columns if col not in [column_name, y_col]]
		df_winsorized = pd.DataFrame(df_winsorized, columns=winsorized_columns)

		# Add the y_col back to the DataFrame
		df_winsorized[y_col] = df[y_col].values

		# Reapply the original data types
		for col in df_winsorized.columns:
			df_winsorized[col] = df_winsorized[col].astype(original_dtypes[col])

		# Reorder columns to match the original DataFrame
		df_winsorized = df_winsorized[df.columns]
	except Exception as e:
		print("Exception occurred:", e)
		return preprocessor, df

	return preprocessor, df_winsorized


# def treat_outliers_full_df(df, method, z_score_threshold=3, lower_limit=
# 	0.05, upper_limit=0.05):
# 	"""
# 	Treat outliers in a given DataFrame.

# 	:param df: DataFrame to process
# 	:param method: Method to use for outlier treatment ("Delete Outliers", "Winsorization")
# 	:param z_score_threshold: Z-score threshold for identifying outliers (used for "Delete Outliers")
# 	:param lower_limit: Lower limit for Winsorization
# 	:param upper_limit: Upper limit for Winsorization
# 	:return: DataFrame with outliers treated
# 	"""
# 	print(16)
# 	try:
# 		numerical_columns = df.select_dtypes(include=[np.number]
# 			).columns.tolist()
# 		if method == 'Delete Outliers':
# 			for column_name in numerical_columns:
# 				z_scores = np.abs((df[column_name] - df[column_name].mean()
# 					) / df[column_name].std())
# 				df = df[z_scores < float(z_score_threshold)]
# 			st.warning(
# 				"Outliers deleted. Changes will not be saved until you press the 'Confirm Changes' button below."
# 				)
# 			print(17)
# 			return df
# 		elif method == 'Winsorization':
# 			for column_name in numerical_columns:
# 				df[column_name] = winsorize(df[column_name], limits=[
# 					lower_limit, upper_limit])
# 			st.warning(
# 				"Outliers have been treated using Winsorization, but changes will not be saved until the 'Confirm Changes' button is pressed."
# 				)
# 			print(18)
# 			return df
# 		elif method == 'IQR':
#         # Apply IQR method to all numerical columns
# 			for column_name in numerical_columns:
# 				# Calculate IQR
# 				Q1 = df[column_name].quantile(0.25)  # First quartile
# 				Q3 = df[column_name].quantile(0.75)  # Third quartile
# 				IQR = Q3 - Q1  # Interquartile range
				
# 				# Define lower and upper bounds for outliers
# 				lower_bound = Q1 - 1.5 * IQR
# 				upper_bound = Q3 + 1.5 * IQR
				
# 				# Cap values outside these bounds
# 				df[column_name] = np.where(df[column_name] < lower_bound, lower_bound, df[column_name])
# 				df[column_name] = np.where(df[column_name] > upper_bound, upper_bound, df[column_name])
			
# 			st.warning("Outliers have been treated using the IQR method, but changes will not be saved until the 'Confirm Changes' button is pressed.")
# 			print(19)
		
# 			return df
# 		else:
# 			st.error('Invalid method specified for treating outliers.')
# 			print(19)
# 			return df
# 	except Exception as e:
# 		st.warning(f'Error occurred: {e}')
# 		print(20)
# 		return df

# def apply_winsorization_for_full_df(df):
# 	# Get numerical columns
# 	df_num = df.select_dtypes(include=['number'])
# 	numeric_columns = df_num.columns.tolist()

# 	# Apply Winsorization to all numeric columns
# 	winsor = Winsorizer(capping_method='iqr', 
# 						tail='both',
# 						fold=1.5,
# 						variables=numeric_columns)

# 	# Creating pipeline for winsor
# 	outlier_pipeline = Pipeline(steps=[('winsor', winsor)])

# 	# Creating ColumnTransformer for winsorization
# 	preprocessor = ColumnTransformer(transformers=[('wins', outlier_pipeline, numeric_columns)], remainder='passthrough')

# 	# Fit and transform the winsorization on the specified column
# 	df_transformed = preprocessor.fit_transform(df)

# 	# df_transformed = pd.to_numeric(df_transformed[numeric_columns])

# 	# Convert the transformed array back to a DataFrame
# 	df_transformed = pd.DataFrame(df_transformed, columns=df.columns)


# 	return preprocessor, df_transformed

def apply_winsorization_for_full_df_unsup(df, capping_method, tail, fold):
	winsor = None
	try:
		# Make a copy of the DataFrame to preserve original data and column order
		df_copy = df.copy()

		# Get numerical columns
		numeric_columns = df.select_dtypes(include=['number']).columns.tolist()

		# Apply Winsorization to all numeric columns
		winsor = Winsorizer(capping_method=capping_method, 
							tail=tail,
							fold=fold,
							variables=numeric_columns,)

		# Fit and transform the winsorization on the specified column
		df_transformed = winsor.fit_transform(df[numeric_columns])
		st.warning(f"Winsorization ({capping_method}) applied on full df, but changes will not be saved until you press the 'Confirm Changes' button below.")
		# Convert the transformed array back to a DataFrame
		df_transformed = pd.DataFrame(df_transformed, columns=numeric_columns)

		# Replace original numeric columns with transformed ones
		df_copy[numeric_columns] = df_transformed
		return winsor, df_copy
	except Exception as e:
		print("e:l", e)
		st.error(e)
		return winsor, df

# 	return winsor, df_copy
def apply_winsorization_for_full_df(df, y_col, capping_method, tail, fold):
	winsor = None
	try:
		# Make a copy of the DataFrame to preserve original data and column order
		df_copy = df.copy()

		# Get numerical columns excluding the Y column
		numeric_columns = df.select_dtypes(include=['number']).columns.difference([y_col]).tolist()

		# Apply Winsorization to all numeric columns except the Y column
		winsor = Winsorizer(capping_method=capping_method, 
							tail=tail,
							fold=fold,
							variables=numeric_columns)

		# Fit and transform the winsorization on the specified columns
		df_transformed = winsor.fit_transform(df[numeric_columns])

		st.warning(f"Winsorization ({capping_method}) applied on full df except '{y_col}', but changes will not be saved until you press the 'Confirm Changes' button below.")

		# Convert the transformed array back to a DataFrame
		df_transformed = pd.DataFrame(df_transformed, columns=numeric_columns)

		# Replace original numeric columns with transformed ones
		df_copy[numeric_columns] = df_transformed
	except Exception as e:
		print("Exception occurred:", e)
		st.error(e)
		return winsor, df

	return winsor, df_copy


def treat_missing_values(df, column_name, method):
	try:
		if method == 'Delete Missing Values':
			df.dropna(subset=[column_name], inplace=True)
			print(21)
			st.warning(
				"Missing values have been deleted, but changes will not be saved until you press the 'Confirm Changes' button below."
				)
			print(22)
			return df
		elif method == 'Mean Imputation':
			Imputer = Pipeline(steps=[
				('imputer', SimpleImputer(strategy='median'))
			])

			df[column_name] = Imputer.fit_transform(df[[column_name]]).ravel()
			# df[column_name] = pd.to_numeric(df[column_name])
			# imputer = SimpleImputer(strategy='mean')
			# df[column_name] = imputer.fit_transform(df[[column_name]])
			st.warning(
				"Missing values have been imputed using the mean, but changes will not be saved until you press the 'Confirm Changes' button below."
				)
			print(23)
			return df, Imputer
		elif method == 'Median Imputation':
			Imputer = Pipeline(steps=[
				('imputer', SimpleImputer(strategy='median'))
			])

			df[column_name] = Imputer.fit_transform(df[[column_name]]).ravel()
			# imputer = SimpleImputer(strategy='median')
			# df[column_name] = imputer.fit_transform(df[[column_name]])
			st.warning(
				"Missing values have been imputed using the median, but changes will not be saved until you press the 'Confirm Changes' button below."
				)
			print(24)
			return df, Imputer
		elif method == 'Mode Imputation':
			Imputer = Pipeline(steps=[
				('imputer', SimpleImputer(strategy='most_frequent'))
			])

			df[column_name] = Imputer.fit_transform(df[[column_name]]).ravel()
			# imputer = SimpleImputer(strategy='most_frequent')
			# df[column_name] = col_categorical_transformer.fit_transform(df[[column_name]])
			
			st.warning(
				"Missing values have been imputed using the mode, but changes will not be saved until you press the 'Confirm Changes' button below."
				)
			print(25)
			return df, Imputer
	except Exception as e:
		st.error(f'Error occurred: {e}')
		print(26)
		return df, None


def apply_transformation(df, column_name, method):
	try:
		if method == 'Log Transformation':
			df[column_name] = np.log1p(df[column_name])
			pipe = method
			st.warning(
				"Log transformation applied, but changes will not be saved until you press the 'Confirm Changes' button below."
				)
			print(27)
			return df, pipe
		elif method == 'Exponential Transformation':
			df[column_name] = np.exp(df[column_name])
			pipe = method
			st.warning(
				"Exponential transformation applied, but changes will not be saved until you press the 'Confirm Changes' button below."
				)
			print(28)
			return df, pipe
		elif method == 'Square Root Transformation':
			df[column_name] = np.sqrt(df[column_name])
			pipe = method
			st.warning(
				"Square root transformation applied, but changes will not be saved until you press the 'Confirm Changes' button below."
				)
			print(29)
			return df, pipe
		# elif method == 'Box-Cox Transformation':
		# 	transformed_data, _ = boxcox(df[column_name] + 1)
		# 	df[column_name] = transformed_data
		# 	st.warning(
		# 		"Box-Cox transformation applied, but changes will not be saved until you press the 'Confirm Changes' button below."
		# 		)
		# 	print(30)
		# 	return df
		# elif method == 'Yeo-Johnson Transformation':
		# 	# transformed_data, _ = yeojohnson(df[column_name] + 1)
		# 	# Yeo-Johnson transformation function
		# 	# Pipeline for Yeo-Johnson transformation
		# 	# Yeo-Johnson transformation function
		# 	def yeojohnson_transform(X):
		# 		transformed, _ = yeojohnson(X)
		# 		return transformed.reshape(-1, 1)  # Reshape to ensure correct format for pipeline

		# 	# Pipeline for Yeo-Johnson transformation
		# 	yeojohnson_pipeline = Pipeline(steps=[
		# 		('yeojohnson', FunctionTransformer(func=yeojohnson_transform))
		# 	])

		# 	# Apply Yeo-Johnson transformation to specific column
		# 	df_transformed = df.copy()  # Create a copy of the original DataFrame

		# 	# Transform the specified column
		# 	df_transformed[column_name] = yeojohnson_pipeline.fit_transform(df[[column_name]])
		# 	# df[column_name] = transformed_data
		# 	st.warning(
		# 		"Yeo-Johnson transformation applied, but changes will not be saved until you press the 'Confirm Changes' button below."
		# 		)
		# 	print(31)
		# 	return df_transformed, yeojohnson_pipeline
	except Exception as e:
		st.warning('Select Proper Column')
		print(32)
		st.error(f'Error occurred: {e}')
		print(33)
		return df, None

# def create_dummy_variables(df, method, opt):
def create_dummy_variables(df, drop='first'):
	"""
	Encode all categorical columns in the DataFrame using one-hot encoding
	and return the modified DataFrame and the encoder pipeline.

	Parameters:
	df (pd.DataFrame): The input DataFrame.
	drop (str): The parameter to drop for OneHotEncoder. Default is 'first'.

	Returns:
	pd.DataFrame: The modified DataFrame with encoded columns.
	Pipeline: The encoder pipeline used for the transformation.
	"""
	try:
		# Identify categorical columns
		categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

		# Create a Pipeline for one-hot encoding all categorical columns
		encoder_pipeline = Pipeline(steps=[
			('encoder', ColumnTransformer(
				transformers=[
					('onehot', OneHotEncoder(drop=drop, handle_unknown='ignore'), categorical_columns)
				], 
				remainder='passthrough'))
		])

		# Fit and transform the encoder on the DataFrame
		df_encoded = encoder_pipeline.fit_transform(df)

		# Get feature names for the transformed columns
		encoded_categorical_columns = encoder_pipeline.named_steps['encoder'].transformers_[0][1].get_feature_names_out(categorical_columns)
		numerical_columns = df.columns.difference(categorical_columns)
		encoded_feature_names = list(encoded_categorical_columns) + list(numerical_columns)

		# Create a DataFrame with the encoded columns
		df_encoded = pd.DataFrame(df_encoded, columns=encoded_feature_names, index=df.index)

		# Drop the original columns used for encoding
		df_final = df.drop(columns=categorical_columns).join(df_encoded[encoded_categorical_columns])

		return df_final, encoder_pipeline
	except Exception as e:
		print("e:_-:", e)
		return df, None


def create_dummy_variables_for_sup(df, y_col, drop='first'):
	"""
	Encode all categorical columns in the DataFrame using one-hot encoding,
	excluding the specified Y column, and return the modified DataFrame and the encoder pipeline.

	Parameters:
	df (pd.DataFrame): The input DataFrame.
	y_col (str): The name of the Y column to be excluded from encoding.
	drop (str): The parameter to drop for OneHotEncoder. Default is 'first'.

	Returns:
	pd.DataFrame: The modified DataFrame with encoded columns.
	Pipeline: The encoder pipeline used for the transformation.
	"""
	# encoder_pipeline = 
	try:
		# Identify categorical columns excluding the Y column
		categorical_columns = df.select_dtypes(include=['object']).columns.difference([y_col]).tolist()

		df[categorical_columns] = df[categorical_columns].astype(str)

		# Create a Pipeline for one-hot encoding all categorical columns
		encoder_pipeline = Pipeline(steps=[
			('encoder', ColumnTransformer(
				transformers=[
					('onehot', OneHotEncoder(drop=drop, handle_unknown='ignore'), categorical_columns)
				], 
				remainder='passthrough'))
		])
		# st.dataframe(df)
		# Fit and transform the encoder on the DataFrame excluding the Y column
		df_encoded = encoder_pipeline.fit_transform(df.drop(columns=[y_col]))
		
		# Get feature names for the transformed columns
		encoded_categorical_columns = encoder_pipeline.named_steps['encoder'].transformers_[0][1].get_feature_names_out(categorical_columns)
		
		numerical_columns = df.drop(columns=categorical_columns + [y_col]).columns
		
		encoded_feature_names = list(encoded_categorical_columns) + list(numerical_columns)
		
		# Create a DataFrame with the encoded columns
		df_encoded = pd.DataFrame(df_encoded, columns=encoded_feature_names, index=df.index)
		# st.write(df_encoded)
		# Add the Y column back to the DataFrame
		df_encoded[y_col] = df[y_col]
		df_final = df.drop(columns=categorical_columns).join(df_encoded[encoded_categorical_columns])
	except Exception as e:
		print("e---_", e)
		traceback.print_exc()
		return df, None

	return df_final, encoder_pipeline

	# try:
	# 	if method == 'One-Hot Encoding':
			
	# 		cat_columns = df.select_dtypes(include=['object']).columns
	# 		encoded_df = pd.get_dummies(df, columns=cat_columns, drop_first=opt
	# 			)
	# 		encoded_df.replace({(True): 1, (False): 0}, inplace=True)
	# 		print(34)
	# 		st.warning(
	# 			"One-hot encoding applied, but changes will not be saved until you press the 'Confirm Changes' button below."
	# 			)
	# 		print(35)
	# 		return encoded_df
	# 	elif method == 'Label Encoding':
	# 		encoded_df = df.copy()
	# 		label_encoder = LabelEncoder()
	# 		for column in encoded_df.select_dtypes(include=['object']).columns:
	# 			encoded_df[column] = label_encoder.fit_transform(encoded_df
	# 				[column])
	# 		st.warning(
	# 			"Label encoding applied, but changes will not be saved until you press the 'Confirm Changes' button below."
	# 			)
	# 		print(36)
	# 		return encoded_df
	# except Exception as e:
	# 	st.error(f'Error occurred: {e}')
	# 	print(37)
	# 	return df


# def create_dummy_variables_for_col(df, column_name, method, opt):
# 	try:
# 		if method == 'One-Hot Encoding':
# 			encoded_df = pd.get_dummies(df[column_name], prefix=column_name, drop_first=opt)
# 			df = pd.concat([df, encoded_df], axis=1)
# 			df.drop(column_name, axis=1, inplace=True)
# 			print(38)
# 			df.replace({(True): 1, (False): 0}, inplace=True)
# 			print(39)
# 			st.warning(
# 				"One-hot encoding applied, but changes will not be saved until you press the 'Confirm Changes' button below."
# 				)
# 			print(40)
# 			return df
# 		elif method == 'Label Encoding':
# 			label_encoder = LabelEncoder()
# 			df[column_name] = label_encoder.fit_transform(df[column_name])
# 			st.warning(
# 				"Label encoding applied, but changes will not be saved until you press the 'Confirm Changes' button below."
# 				)
# 			print(41)
# 			return df
# 	except Exception as e:
# 		st.error(f'Error occurred: {e}')
# 		print(42)
# 		return df

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

def create_dummy_variables_for_col(df, column_name, encoding_method='onehot', drop='first'):
	encoder_pipeline = None
	try:
		numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
		df_copy = df
		if encoding_method == 'onehot':
			# Create a Pipeline for one-hot encoding the specific column
			encoder_pipeline = CustomPipeline(steps=[
				('encoder', ColumnTransformer([
					('onehot', OneHotEncoder(drop=drop, handle_unknown='ignore'), [column_name])
				], remainder='passthrough'))
			], encoding_method=encoding_method)
		elif encoding_method == 'label':
			# Create a custom pipeline-friendly LabelEncoder
			encoder_pipeline = CustomPipeline(steps=[
				('encoder', ColumnTransformer([
					('label', LabelEncoderPipelineFriendly(), [column_name])
				], remainder='passthrough'))
			], encoding_method=encoding_method)

		# Fit and transform the encoder on the specified column
		df_encoded = encoder_pipeline.fit_transform(df_copy)

		# Get feature names for the transformed columns
		if encoding_method == 'onehot':
			encoded_categorical_columns = encoder_pipeline.named_steps['encoder'].transformers_[0][1].get_feature_names_out([column_name])
			numerical_columns = df_copy.columns.difference([column_name])
			# Combine feature names for the encoded categorical columns and numerical columns
			encoded_feature_names = list(encoded_categorical_columns) + list(numerical_columns)
		else:
			encoded_categorical_columns = [column_name]
			numerical_columns = df_copy.columns.difference([column_name])
			encoded_feature_names = encoded_categorical_columns + list(numerical_columns)

		# Create a DataFrame with the encoded columns
		df_encoded = pd.DataFrame(df_encoded, columns=encoded_feature_names, index=df_copy.index)

		# Drop the original column used for encoding
		df_final = df_copy.drop(columns=[column_name]).join(df_encoded[encoded_categorical_columns])
		
		df_final[numeric_columns] = df_final[numeric_columns].apply(pd.to_numeric)
		if encoding_method == 'label':
			df_final[column_name] = df_final[column_name].apply(pd.to_numeric)
	except Exception as e:
		print("e:0=", e)
		return df, encoder_pipeline
	return df_final, encoder_pipeline

def create_dummy_variables_for_col_sup(df, column_name, y_col, encoding_method='onehot', drop='first'):
	encoder_pipeline = None
	try:
		# Exclude the Y column from encoding
		columns_to_encode = [col for col in df.columns if col != y_col]

		if encoding_method == 'onehot':
			# Create a Pipeline for one-hot encoding the specific column
			encoder_pipeline = CustomPipeline(steps=[
				('encoder', ColumnTransformer([
					('onehot', OneHotEncoder(drop=drop, handle_unknown='ignore'), [column_name])
				], remainder='passthrough'))
			], encoding_method=encoding_method)
		elif encoding_method == 'label':
			# Create a custom pipeline-friendly LabelEncoder
			encoder_pipeline = CustomPipeline(steps=[
				('encoder', ColumnTransformer([
					('label', LabelEncoderPipelineFriendly(), [column_name])
				], remainder='passthrough'))
			], encoding_method=encoding_method)

		# Fit and transform the encoder on the specified column
		df_encoded = encoder_pipeline.fit_transform(df[columns_to_encode])

		# Get feature names for the transformed columns
		if encoding_method == 'onehot':
			encoded_categorical_columns = encoder_pipeline.named_steps['encoder'].transformers_[0][1].get_feature_names_out([column_name])
			numerical_columns = df.drop(columns=[column_name, y_col]).columns
			encoded_feature_names = list(encoded_categorical_columns) + list(numerical_columns)
		else:
			encoded_categorical_columns = [column_name]
			numerical_columns = df.drop(columns=[column_name, y_col]).columns
			encoded_feature_names = encoded_categorical_columns + list(numerical_columns)

		# Create a DataFrame with the encoded columns
		df_encoded = pd.DataFrame(df_encoded, columns=encoded_feature_names, index=df.index)

		# Drop the original column used for encoding
		df_final = df.drop(columns=[column_name]).join(df_encoded[encoded_categorical_columns])

	except Exception as e:
		print("Exception occurred:", e)
		return df, encoder_pipeline

	return df_final, encoder_pipeline

def apply_scaling(df, method):
	try:
		if method == 'Standardize Scaling':
			scaler = StandardScaler()
			scaled_data = scaler.fit_transform(df)
			scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
			st.warning(
				"Standardize scaling has been applied, but the changes will not be saved until you press the 'Confirm Changes' button below."
				)
			print(43)
			return scaled_df
		elif method == 'Min-Max Scaling':
			scaler = MinMaxScaler()
			scaled_data = scaler.fit_transform(df)
			scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
			st.warning(
				"Min-Max scaling has been applied, but the changes will not be saved until you press the 'Confirm Changes' button below."
				)
			print(44)
			return scaled_df
		elif method == 'Robust Scaling':
			scaler = RobustScaler()
			scaled_data = scaler.fit_transform(df)
			scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
			st.warning(
				"Robust scaling has been applied, but the changes will not be saved until you press the 'Confirm Changes' button below."
				)
			print(45)
			return scaled_df
		
	except Exception as e:
		st.warning('Only Numerical Columns Should be present.')
		print(46)
		st.error(f'Error occurred: {e}')
		print(47)
		return df
	
# def apply_scaling_for_sup(df, y_col, method):
# 	try:
# 		# Separate the Y column
# 		df_to_scale = df.drop(columns=[y_col])
# 		y_data = df[y_col]

# 		if method == 'Standardize Scaling':
# 			scaler = StandardScaler()
# 			scaled_data = scaler.fit_transform(df_to_scale)
# 			scaled_df = pd.DataFrame(scaled_data, columns=df_to_scale.columns)
# 			print("Standardize scaling has been applied, but the changes will not be saved until you press the 'Confirm Changes' button below.")
# 			print(43)
# 		elif method == 'Min-Max Scaling':
# 			scaler = MinMaxScaler()
# 			scaled_data = scaler.fit_transform(df_to_scale)
# 			scaled_df = pd.DataFrame(scaled_data, columns=df_to_scale.columns)
# 			print("Min-Max scaling has been applied, but the changes will not be saved until you press the 'Confirm Changes' button below.")
# 			print(44)
# 		elif method == 'Robust Scaling':
# 			scaler = RobustScaler()
# 			scaled_data = scaler.fit_transform(df_to_scale)
# 			scaled_df = pd.DataFrame(scaled_data, columns=df_to_scale.columns)
# 			print("Robust scaling has been applied, but the changes will not be saved until you press the 'Confirm Changes' button below.")
# 			print(45)

# 		# Add the Y column back to the scaled DataFrame
# 		scaled_df[y_col] = y_data

# 		scaled_df = scaled_df[df.columns]

# 		return scaled_df

# 	except Exception as e:
# 		print('Only Numerical Columns Should be present.')
# 		print(46)
# 		print(f'Error occurred: {e}')
# 		print(47)
# 		return df

def apply_scaling_for_sup(df, y_col, method):
	try:
		# Separate the Y column
		df_to_scale = df.drop(columns=[y_col])
		# df_to_scale_cols = df_to_scale.columns                     # IMP for phse 4
		y_data = df[y_col]

		# Select the appropriate scaler based on the method
		if method == 'Standardize Scaling':
			scaler = StandardScaler()
			# preprocessor = ColumnTransformer(transformers=[
			# 	('num', StandardScaler(), df_to_scale_cols)
			# ])
			print("Standardize scaling has been applied, but the changes will not be saved until you press the 'Confirm Changes' button below.")
		elif method == 'Min-Max Scaling':
			scaler = MinMaxScaler()
			# preprocessor = ColumnTransformer(transformers=[
			# 	('num', MinMaxScaler(), df_to_scale_cols)
			# ])
			print("Min-Max scaling has been applied, but the changes will not be saved until you press the 'Confirm Changes' button below.")
		elif method == 'Robust Scaling':
			scaler = RobustScaler()
			# preprocessor = ColumnTransformer(transformers=[
			# 	('num', RobustScaler(), df_to_scale_cols)
			# ])
			print("Robust scaling has been applied, but the changes will not be saved until you press the 'Confirm Changes' button below.")
		else:
			raise ValueError(f"Unknown scaling method: {method}")

		# Fit and transform the data using the selected scaler
		scaled_data = scaler.fit_transform(df_to_scale)
		scaled_df = pd.DataFrame(scaled_data, columns=df_to_scale.columns, index=df.index)

		# Add the Y column back to the scaled DataFrame
		scaled_df[y_col] = y_data

		# Reorder the columns to match the original DataFrame
		scaled_df = scaled_df[df.columns]

		return scaled_df

	except Exception as e:
		print('Only Numerical Columns Should be present.')
		print(f'Error occurred: {e}')
		return df


def discretize_output(df, column_name, bins, strategy):
	try:
		# discretizer = KBinsDiscretizer(n_bins=bins, encode='ordinal',
		# 	strategy=strategy)
		# df[column_name] = discretizer.fit_transform(df[[column_name]])
		df_copy = df.copy()
		discretizer = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy=strategy)

		# Creating pipeline for discretizer
		discretizer_pipeline = Pipeline(steps=[
			('discretizer', discretizer)
		])

		# Fitting and transforming the discretizer on the specified column
		df_transformed = discretizer_pipeline.fit_transform(df_copy[[column_name]])
		df_copy[column_name] = df_transformed


		st.warning(
			f"Output variable '{column_name}' discretized, but changes will not be saved until you press the 'Confirm Changes' button below."
			)
		print(48)
		return df_copy, discretizer_pipeline
	except Exception as e:
		st.warning('Works for only numerical columns')
		print(49)
		st.error(f'Error occurred: {e}')
		print(50)
		return df


def calculate_insights(column_data):
	try:
		insights = {}
		insights['Distinct'] = len(column_data.dropna().unique())
		insights['Distinct (%)'] = len(column_data.dropna().unique()) / len(column_data.dropna()) * 100
		insights['Missing'] = column_data.isnull().sum()
		insights['Missing (%)'] = column_data.isnull().sum() / len(column_data) * 100
		insights['Infinite'] = np.isinf(column_data).sum()
		insights['Infinite (%)'] = np.isinf(column_data).sum() / len(column_data) * 100
		insights['Mean'] = column_data.mean()
		insights['Median'] = column_data.median()
		insights['Mode'] = column_data.mode().iloc[0]
		insights['Minimum'] = column_data.min()
		insights['Maximum'] = column_data.max()
		insights['Zeros'] = (column_data == 0).sum()
		insights['Zeros (%)'] = (column_data == 0).sum() / len(column_data) * 100
		insights['Negative'] = (column_data < 0).sum()
		insights['Negative (%)'] = (column_data < 0).sum() / len(column_data) * 100
		insights['Memory size'] = column_data.memory_usage(deep=True)
		insights['5-th percentile'] = np.percentile(column_data.dropna(), 5)
		insights['Q1'] = np.percentile(column_data.dropna(), 25)
		insights['Median'] = np.median(column_data.dropna())
		insights['Q3'] = np.percentile(column_data.dropna(), 75)
		insights['95-th percentile'] = np.percentile(column_data.dropna(), 95)
		insights['Range'] = insights['Maximum'] - insights['Minimum']
		insights['Interquartile range'] = insights['Q3'] - insights['Q1']
		insights['Descriptive statistics'] = column_data.describe()
		insights['Standard deviation'] = column_data.std()
		if insights['Mean'] != 0:
			insights['Coefficient of variation (CV)'] = insights[
				'Standard deviation'] / insights['Mean']
		else:
			insights['Coefficient of variation (CV)'] = float('NaN')
		insights['Kurtosis'] = column_data.kurtosis()
		mad = np.median(np.abs(column_data.dropna() - np.median(column_data.dropna())))
		insights['Median Absolute Deviation (MAD)'] = mad
		insights['Skewness'] = column_data.skew()
		insights['Sum'] = column_data.sum()
		insights['Variance'] = column_data.var()
	except ZeroDivisionError as e:
		print('_e:=_:', e)
		pass
	except Exception as e:
		print('_e:=_', e)
		print(51)
		pass
		print(52)
		insights
	return insights


if 'button_clicked' not in st.session_state:
	st.session_state.button_clicked = False


def callback():
	st.session_state.button_clicked = True


st.subheader(':green[Try Best Pre-Processing Step]')
sup_unsup = st.sidebar.radio("Select Supervised or Unsupervised Learning:", ['Supervised Learning', 'Unsupervised Learning'])
print(53)
if sup_unsup == 'Unsupervised Learning':
	try:
		# st.write('Session State:->', st.session_state['shared'])
		print(54)
		if 'df_pre' in st.session_state:
			# oragnial_df = st.session_state.df
			df_to_pre = st.session_state.df_pre
			# st.warning("kl")
			# to_select = st.selectbox("Select Data Frame (Recommended: DF to Pre-Process)", ["oragnial_df", "df_to_pre_process"], index=1)
			# if to_select == "oragnial_df":
			# 	df = oragnial_df
			# elif to_select == "df_to_pre_process":
			df = df_to_pre
			
			print(1)
			print(55)
			selected_column = st.selectbox('Select a column', df.columns)
			print(2)
			print(56)
			html_content = (
				f"<div class='column-header'>Insights for column:<code>{selected_column}</code></div>"
				)
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
			st.markdown(css_style, unsafe_allow_html=True)
			print(57)
			st.markdown(html_content, unsafe_allow_html=True)
			print(58)
			print(3)
			print(59)
			if df[selected_column].dtype == 'object':
				col1, col2 = st.columns(2)
				print(4)
				print(60)
				with col1:
					unique_values = df[selected_column].nunique()
					print(5)
					print(61)
					st.write(f'  - Data Type: :green[{df[selected_column].dtype}]')
					print(62)
					print(6)
					print(63)
					st.write(
						f'  - Number of Unique Values: :green[{unique_values}]')
					print(64)
					print(7)
					print(65)
					if unique_values <= 20:
						print(8)
						print(66)
						st.write(
							f"  - Unique Values: :green[{', '.join(map(str, df[selected_column].unique()))}]"
							)
						print(67)
					else:
						print(9)
						print(68)
						st.write(f'  - Top 20 Unique Values:')
						print(69)
						print(10)
						print(70)
						st.write(
							f":green[{', '.join(map(str, df[selected_column].value_counts().head(20).index))}]"
							)
						print(71)
				with col2:
					print(10)
					print(72)
					plt.figure(figsize=(10, 6))
					print(73)
					print(11)
					print(74)
					try:
						print(12)
						print(75)
						sns.countplot(x=limit_unique_values(df[selected_column]
							), data=df, color='green')
						print(76)
					except:
						print(13)
						print(77)
						sns.countplot(x=df[selected_column], data=df, color='green'
							)
						print(78)
					plt.xticks(rotation=45)
					print(79)
					st.pyplot()
					print(80)
					plt.close()
					print(81)
				with st.expander('More Info'):
					print(14)
					print(82)
					tab1, tab2 = st.tabs(['Insights', 'Donut chart'])
					print(15)
					print(83)
					with tab1:
						print(16)
						print(84)
						col7, col8, col9 = st.columns(3)
						with col7:
							print(17)
							print(85)
							st.write('## Insights')
							print(86)
							approximate_distinct_count = df[selected_column
								].nunique()
							approximate_unique_percent = (
								approximate_distinct_count / len(df) * 100)
							missing = df[selected_column].isna().sum()
							missing_percent = missing / len(df) * 100
							memory_size = df[selected_column].memory_usage(deep
								=True)
							st.write(
								f'Approximate Distinct Count: :green[{approximate_distinct_count}]'
								)
							print(87)
							st.write(
								f'Approximate Unique (%): :green[{approximate_unique_percent:.2f}%]'
								)
							print(88)
							st.write(f'Missing: :green[{missing}]')
							print(89)
							st.write(f'Missing (%): :green[{missing_percent:.2f}%]'
								)
							print(90)
							st.write(f'Memory Size: :green[{memory_size}]')
							print(91)
							print(18)
							print(92)
						with col8:
							print(19)
							print(93)
							st.write('## Mode')
							print(94)
							mode = df[selected_column].mode().iloc[0]
							st.write(f'Mode: :green[{mode}]')
							print(95)
							print(20)
							print(96)
						with col9:
							print(21)
							print(97)
							st.write('## First 5 Sample Rows')
							print(98)
							st.write(df[selected_column].head())
							print(99)
							print(22)
							print(100)
					with tab2:
						print(23)
						print(101)
						data = limit_unique_values(df[selected_column]
							).value_counts().reset_index()
						data.columns = [selected_column, 'count']
						fig = px.pie(data, values='count', names=
							selected_column, hole=0.5)
						fig.update_traces(textposition='inside', textinfo=
							'percent+label')
						print(102)
						fig.update_layout(legend=dict(orientation='h', yanchor=
							'bottom', y=1.02, xanchor='right', x=1))
						print(103)
						st.plotly_chart(fig)
						print(104)
						print(24)
						print(105)
			elif pd.api.types.is_numeric_dtype(df[selected_column]):
				print(25)
				print(106)
				col3, col4 = st.columns(2)
				print(26)
				print(107)
				with col3:
					print(27)
					print(108)
					st.write(f'  - Data Type: :green[{df[selected_column].dtype}]')
					print(109)
					st.write(f'  - Mean: :green[{df[selected_column].mean()}]')
					print(110)
					st.write(
						f'  - Standard Deviation: :green[{df[selected_column].std()}]'
						)
					print(111)
					st.write(f'  - Min Value: :green[{df[selected_column].min()}]')
					print(112)
					st.write(f'  - Max Value: :green[{df[selected_column].max()}]')
					print(113)
					print(28)
					print(114)
				with col4:
					plt.figure(figsize=(10, 6))
					print(115)
					sns.histplot(df[selected_column], kde=True, color='green')
					print(116)
					st.pyplot()
					print(117)
					plt.close()
					print(118)
					print(29)
					print(119)
				with st.expander('More Info'):
					print(30)
					print(120)
					tab1, tab2, tab3 = st.tabs(['Insights', 'Box plot', 'QQ plot'])
					with tab1:
						print(31)
						print(121)
						col4, col5, col6 = st.columns(3)
						with col4:
							print(32)
							print(122)
							st.write('#### Basic Statistics')
							print(123)
							insights = calculate_insights(df[selected_column])
							basic_stats = {key: value for key, value in
								insights.items() if key in ['Mean', 'Median',
								'Mode', 'Standard deviation', 'Variance',
								'Kurtosis', 'Skewness']}
							for key, value in basic_stats.items():
								st.write(f'**{key}:** :green[{value:.3f}]')
								print(124)
							st.write(
								f"**Memory size:** :green[{insights.get('Memory size', 'N/A'):.3f}]"
								)
							print(125)
							st.write(
								f"**Range:** :green[{insights.get('Range', 'N/A'):.3f}]"
								)
							print(126)
							st.write(
								f"**Interquartile range:** :green[{insights.get('Interquartile range', 'N/A'):.3f}]"
								)
							print(127)
							print(33)
							print(128)
						with col5:
							print(34)
							print(129)
							st.write('#### Percentiles')
							print(130)
							descriptive_stats = insights.get(
								'Descriptive statistics')
							if descriptive_stats is not None:
								percentiles = descriptive_stats.loc[['min',
									'25%', '50%', '75%', 'max']]
								if '5%' in descriptive_stats.index:
									percentiles['5%'] = descriptive_stats['5%']
								if '95%' in descriptive_stats.index:
									percentiles['95%'] = descriptive_stats['95%']
								st.write(percentiles)
								print(131)
							print(35)
							print(132)
						with col6:
							print(36)
							print(133)
							st.write('#### Additional Statistics')
							print(134)
							additional_stats = {key: value for key, value in
								insights.items() if key in ['Distinct',
								'Distinct (%)', 'Missing', 'Missing (%)',
								'Zeros', 'Zeros (%)', 'Negative', 'Negative (%)']}
							for key, value in additional_stats.items():
								st.write(f'**{key}:** :green[{value:.3f}]')
								print(135)
							st.write(
								f"**Coefficient of variation (CV):** :green[{insights.get('Coefficient of variation (CV)', 'N/A'):.3f}]"
								)
							print(136)
							st.write(
								f"**Median Absolute Deviation (MAD):** :green[{insights.get('Median Absolute Deviation (MAD)', 'N/A'):.3f}]"
								)
							print(137)
							st.write(
								f"**Sum:** :green[{insights.get('Sum', 'N/A'):.3f}]"
								)
							print(138)
							print(37)
							print(139)
					with tab2:
						print(38)
						print(140)
						fig = px.box(df, y=selected_column)
						st.plotly_chart(fig)
						print(141)
						print(39)
						print(142)
					with tab3:
						print(40)
						print(143)
						plt.figure(figsize=(10, 6))
						print(144)
						qqplot_data = sm.qqplot(df[selected_column], line='s').gca(
							).lines
						fig = go.Figure()
						fig.add_trace({'type': 'scatter', 'x': qqplot_data[0].
							get_xdata(), 'y': qqplot_data[0].get_ydata(),
							'mode': 'markers', 'marker': {'color': '#19d3f3'}})
						print(145)
						fig.add_trace({'type': 'scatter', 'x': qqplot_data[1].
							get_xdata(), 'y': qqplot_data[1].get_ydata(),
							'mode': 'lines', 'line': {'color': '#636efa'}})
						print(146)
						x_min = min(qqplot_data[0].get_xdata())
						x_max = max(qqplot_data[0].get_xdata())
						y_min = min(qqplot_data[0].get_ydata())
						y_max = max(qqplot_data[0].get_ydata())
						fig.add_trace(go.Scatter(x=[x_min, x_max], y=[y_min,
							y_max], mode='lines', line=dict(color='red', width=
							2), name='Identity Line'))
						print(147)
						fig.update_layout({'title':
							f'QQ Plot for {selected_column}', 'xaxis': {'title':
							'Theoretical Quantiles', 'zeroline': False},
							'yaxis': {'title': 'Sample Quantiles'},
							'showlegend': False, 'width': 800, 'height': 700})
						print(148)
						st.plotly_chart(fig)
						print(149)
						print(41)
						print(150)
		else:
			st.write('DataFrame not found.')
			print(151)
	except ZeroDivisionError:
		pass
	except Exception as e:
		st.error(e)
		print(152)
		st.subheader('⚠️Please upload a file⚠️')
		print(153)
		pass
	print(42)
	print(154)
	preprocessing_action = st.sidebar.radio('Select preprocessing action', [
		'Select Pre-Processing Step⬇️', 'Drop Column :orange[(For Col)]', 'Treat Missing :orange[(For Col)]',
		'Treat Missing :green[(Full DF)]', 'Change Data Type :orange[(For Col)]', 'Drop Duplicates :green[(Full DF)]',
		'Treat Outliers :orange[(For Col)]', 'Treat Outliers :green[(Full DF)]', 'Apply Transformation :orange[(For Col)]',
		'Column Unique Value Replacement :orange[(For Col)]', 'Discretize Variable :orange[(For Col)]',
		'Dummy Variables :green[(Full DF)]', 'Dummy Variable :orange[(For Col)]', 'Apply Scaling :green[(Full DF)]', 'Clustering (Unsupervised Learning)'], help=":orange[(For Col)] means -> operation on particular Column, :green[(Full DF)] means -> operation on full DataFrame")
	st.sidebar.write(":orange[Confirmed Changes] ⤵️")
	if len(st.session_state.full_flow) == 0:
		non = ["Not Confirmed any Changes Yet"]
		st.sidebar.write(non)
	else:
		st.sidebar.write(st.session_state.full_flow)

	try:
		if preprocessing_action == 'Select Pre-Processing Step⬇️':
			print('You have successfuly reached Pre-Proceaaing Phase')
			st.dataframe(df)
			print(155)
		elif preprocessing_action == 'Change Data Type :orange[(For Col)]':
			dtype_options = ['int', 'int32', 'Int32', 'int64', 'Int64', 'float',
				'float32', 'Float32', 'float64', 'Float64', 'str', 'bool']
			new_dtype = st.selectbox('Select new data type', dtype_options)
			if st.button('Apply', on_click=callback
				) or st.session_state.button_clicked:
				modified_df = df.copy()
				modified_df = change_dtype(modified_df, selected_column, new_dtype)
				print(156)
				st.write('Modified DataFrame:')
				print(157)
				st.write(modified_df)
				print(158)
				try:
					if modified_df[selected_column].dtype == 'object':
						col1, col2 = st.columns(2)
						with col1:
							unique_values = modified_df[selected_column].nunique()
							st.write(
								f'  - Data Type: :green[{modified_df[selected_column].dtype}]'
								)
							print(159)
							st.write(
								f'  - Number of Unique Values: :green[{unique_values}]'
								)
							print(160)
							if unique_values <= 20:
								st.write(
									f"  - Unique Values: :green[{', '.join(map(str, modified_df[selected_column].unique()))}]"
									)
								print(161)
							else:
								st.write(f'  - Top 20 Unique Values:')
								print(162)
								st.write(
									f":green[{', '.join(map(str, modified_df[selected_column].value_counts().head(20).index))}]"
									)
								print(163)
						with col2:
							plt.figure(figsize=(10, 6))
							print(164)
							try:
								sns.countplot(x=limit_unique_values(modified_df
									[selected_column]), data=modified_df, color
									='green')
								print(165)
							except:
								sns.countplot(x=modified_df[selected_column],
									data=modified_df, color='green')
								print(166)
							plt.xticks(rotation=45)
							print(167)
							st.pyplot()
							print(168)
							plt.close()
							print(169)
						with st.expander('More Info'):
							tab1, tab2 = st.tabs(['Insights', 'Donut chart'])
							with tab1:
								col7, col8, col9 = st.columns(3)
								with col7:
									st.write('## Insights')
									print(170)
									approximate_distinct_count = modified_df[
										selected_column].nunique()
									approximate_unique_percent = (
										approximate_distinct_count / len(
										modified_df) * 100)
									missing = modified_df[selected_column].isna(
										).sum()
									missing_percent = missing / len(modified_df
										) * 100
									memory_size = modified_df[selected_column
										].memory_usage(deep=True)
									st.write(
										f'Approximate Distinct Count: :green[{approximate_distinct_count}]'
										)
									print(171)
									st.write(
										f'Approximate Unique (%): :green[{approximate_unique_percent:.2f}%]'
										)
									print(172)
									st.write(f'Missing: :green[{missing}]')
									print(173)
									st.write(
										f'Missing (%): :green[{missing_percent:.2f}%]'
										)
									print(174)
									st.write(f'Memory Size: :green[{memory_size}]')
									print(175)
								with col8:
									st.write('## Mode')
									print(176)
									mode = modified_df[selected_column].mode(
										).iloc[0]
									st.write(f'Mode: :green[{mode}]')
									print(177)
								with col9:
									st.write('## First 5 Sample Rows')
									print(178)
									st.write(modified_df[selected_column].head())
									print(179)
							with tab2:
								data = limit_unique_values(modified_df[
									selected_column]).value_counts().reset_index()
								data.columns = [selected_column, 'count']
								fig = px.pie(data, values='count', names=
									selected_column, hole=0.5)
								fig.update_traces(textposition='inside',
									textinfo='percent+label')
								print(180)
								fig.update_layout(legend=dict(orientation='h',
									yanchor='bottom', y=1.02, xanchor='right', x=1)
									)
								print(181)
								st.plotly_chart(fig)
								print(182)
					elif pd.api.types.is_numeric_dtype(modified_df[selected_column]
						):
						col3, col4 = st.columns(2)
						with col3:
							st.write(
								f'  - Data Type: :green[{modified_df[selected_column].dtype}]'
								)
							print(183)
							st.write(
								f'  - Mean: :green[{modified_df[selected_column].mean()}]'
								)
							print(184)
							st.write(
								f'  - Standard Deviation: :green[{modified_df[selected_column].std():.3}]'
								)
							print(185)
							st.write(
								f'  - Min Value: :green[{modified_df[selected_column].min()}]'
								)
							print(186)
							st.write(
								f'  - Max Value: :green[{modified_df[selected_column].max()}]'
								)
							print(187)
						with col4:
							plt.figure(figsize=(10, 6))
							print(188)
							sns.histplot(modified_df[selected_column], kde=True,
								color='green')
							print(189)
							st.pyplot()
							print(190)
							plt.close()
							print(191)
						with st.expander('More Info'):
							tab1, tab2, tab3 = st.tabs(['Insights', 'Box plot',
								'QQ plot'])
							with tab1:
								col4, col5, col6 = st.columns(3)
								with col4:
									st.write('#### Basic Statistics')
									print(192)
									insights = calculate_insights(modified_df[
										selected_column])
									basic_stats = {key: value for key, value in
										insights.items() if key in ['Mean',
										'Median', 'Mode', 'Standard deviation',
										'Variance', 'Kurtosis', 'Skewness']}
									for key, value in basic_stats.items():
										st.write(f'**{key}:** :green[{value:.3f}]')
										print(193)
									st.write(
										f"**Memory size:** :green[{insights.get('Memory size', 'N/A'):.3f}]"
										)
									print(194)
									st.write(
										f"**Range:** :green[{insights.get('Range', 'N/A'):.3f}]"
										)
									print(195)
									st.write(
										f"**Interquartile range:** :green[{insights.get('Interquartile range', 'N/A'):.3f}]"
										)
									print(196)
								with col5:
									st.write('#### Percentiles')
									print(197)
									descriptive_stats = insights.get(
										'Descriptive statistics')
									if descriptive_stats is not None:
										percentiles = descriptive_stats.loc[[
											'min', '25%', '50%', '75%', 'max']]
										if '5%' in descriptive_stats.index:
											percentiles['5%'] = descriptive_stats['5%']
										if '95%' in descriptive_stats.index:
											percentiles['95%'] = descriptive_stats[
												'95%']
										st.write(percentiles)
										print(198)
								with col6:
									st.write('#### Additional Statistics')
									print(199)
									additional_stats = {key: value for key,
										value in insights.items() if key in [
										'Distinct', 'Distinct (%)', 'Missing',
										'Missing (%)', 'Zeros', 'Zeros (%)',
										'Negative', 'Negative (%)']}
									for key, value in additional_stats.items():
										st.write(f'**{key}:** :green[{value:.3f}]')
										print(200)
									st.write(
										f"**Coefficient of variation (CV):** :green[{insights.get('Coefficient of variation (CV)', 'N/A'):.3f}]"
										)
									print(201)
									st.write(
										f"**Median Absolute Deviation (MAD):** :green[{insights.get('Median Absolute Deviation (MAD)', 'N/A'):.3f}]"
										)
									print(202)
									st.write(
										f"**Sum:** :green[{insights.get('Sum', 'N/A'):.3f}]"
										)
									print(203)
							with tab2:
								fig = px.box(modified_df, y=selected_column)
								st.plotly_chart(fig)
								print(204)
							with tab3:
								plt.figure(figsize=(10, 6))
								print(205)
								qqplot_data = sm.qqplot(modified_df[
									selected_column], line='s').gca().lines
								fig = go.Figure()
								fig.add_trace({'type': 'scatter', 'x':
									qqplot_data[0].get_xdata(), 'y':
									qqplot_data[0].get_ydata(), 'mode':
									'markers', 'marker': {'color': '#19d3f3'}})
								print(206)
								fig.add_trace({'type': 'scatter', 'x':
									qqplot_data[1].get_xdata(), 'y':
									qqplot_data[1].get_ydata(), 'mode': 'lines',
									'line': {'color': '#636efa'}})
								print(207)
								x_min = min(qqplot_data[0].get_xdata())
								x_max = max(qqplot_data[0].get_xdata())
								y_min = min(qqplot_data[0].get_ydata())
								y_max = max(qqplot_data[0].get_ydata())
								fig.add_trace(go.Scatter(x=[x_min, x_max], y=[
									y_min, y_max], mode='lines', line=dict(
									color='red', width=2), name='Identity Line'))
								print(208)
								fig.update_layout({'title':
									f'QQ Plot for {selected_column}', 'xaxis':
									{'title': 'Theoretical Quantiles',
									'zeroline': False}, 'yaxis': {'title':
									'Sample Quantiles'}, 'showlegend': False,
									'width': 800, 'height': 700})
								print(209)
								st.plotly_chart(fig)
								print(210)
						
						print(211)
					else:
						st.write('DataFrame not found.')
						print(212)
					print(213)
				except ZeroDivisionError:
					pass
				except Exception as e:
					st.error(e)
					print(214)
					st.subheader('⚠️Please upload a file⚠️')
					print(215)
					pass
				confirm_change = st.button('Confirm Change')
				if confirm_change:
					if 'type_ch' in st.session_state:
						type_ch_no = st.session_state.type_ch
					st.session_state.df_pre = modified_df
					pipeline = {'column': f'{selected_column}', 'new_dtype':
						f'{new_dtype}'}
					
					with open(f'{type_ch_no}_datatype_pipeline.pkl', 'wb') as f:
						joblib.dump(pipeline, f)
						print(216)
					if type_ch_no == 0:
						st.session_state.pre_act.append(preprocessing_action)
					type_ch_no += 1
					st.session_state.full_flow.append(f"{preprocessing_action}->`{selected_column}`")
					st.session_state.type_ch = type_ch_no
					st.rerun()
					print(217)

		elif preprocessing_action == 'Drop Column :orange[(For Col)]':
			if st.button('Apply', on_click=callback
				) or st.session_state.button_clicked:
				modified_df = df.copy()
				drop_column(modified_df, selected_column)
				print(218)
				st.write('Modified DataFrame:')
				print(219)
				st.write(modified_df)
				print(220)
				try:
					if modified_df[selected_column].dtype == 'object':
						col1, col2 = st.columns(2)
						with col1:
							unique_values = modified_df[selected_column].nunique()
							st.write(
								f'  - Data Type: :green[{modified_df[selected_column].dtype}]'
								)
							print(221)
							st.write(
								f'  - Number of Unique Values: :green[{unique_values}]'
								)
							print(222)
							if unique_values <= 20:
								st.write(
									f"  - Unique Values: :green[{', '.join(map(str, modified_df[selected_column].unique()))}]"
									)
								print(223)
							else:
								st.write(f'  - Top 20 Unique Values:')
								print(224)
								st.write(
									f":green[{', '.join(map(str, modified_df[selected_column].value_counts().head(20).index))}]"
									)
								print(225)
						with col2:
							plt.figure(figsize=(10, 6))
							print(226)
							try:
								sns.countplot(x=limit_unique_values(modified_df
									[selected_column]), data=modified_df, color
									='green')
								print(227)
							except:
								sns.countplot(x=modified_df[selected_column],
									data=modified_df, color='green')
								print(228)
							plt.xticks(rotation=45)
							print(229)
							st.pyplot()
							print(230)
							plt.close()
							print(231)
						with st.expander('More Info'):
							tab1, tab2 = st.tabs(['Insights', 'Donut chart'])
							with tab1:
								col7, col8, col9 = st.columns(3)
								with col7:
									st.write('## Insights')
									print(232)
									approximate_distinct_count = modified_df[
										selected_column].nunique()
									approximate_unique_percent = (
										approximate_distinct_count / len(
										modified_df) * 100)
									missing = modified_df[selected_column].isna(
										).sum()
									missing_percent = missing / len(modified_df
										) * 100
									memory_size = modified_df[selected_column
										].memory_usage(deep=True)
									st.write(
										f'Approximate Distinct Count: :green[{approximate_distinct_count}]'
										)
									print(233)
									st.write(
										f'Approximate Unique (%): :green[{approximate_unique_percent:.2f}%]'
										)
									print(234)
									st.write(f'Missing: :green[{missing}]')
									print(235)
									st.write(
										f'Missing (%): :green[{missing_percent:.2f}%]'
										)
									print(236)
									st.write(f'Memory Size: :green[{memory_size}]')
									print(237)
								with col8:
									st.write('## Mode')
									print(238)
									mode = modified_df[selected_column].mode(
										).iloc[0]
									st.write(f'Mode: :green[{mode}]')
									print(239)
								with col9:
									st.write('## First 5 Sample Rows')
									print(240)
									st.write(modified_df[selected_column].head())
									print(241)
							with tab2:
								data = limit_unique_values(modified_df[
									selected_column]).value_counts().reset_index()
								data.columns = [selected_column, 'count']
								fig = px.pie(data, values='count', names=
									selected_column, hole=0.5)
								fig.update_traces(textposition='inside',
									textinfo='percent+label')
								print(242)
								fig.update_layout(legend=dict(orientation='h',
									yanchor='bottom', y=1.02, xanchor='right', x=1)
									)
								print(243)
								st.plotly_chart(fig)
								print(244)
					elif pd.api.types.is_numeric_dtype(modified_df[selected_column]
						):
						col3, col4 = st.columns(2)
						with col3:
							st.write(
								f'  - Data Type: :green[{modified_df[selected_column].dtype}]'
								)
							print(245)
							st.write(
								f'  - Mean: :green[{modified_df[selected_column].mean()}]'
								)
							print(246)
							st.write(
								f'  - Standard Deviation: :green[{modified_df[selected_column].std():.3}]'
								)
							print(247)
							st.write(
								f'  - Min Value: :green[{modified_df[selected_column].min()}]'
								)
							print(248)
							st.write(
								f'  - Max Value: :green[{modified_df[selected_column].max()}]'
								)
							print(249)
						with col4:
							plt.figure(figsize=(10, 6))
							print(250)
							sns.histplot(modified_df[selected_column], kde=True,
								color='green')
							print(251)
							st.pyplot()
							print(252)
							plt.close()
							print(253)
						with st.expander('More Info'):
							tab1, tab2, tab3 = st.tabs(['Insights', 'Box plot',
								'QQ plot'])
							with tab1:
								col4, col5, col6 = st.columns(3)
								with col4:
									st.write('#### Basic Statistics')
									print(254)
									insights = calculate_insights(modified_df[
										selected_column])
									basic_stats = {key: value for key, value in
										insights.items() if key in ['Mean',
										'Median', 'Mode', 'Standard deviation',
										'Variance', 'Kurtosis', 'Skewness']}
									for key, value in basic_stats.items():
										st.write(f'**{key}:** :green[{value:.3f}]')
										print(255)
									st.write(
										f"**Memory size:** :green[{insights.get('Memory size', 'N/A'):.3f}]"
										)
									print(256)
									st.write(
										f"**Range:** :green[{insights.get('Range', 'N/A'):.3f}]"
										)
									print(257)
									st.write(
										f"**Interquartile range:** :green[{insights.get('Interquartile range', 'N/A'):.3f}]"
										)
									print(258)
								with col5:
									st.write('#### Percentiles')
									print(259)
									descriptive_stats = insights.get(
										'Descriptive statistics')
									if descriptive_stats is not None:
										percentiles = descriptive_stats.loc[[
											'min', '25%', '50%', '75%', 'max']]
										if '5%' in descriptive_stats.index:
											percentiles['5%'] = descriptive_stats['5%']
										if '95%' in descriptive_stats.index:
											percentiles['95%'] = descriptive_stats[
												'95%']
										st.write(percentiles)
										print(260)
								with col6:
									st.write('#### Additional Statistics')
									print(261)
									additional_stats = {key: value for key,
										value in insights.items() if key in [
										'Distinct', 'Distinct (%)', 'Missing',
										'Missing (%)', 'Zeros', 'Zeros (%)',
										'Negative', 'Negative (%)']}
									for key, value in additional_stats.items():
										st.write(f'**{key}:** :green[{value:.3f}]')
										print(262)
									st.write(
										f"**Coefficient of variation (CV):** :green[{insights.get('Coefficient of variation (CV)', 'N/A'):.3f}]"
										)
									print(263)
									st.write(
										f"**Median Absolute Deviation (MAD):** :green[{insights.get('Median Absolute Deviation (MAD)', 'N/A'):.3f}]"
										)
									print(264)
									st.write(
										f"**Sum:** :green[{insights.get('Sum', 'N/A'):.3f}]"
										)
									print(265)
							with tab2:
								fig = px.box(modified_df, y=selected_column)
								st.plotly_chart(fig)
								print(266)
							with tab3:
								plt.figure(figsize=(10, 6))
								print(267)
								qqplot_data = sm.qqplot(modified_df[
									selected_column], line='s').gca().lines
								fig = go.Figure()
								fig.add_trace({'type': 'scatter', 'x':
									qqplot_data[0].get_xdata(), 'y':
									qqplot_data[0].get_ydata(), 'mode':
									'markers', 'marker': {'color': '#19d3f3'}})
								print(268)
								fig.add_trace({'type': 'scatter', 'x':
									qqplot_data[1].get_xdata(), 'y':
									qqplot_data[1].get_ydata(), 'mode': 'lines',
									'line': {'color': '#636efa'}})
								print(269)
								x_min = min(qqplot_data[0].get_xdata())
								x_max = max(qqplot_data[0].get_xdata())
								y_min = min(qqplot_data[0].get_ydata())
								y_max = max(qqplot_data[0].get_ydata())
								fig.add_trace(go.Scatter(x=[x_min, x_max], y=[
									y_min, y_max], mode='lines', line=dict(
									color='red', width=2), name='Identity Line'))
								print(270)
								fig.update_layout({'title':
									f'QQ Plot for {selected_column}', 'xaxis':
									{'title': 'Theoretical Quantiles',
									'zeroline': False}, 'yaxis': {'title':
									'Sample Quantiles'}, 'showlegend': False,
									'width': 800, 'height': 700})
								print(271)
								st.plotly_chart(fig)
								print(272)
						
						print(273)
					else:
						st.write('DataFrame not found.')
						print(274)
					print(275)
				except ZeroDivisionError:
					pass
				except Exception as e:
					pass
				confirm_change = st.button('Confirm Change')
				if confirm_change:
					if 'drop' in st.session_state:
						drop_no = st.session_state.drop
					pipeline1 = {'column': f'{selected_column}'}
					
					with open(f'{drop_no}_drop_pipeline.pkl', 'wb') as f:
						joblib.dump(pipeline1, f)
						print(276)
					st.session_state.df_pre = modified_df
					if drop_no == 0:
						st.session_state.pre_act.append(preprocessing_action)
					drop_no += 1
					st.session_state.full_flow.append(f"{preprocessing_action}->`{selected_column}`")
					st.session_state.drop = drop_no
					# modified_df.to_csv("clean.csv", index=False)
					st.rerun()
					print(277)
		elif preprocessing_action == 'Drop Duplicates :green[(Full DF)]':
			if st.button('Apply', on_click=callback
				) or st.session_state.button_clicked:
				modified_df = df.copy()
				drop_duplicates(modified_df)
				print(278)
				st.write('Modified DataFrame:')
				print(279)
				st.write(modified_df)
				print(280)
				confirm_change = st.button('Confirm Change')
				if confirm_change:
					pipeline2 = {'action': 'drop_duplicates'}
					with open('drop_dup_pipeline.pkl', 'wb') as f:
						joblib.dump(pipeline2, f)
						print(281)
					st.session_state.pre_act.append(preprocessing_action)
					st.session_state.df_pre = modified_df
					st.session_state.full_flow.append(f"{preprocessing_action}")
					st.rerun()
					print(282)
		elif preprocessing_action == 'Treat Outliers :orange[(For Col)]':
			# outlier_method = st.radio('Select Outlier Treatment Method', [
			# 	'Delete Outliers', 'Winsorization'])
			# Custom CSS for green text
			st.markdown(
				"""
				<style>
				.green-text {
					color: green;
				}
				</style>
				""",
				unsafe_allow_html=True
			)

			# Using st.markdown with custom CSS class
			st.markdown('- <h2 class="green-text">Winsorization:</h2>', unsafe_allow_html=True)
			if st.button('Apply', on_click=callback
				) or st.session_state.button_clicked:
				modified_df = df.copy()
				# if outlier_method == 'Winsorization':
				# 	lower_limit = st.slider('Select Lower Limit', min_value=0.0,
				# 		max_value=0.1, step=0.01, value=0.05)
				# 	upper_limit = st.slider('Select Upper Limit', min_value=0.0,
				# 		max_value=0.1, step=0.01, value=0.05)
				# else:
				# 	lower_limit = 0.05
				# 	upper_limit = 0.05
				# if outlier_method == 'Delete Outliers':
				# 	z_score_threshold = st.slider('Select Z-Score Threshold',
				# 		min_value=1.0, max_value=5.0, step=0.1, value=3.0)
				# else:
				# 	z_score_threshold = 3.0
				# modified_df = apply_winsorization_col(modified_df, selected_column, outlier_method, z_score_threshold, lower_limit, upper_limit)
				default_capping_method = 'iqr'
				default_tail = 'both'
				default_fold = 1.5

				# User inputs for configuring Winsorizer
				capping_method = st.selectbox('Capping Method', ['iqr', 'gaussian', 'percentile', 'mad'], index=0)
				tail = st.selectbox('Tail', ['both', 'left', 'right'], index=0)
				fold = st.number_input('Fold', min_value=0.1, max_value=10.0, value=default_fold, step=0.1)

				preprocessor, modified_df = apply_winsorization_col(modified_df, selected_column, capping_method, tail, fold)
				st.write('Modified DataFrame:')
				print(283)
				st.write(modified_df)
				print(284)
				try:
					if modified_df[selected_column].dtype == 'object':
						col1, col2 = st.columns(2)
						with col1:
							unique_values = modified_df[selected_column].nunique()
							st.write(
								f'  - Data Type: :green[{modified_df[selected_column].dtype}]'
								)
							print(285)
							st.write(
								f'  - Number of Unique Values: :green[{unique_values}]'
								)
							print(286)
							if unique_values <= 20:
								st.write(
									f"  - Unique Values: :green[{', '.join(map(str, modified_df[selected_column].unique()))}]"
									)
								print(287)
							else:
								st.write(f'  - Top 20 Unique Values:')
								print(288)
								st.write(
									f":green[{', '.join(map(str, modified_df[selected_column].value_counts().head(20).index))}]"
									)
								print(289)
						with col2:
							plt.figure(figsize=(10, 6))
							print(290)
							try:
								sns.countplot(x=limit_unique_values(modified_df
									[selected_column]), data=modified_df, color
									='green')
								print(291)
							except:
								sns.countplot(x=modified_df[selected_column],
									data=modified_df, color='green')
								print(292)
							plt.xticks(rotation=45)
							print(293)
							st.pyplot()
							print(294)
							plt.close()
							print(295)
						with st.expander('More Info'):
							tab1, tab2 = st.tabs(['Insights', 'Donut chart'])
							with tab1:
								col7, col8, col9 = st.columns(3)
								with col7:
									st.write('## Insights')
									print(296)
									approximate_distinct_count = modified_df[
										selected_column].nunique()
									approximate_unique_percent = (
										approximate_distinct_count / len(
										modified_df) * 100)
									missing = modified_df[selected_column].isna(
										).sum()
									missing_percent = missing / len(modified_df
										) * 100
									memory_size = modified_df[selected_column
										].memory_usage(deep=True)
									st.write(
										f'Approximate Distinct Count: :green[{approximate_distinct_count}]'
										)
									print(297)
									st.write(
										f'Approximate Unique (%): :green[{approximate_unique_percent:.2f}%]'
										)
									print(298)
									st.write(f'Missing: :green[{missing}]')
									print(299)
									st.write(
										f'Missing (%): :green[{missing_percent:.2f}%]'
										)
									print(300)
									st.write(f'Memory Size: :green[{memory_size}]')
									print(301)
								with col8:
									st.write('## Mode')
									print(302)
									mode = modified_df[selected_column].mode(
										).iloc[0]
									st.write(f'Mode: :green[{mode}]')
									print(303)
								with col9:
									st.write('## First 5 Sample Rows')
									print(304)
									st.write(modified_df[selected_column].head())
									print(305)
							with tab2:
								data = limit_unique_values(modified_df[
									selected_column]).value_counts().reset_index()
								data.columns = [selected_column, 'count']
								fig = px.pie(data, values='count', names=
									selected_column, hole=0.5)
								fig.update_traces(textposition='inside',
									textinfo='percent+label')
								print(306)
								fig.update_layout(legend=dict(orientation='h',
									yanchor='bottom', y=1.02, xanchor='right', x=1)
									)
								print(307)
								st.plotly_chart(fig)
								print(308)
					elif pd.api.types.is_numeric_dtype(modified_df[selected_column]
						):
						col3, col4 = st.columns(2)
						with col3:
							st.write(
								f'  - Data Type: :green[{modified_df[selected_column].dtype}]'
								)
							print(309)
							st.write(
								f'  - Mean: :green[{modified_df[selected_column].mean()}]'
								)
							print(310)
							st.write(
								f'  - Standard Deviation: :green[{modified_df[selected_column].std():.3}]'
								)
							print(311)
							st.write(
								f'  - Min Value: :green[{modified_df[selected_column].min()}]'
								)
							print(312)
							st.write(
								f'  - Max Value: :green[{modified_df[selected_column].max()}]'
								)
							print(313)
						with col4:
							plt.figure(figsize=(10, 6))
							print(314)
							sns.histplot(modified_df[selected_column], kde=True,
								color='green')
							print(315)
							st.pyplot()
							print(316)
							plt.close()
							print(317)
						with st.expander('More Info'):
							tab1, tab2, tab3 = st.tabs(['Insights', 'Box plot',
								'QQ plot'])
							with tab1:
								col4, col5, col6 = st.columns(3)
								with col4:
									st.write('#### Basic Statistics')
									print(318)
									insights = calculate_insights(modified_df[
										selected_column])
									basic_stats = {key: value for key, value in
										insights.items() if key in ['Mean',
										'Median', 'Mode', 'Standard deviation',
										'Variance', 'Kurtosis', 'Skewness']}
									for key, value in basic_stats.items():
										st.write(f'**{key}:** :green[{value:.3f}]')
										print(319)
									st.write(
										f"**Memory size:** :green[{insights.get('Memory size', 'N/A'):.3f}]"
										)
									print(320)
									st.write(
										f"**Range:** :green[{insights.get('Range', 'N/A'):.3f}]"
										)
									print(321)
									st.write(
										f"**Interquartile range:** :green[{insights.get('Interquartile range', 'N/A'):.3f}]"
										)
									print(322)
								with col5:
									st.write('#### Percentiles')
									print(323)
									descriptive_stats = insights.get(
										'Descriptive statistics')
									if descriptive_stats is not None:
										percentiles = descriptive_stats.loc[[
											'min', '25%', '50%', '75%', 'max']]
										if '5%' in descriptive_stats.index:
											percentiles['5%'] = descriptive_stats['5%']
										if '95%' in descriptive_stats.index:
											percentiles['95%'] = descriptive_stats[
												'95%']
										st.write(percentiles)
										print(324)
								with col6:
									st.write('#### Additional Statistics')
									print(325)
									additional_stats = {key: value for key,
										value in insights.items() if key in [
										'Distinct', 'Distinct (%)', 'Missing',
										'Missing (%)', 'Zeros', 'Zeros (%)',
										'Negative', 'Negative (%)']}
									for key, value in additional_stats.items():
										st.write(f'**{key}:** :green[{value:.3f}]')
										print(326)
									st.write(
										f"**Coefficient of variation (CV):** :green[{insights.get('Coefficient of variation (CV)', 'N/A'):.3f}]"
										)
									print(327)
									st.write(
										f"**Median Absolute Deviation (MAD):** :green[{insights.get('Median Absolute Deviation (MAD)', 'N/A'):.3f}]"
										)
									print(328)
									st.write(
										f"**Sum:** :green[{insights.get('Sum', 'N/A'):.3f}]"
										)
									print(329)
							with tab2:
								fig = px.box(modified_df, y=selected_column)
								st.plotly_chart(fig)
								print(330)
							with tab3:
								plt.figure(figsize=(10, 6))
								print(331)
								qqplot_data = sm.qqplot(modified_df[
									selected_column], line='s').gca().lines
								fig = go.Figure()
								fig.add_trace({'type': 'scatter', 'x':
									qqplot_data[0].get_xdata(), 'y':
									qqplot_data[0].get_ydata(), 'mode':
									'markers', 'marker': {'color': '#19d3f3'}})
								print(332)
								fig.add_trace({'type': 'scatter', 'x':
									qqplot_data[1].get_xdata(), 'y':
									qqplot_data[1].get_ydata(), 'mode': 'lines',
									'line': {'color': '#636efa'}})
								print(333)
								x_min = min(qqplot_data[0].get_xdata())
								x_max = max(qqplot_data[0].get_xdata())
								y_min = min(qqplot_data[0].get_ydata())
								y_max = max(qqplot_data[0].get_ydata())
								fig.add_trace(go.Scatter(x=[x_min, x_max], y=[
									y_min, y_max], mode='lines', line=dict(
									color='red', width=2), name='Identity Line'))
								print(334)
								fig.update_layout({'title':
									f'QQ Plot for {selected_column}', 'xaxis':
									{'title': 'Theoretical Quantiles',
									'zeroline': False}, 'yaxis': {'title':
									'Sample Quantiles'}, 'showlegend': False,
									'width': 800, 'height': 700})
								print(335)
								st.plotly_chart(fig)
								print(336)
						
						print(337)
					else:
						st.write('DataFrame not found.')
						print(338)
					print(339)
				except ZeroDivisionError:
					pass
				except Exception as e:
					st.error(e)
					print(340)
					st.subheader('⚠️Please upload a file⚠️')
					print(341)
					pass
				confirm_change = st.button('Confirm Change')
				if confirm_change:
					if 'treat_out' in st.session_state:
						treat_out_no = st.session_state.treat_out
					
					joblib.dump(preprocessor, f'{treat_out_no}_outliers_for_col_pipeline.pkl')
					st.session_state.full_flow.append(f"{preprocessing_action}->`{selected_column}`")
					print(342)
					st.session_state.df_pre = modified_df
					if treat_out_no == 0:
						st.session_state.pre_act.append(preprocessing_action)
					treat_out_no += 1
					st.session_state.treat_out = treat_out_no
					st.rerun()
					print(343)
					
					
		elif preprocessing_action == 'Treat Outliers :green[(Full DF)]':
			# method = st.selectbox('Select Method to Treat Outliers', [
			# 	'Delete Outliers', 'Winsorization', 'IQR'])
			st.markdown(
				"""
				<style>
				.green-text {
					color: green;
				}
				</style>
				""",
				unsafe_allow_html=True
			)

			# Using st.markdown with custom CSS class
			st.markdown('- <h2 class="green-text">Winsorization on Full DataFrame:</h2>', unsafe_allow_html=True)
			if st.button('Apply', on_click=callback
				) or st.session_state.button_clicked:
				modified_df = df.copy()
				# if method == 'Delete Outliers':
				# 	z_score_threshold = st.slider('Select Z-Score Threshold',
				# 		min_value=1.0, max_value=5.0, step=0.1, value=3.0)
				# else:
				# 	z_score_threshold = 3.0
				# if method == 'Winsorization':
				# 	lower_limit = st.slider('Select Lower Limit:', min_value=
				# 		0.0, max_value=0.1, step=0.01, value=0.05)
				# 	upper_limit = st.slider('Select Upper Limit:', min_value=
				# 		0.0, max_value=0.1, step=0.01, value=0.05)
				# else:
				# 	lower_limit = upper_limit = 0.05
				# modified_df = treat_outliers_full_df(modified_df, method,
				# 	z_score_threshold=3, lower_limit=lower_limit, upper_limit=
				# 	upper_limit)

				default_capping_method = 'iqr'
				default_tail = 'both'
				default_fold = 1.5

				# User inputs for configuring Winsorizer
				capping_method = st.selectbox('Capping Method', ['iqr', 'gaussian', 'percentile', 'mad'], index=0)
				tail = st.selectbox('Tail', ['both', 'left', 'right'], index=0)
				fold = st.number_input('Fold', min_value=0.1, max_value=10.0, value=default_fold, step=0.1)

				pipeline, modified_df = apply_winsorization_for_full_df_unsup(modified_df, capping_method, tail, fold)
				st.write('Modified DataFrame:')
				print(344)
				st.write(modified_df)
				print(345)
				confirm_change = st.button('Confirm Change')
				if confirm_change:
					st.session_state.df_pre = modified_df
					# pipeline4 = {'method': f'{method}', 'lower_limit':
					# 	f'{lower_limit}', 'upper_limit': f'{upper_limit}',
					# 	'z_score_threshold': f'{z_score_threshold}'}
					st.session_state.pre_act.append(preprocessing_action)
					st.session_state.full_flow.append(f"{preprocessing_action}")
					joblib.dump(pipeline, 'outliers_for_full_df_pipeline.pkl')
					# with open('outliers_for_full_df_pipeline.pkl', 'wb') as f:
					# 	joblib.dump(pipeline4, f)
					print(346)
					st.rerun()
					print(347)
		elif preprocessing_action == 'Treat Missing :orange[(For Col)]':
			missing_method = st.radio('Select Missing Value Treatment Method',
				['Mean Imputation',
				'Median Imputation', 'Mode Imputation'])
			if st.button('Apply', on_click=callback
				) or st.session_state.button_clicked:
				modified_df = df.copy()
				modified_df, Imputer = treat_missing_values(modified_df, selected_column,
					missing_method)
				st.write('Modified DataFrame:')
				print(348)
				st.write(modified_df)
				print(349)
				try:
					if modified_df[selected_column].dtype == 'object':
						col1, col2 = st.columns(2)
						with col1:
							unique_values = modified_df[selected_column].nunique()
							st.write(
								f'  - Data Type: :green[{modified_df[selected_column].dtype}]'
								)
							print(350)
							st.write(
								f'  - Number of Unique Values: :green[{unique_values}]'
								)
							print(351)
							if unique_values <= 20:
								st.write(
									f"  - Unique Values: :green[{', '.join(map(str, modified_df[selected_column].unique()))}]"
									)
								print(352)
							else:
								st.write(f'  - Top 20 Unique Values:')
								print(353)
								st.write(
									f":green[{', '.join(map(str, modified_df[selected_column].value_counts().head(20).index))}]"
									)
								print(354)
						with col2:
							plt.figure(figsize=(10, 6))
							print(355)
							try:
								sns.countplot(x=limit_unique_values(modified_df
									[selected_column]), data=modified_df, color
									='green')
								print(356)
							except:
								sns.countplot(x=modified_df[selected_column],
									data=modified_df, color='green')
								print(357)
							plt.xticks(rotation=45)
							print(358)
							st.pyplot()
							print(359)
							plt.close()
							print(360)
						with st.expander('More Info'):
							tab1, tab2 = st.tabs(['Insights', 'Donut chart'])
							with tab1:
								col7, col8, col9 = st.columns(3)
								with col7:
									st.write('## Insights')
									print(361)
									approximate_distinct_count = modified_df[
										selected_column].nunique()
									approximate_unique_percent = (
										approximate_distinct_count / len(
										modified_df) * 100)
									missing = modified_df[selected_column].isna(
										).sum()
									missing_percent = missing / len(modified_df
										) * 100
									memory_size = modified_df[selected_column
										].memory_usage(deep=True)
									st.write(
										f'Approximate Distinct Count: :green[{approximate_distinct_count}]'
										)
									print(362)
									st.write(
										f'Approximate Unique (%): :green[{approximate_unique_percent:.2f}%]'
										)
									print(363)
									st.write(f'Missing: :green[{missing}]')
									print(364)
									st.write(
										f'Missing (%): :green[{missing_percent:.2f}%]'
										)
									print(365)
									st.write(f'Memory Size: :green[{memory_size}]')
									print(366)
								with col8:
									st.write('## Mode')
									print(367)
									mode = modified_df[selected_column].mode(
										).iloc[0]
									st.write(f'Mode: :green[{mode}]')
									print(368)
								with col9:
									st.write('## First 5 Sample Rows')
									print(369)
									st.write(modified_df[selected_column].head())
									print(370)
							with tab2:
								data = limit_unique_values(modified_df[
									selected_column]).value_counts().reset_index()
								data.columns = [selected_column, 'count']
								fig = px.pie(data, values='count', names=
									selected_column, hole=0.5)
								fig.update_traces(textposition='inside',
									textinfo='percent+label')
								print(371)
								fig.update_layout(legend=dict(orientation='h',
									yanchor='bottom', y=1.02, xanchor='right', x=1)
									)
								print(372)
								st.plotly_chart(fig)
								print(373)
					elif pd.api.types.is_numeric_dtype(modified_df[selected_column]
						):
						col3, col4 = st.columns(2)
						with col3:
							st.write(
								f'  - Data Type: :green[{modified_df[selected_column].dtype}]'
								)
							print(374)
							st.write(
								f'  - Mean: :green[{modified_df[selected_column].mean()}]'
								)
							print(375)
							st.write(
								f'  - Standard Deviation: :green[{modified_df[selected_column].std():.3}]'
								)
							print(376)
							st.write(
								f'  - Min Value: :green[{modified_df[selected_column].min()}]'
								)
							print(377)
							st.write(
								f'  - Max Value: :green[{modified_df[selected_column].max()}]'
								)
							print(378)
						with col4:
							plt.figure(figsize=(10, 6))
							print(379)
							sns.histplot(modified_df[selected_column], kde=True,
								color='green')
							print(380)
							st.pyplot()
							print(381)
							plt.close()
							print(382)
						with st.expander('More Info'):
							tab1, tab2, tab3 = st.tabs(['Insights', 'Box plot',
								'QQ plot'])
							with tab1:
								col4, col5, col6 = st.columns(3)
								with col4:
									st.write('#### Basic Statistics')
									print(383)
									insights = calculate_insights(modified_df[
										selected_column])
									basic_stats = {key: value for key, value in
										insights.items() if key in ['Mean',
										'Median', 'Mode', 'Standard deviation',
										'Variance', 'Kurtosis', 'Skewness']}
									for key, value in basic_stats.items():
										st.write(f'**{key}:** :green[{value:.3f}]')
										print(384)
									st.write(
										f"**Memory size:** :green[{insights.get('Memory size', 'N/A'):.3f}]"
										)
									print(385)
									st.write(
										f"**Range:** :green[{insights.get('Range', 'N/A'):.3f}]"
										)
									print(386)
									st.write(
										f"**Interquartile range:** :green[{insights.get('Interquartile range', 'N/A'):.3f}]"
										)
									print(387)
								with col5:
									st.write('#### Percentiles')
									print(388)
									descriptive_stats = insights.get(
										'Descriptive statistics')
									if descriptive_stats is not None:
										percentiles = descriptive_stats.loc[[
											'min', '25%', '50%', '75%', 'max']]
										if '5%' in descriptive_stats.index:
											percentiles['5%'] = descriptive_stats['5%']
										if '95%' in descriptive_stats.index:
											percentiles['95%'] = descriptive_stats[
												'95%']
										st.write(percentiles)
										print(389)
								with col6:
									st.write('#### Additional Statistics')
									print(390)
									additional_stats = {key: value for key,
										value in insights.items() if key in [
										'Distinct', 'Distinct (%)', 'Missing',
										'Missing (%)', 'Zeros', 'Zeros (%)',
										'Negative', 'Negative (%)']}
									for key, value in additional_stats.items():
										st.write(f'**{key}:** :green[{value:.3f}]')
										print(391)
									st.write(
										f"**Coefficient of variation (CV):** :green[{insights.get('Coefficient of variation (CV)', 'N/A'):.3f}]"
										)
									print(392)
									st.write(
										f"**Median Absolute Deviation (MAD):** :green[{insights.get('Median Absolute Deviation (MAD)', 'N/A'):.3f}]"
										)
									print(393)
									st.write(
										f"**Sum:** :green[{insights.get('Sum', 'N/A'):.3f}]"
										)
									print(394)
							with tab2:
								fig = px.box(modified_df, y=selected_column)
								st.plotly_chart(fig)
								print(395)
							with tab3:
								plt.figure(figsize=(10, 6))
								print(396)
								qqplot_data = sm.qqplot(modified_df[
									selected_column], line='s').gca().lines
								fig = go.Figure()
								fig.add_trace({'type': 'scatter', 'x':
									qqplot_data[0].get_xdata(), 'y':
									qqplot_data[0].get_ydata(), 'mode':
									'markers', 'marker': {'color': '#19d3f3'}})
								print(397)
								fig.add_trace({'type': 'scatter', 'x':
									qqplot_data[1].get_xdata(), 'y':
									qqplot_data[1].get_ydata(), 'mode': 'lines',
									'line': {'color': '#636efa'}})
								print(398)
								x_min = min(qqplot_data[0].get_xdata())
								x_max = max(qqplot_data[0].get_xdata())
								y_min = min(qqplot_data[0].get_ydata())
								y_max = max(qqplot_data[0].get_ydata())
								fig.add_trace(go.Scatter(x=[x_min, x_max], y=[
									y_min, y_max], mode='lines', line=dict(
									color='red', width=2), name='Identity Line'))
								print(399)
								fig.update_layout({'title':
									f'QQ Plot for {selected_column}', 'xaxis':
									{'title': 'Theoretical Quantiles',
									'zeroline': False}, 'yaxis': {'title':
									'Sample Quantiles'}, 'showlegend': False,
									'width': 800, 'height': 700})
								print(400)
								st.plotly_chart(fig)
								print(401)
						
						print(402)
					else:
						st.write('DataFrame not found.')
						print(403)
					print(404)
				except ZeroDivisionError:
					pass
				except Exception as e:
					st.error(e)
					print(405)
					st.subheader('⚠️Please upload a file⚠️')
					print(406)
					pass
				confirm_change = st.button('Confirm Change')
				if confirm_change:
					if 'miss' in st.session_state:
						miss_no = st.session_state.miss
					# if missing_method == 'Delete Missing Values':
						
						pipeline5 = {'method': Imputer, 'column':f'{selected_column}'}
					# 	with open(f'{miss_no}_treat_missing_vaues_for_col.pkl', 'wb') as f:
					# 		joblib.dump(pipeline5, f)
					# 		print(407)
					# 	st.session_state.df_pre = modified_df
					# 	miss_no += 1
					# 	st.session_state.miss = miss_no
					# 	st.rerun()
					# 	print(408)
					# else:
						joblib.dump(pipeline5, f'{miss_no}_treat_missing_vaues_for_col.pkl')
						st.session_state.df_pre = modified_df
						if miss_no == 0:
							st.session_state.pre_act.append(preprocessing_action)
						miss_no += 1
						st.session_state.full_flow.append(f"{preprocessing_action}->`{selected_column}`")
						st.session_state.miss = miss_no
						st.rerun()


		elif preprocessing_action == 'Treat Missing :green[(Full DF)]':
			print(409)
			numeric_columns = df.select_dtypes(include=np.number).columns
			categorical_columns = df.select_dtypes(include=['object', 'category']
				).columns
			numeric_treatment = st.selectbox(
				'Select treatment for missing values in numeric columns', [
				'Mean', 'Median', 'Mode', 'Random'])
			categorical_treatment = st.selectbox(
				'Select treatment for missing values in categorical columns', [
				'Mode', 'Random'])
			if st.button('Apply', on_click=callback
				) or st.session_state.button_clicked:
				modified_df = df.copy()
				transformed_df, pipelines = create_full_df_impute_pipelines(modified_df, numeric_treatment, categorical_treatment)
				st.warning("Missing Value Treatment has been applied on full DF, but the changes will not be saved until you press the 'Confirm Changes' button below.")
				print(415)
				st.write('Modified DataFrame:')
				print(416)
				st.write(modified_df)
				numeric_pipeline, categorical_pipeline = pipelines
				print(417)
				confirm_change = st.button('Confirm Change')
				if confirm_change:
					# pipeline6 = {'numeric_treatment': f'{numeric_treatment}',
					# 	'numeric_strategy': f'{numeric_strategy}',
					# 	'categorical_treatment': f'{categorical_treatment}',
					# 	'categorical_strategy': f'{categorical_strategy}'}
					# Saving the pipelines
					st.session_state.pre_act.append(preprocessing_action)
					joblib.dump(numeric_pipeline, 'numeric_treat_missing_vaues_in_full_df.pkl')
					joblib.dump(categorical_pipeline, 'categorical_treat_missing_vaues_in_full_df.pkl')
					st.session_state.full_flow.append(f"{preprocessing_action}")
					# with open('treat_missing_vaues_in_full_df.pkl', 'wb') as f:
					# 	joblib.dump(pipeline6, f)
					print(418)
					st.session_state.df_pre = modified_df
					st.rerun()
					print(419)
				

		elif preprocessing_action == 'Apply Transformation :orange[(For Col)]':
			# transformation_method = st.selectbox('Select Transformation Technique',
			# 	['Log Transformation', 'Exponential Transformation',
			# 	'Square Root Transformation', 'Box-Cox Transformation',
			# 	'Yeo-Johnson Transformation'])
			transformation_method = st.selectbox('Select Transformation Technique',
				['Log Transformation', 'Exponential Transformation',
				'Square Root Transformation'])
			if st.button('Apply', on_click=callback
				) or st.session_state.button_clicked:
				modified_df = df.copy()
				modified_df, pipe_ = apply_transformation(modified_df, selected_column, transformation_method)
				st.write('Modified DataFrame:')
				print(420)
				st.write(modified_df)
				print(421)
				try:
					if modified_df[selected_column].dtype == 'object':
						col1, col2 = st.columns(2)
						with col1:
							unique_values = modified_df[selected_column].nunique()
							st.write(
								f'  - Data Type: :green[{modified_df[selected_column].dtype}]'
								)
							print(422)
							st.write(
								f'  - Number of Unique Values: :green[{unique_values}]'
								)
							print(423)
							if unique_values <= 20:
								st.write(
									f"  - Unique Values: :green[{', '.join(map(str, modified_df[selected_column].unique()))}]"
									)
								print(424)
							else:
								st.write(f'  - Top 20 Unique Values:')
								print(425)
								st.write(
									f":green[{', '.join(map(str, modified_df[selected_column].value_counts().head(20).index))}]"
									)
								print(426)
						with col2:
							plt.figure(figsize=(10, 6))
							print(427)
							try:
								sns.countplot(x=limit_unique_values(modified_df
									[selected_column]), data=modified_df, color
									='green')
								print(428)
							except:
								sns.countplot(x=modified_df[selected_column],
									data=modified_df, color='green')
								print(429)
							plt.xticks(rotation=45)
							print(430)
							st.pyplot()
							print(431)
							plt.close()
							print(432)
						with st.expander('More Info'):
							tab1, tab2 = st.tabs(['Insights', 'Donut chart'])
							with tab1:
								col7, col8, col9 = st.columns(3)
								with col7:
									st.write('## Insights')
									print(433)
									approximate_distinct_count = modified_df[
										selected_column].nunique()
									approximate_unique_percent = (
										approximate_distinct_count / len(
										modified_df) * 100)
									missing = modified_df[selected_column].isna(
										).sum()
									missing_percent = missing / len(modified_df
										) * 100
									memory_size = modified_df[selected_column
										].memory_usage(deep=True)
									st.write(
										f'Approximate Distinct Count: :green[{approximate_distinct_count}]'
										)
									print(434)
									st.write(
										f'Approximate Unique (%): :green[{approximate_unique_percent:.2f}%]'
										)
									print(435)
									st.write(f'Missing: :green[{missing}]')
									print(436)
									st.write(
										f'Missing (%): :green[{missing_percent:.2f}%]'
										)
									print(437)
									st.write(f'Memory Size: :green[{memory_size}]')
									print(438)
								with col8:
									st.write('## Mode')
									print(439)
									mode = modified_df[selected_column].mode(
										).iloc[0]
									st.write(f'Mode: :green[{mode}]')
									print(440)
								with col9:
									st.write('## First 5 Sample Rows')
									print(441)
									st.write(modified_df[selected_column].head())
									print(442)
							with tab2:
								data = limit_unique_values(modified_df[
									selected_column]).value_counts().reset_index()
								data.columns = [selected_column, 'count']
								fig = px.pie(data, values='count', names=
									selected_column, hole=0.5)
								fig.update_traces(textposition='inside',
									textinfo='percent+label')
								print(443)
								fig.update_layout(legend=dict(orientation='h',
									yanchor='bottom', y=1.02, xanchor='right', x=1)
									)
								print(444)
								st.plotly_chart(fig)
								print(445)
					elif pd.api.types.is_numeric_dtype(modified_df[selected_column]
						):
						col3, col4 = st.columns(2)
						with col3:
							st.write(
								f'  - Data Type: :green[{modified_df[selected_column].dtype}]'
								)
							print(446)
							st.write(
								f'  - Mean: :green[{modified_df[selected_column].mean()}]'
								)
							print(447)
							st.write(
								f'  - Standard Deviation: :green[{modified_df[selected_column].std():.3}]'
								)
							print(448)
							st.write(
								f'  - Min Value: :green[{modified_df[selected_column].min()}]'
								)
							print(449)
							st.write(
								f'  - Max Value: :green[{modified_df[selected_column].max()}]'
								)
							print(450)
						with col4:
							plt.figure(figsize=(10, 6))
							print(451)
							sns.histplot(modified_df[selected_column], kde=True,
								color='green')
							print(452)
							st.pyplot()
							print(453)
							plt.close()
							print(454)
						with st.expander('More Info'):
							tab1, tab2, tab3 = st.tabs(['Insights', 'Box plot',
								'QQ plot'])
							with tab1:
								col4, col5, col6 = st.columns(3)
								with col4:
									st.write('#### Basic Statistics')
									print(455)
									insights = calculate_insights(modified_df[
										selected_column])
									basic_stats = {key: value for key, value in
										insights.items() if key in ['Mean',
										'Median', 'Mode', 'Standard deviation',
										'Variance', 'Kurtosis', 'Skewness']}
									for key, value in basic_stats.items():
										st.write(f'**{key}:** :green[{value:.3f}]')
										print(456)
									st.write(
										f"**Memory size:** :green[{insights.get('Memory size', 'N/A'):.3f}]"
										)
									print(457)
									st.write(
										f"**Range:** :green[{insights.get('Range', 'N/A'):.3f}]"
										)
									print(458)
									st.write(
										f"**Interquartile range:** :green[{insights.get('Interquartile range', 'N/A'):.3f}]"
										)
									print(459)
								with col5:
									st.write('#### Percentiles')
									print(460)
									descriptive_stats = insights.get(
										'Descriptive statistics')
									if descriptive_stats is not None:
										percentiles = descriptive_stats.loc[[
											'min', '25%', '50%', '75%', 'max']]
										if '5%' in descriptive_stats.index:
											percentiles['5%'] = descriptive_stats['5%']
										if '95%' in descriptive_stats.index:
											percentiles['95%'] = descriptive_stats[
												'95%']
										st.write(percentiles)
										print(461)
								with col6:
									st.write('#### Additional Statistics')
									print(462)
									additional_stats = {key: value for key,
										value in insights.items() if key in [
										'Distinct', 'Distinct (%)', 'Missing',
										'Missing (%)', 'Zeros', 'Zeros (%)',
										'Negative', 'Negative (%)']}
									for key, value in additional_stats.items():
										st.write(f'**{key}:** :green[{value:.3f}]')
										print(463)
									st.write(
										f"**Coefficient of variation (CV):** :green[{insights.get('Coefficient of variation (CV)', 'N/A'):.3f}]"
										)
									print(464)
									st.write(
										f"**Median Absolute Deviation (MAD):** :green[{insights.get('Median Absolute Deviation (MAD)', 'N/A'):.3f}]"
										)
									print(465)
									st.write(
										f"**Sum:** :green[{insights.get('Sum', 'N/A'):.3f}]"
										)
									print(466)
							with tab2:
								fig = px.box(modified_df, y=selected_column)
								st.plotly_chart(fig)
								print(467)
							with tab3:
								plt.figure(figsize=(10, 6))
								print(468)
								qqplot_data = sm.qqplot(modified_df[
									selected_column], line='s').gca().lines
								fig = go.Figure()
								fig.add_trace({'type': 'scatter', 'x':
									qqplot_data[0].get_xdata(), 'y':
									qqplot_data[0].get_ydata(), 'mode':
									'markers', 'marker': {'color': '#19d3f3'}})
								print(469)
								fig.add_trace({'type': 'scatter', 'x':
									qqplot_data[1].get_xdata(), 'y':
									qqplot_data[1].get_ydata(), 'mode': 'lines',
									'line': {'color': '#636efa'}})
								print(470)
								x_min = min(qqplot_data[0].get_xdata())
								x_max = max(qqplot_data[0].get_xdata())
								y_min = min(qqplot_data[0].get_ydata())
								y_max = max(qqplot_data[0].get_ydata())
								fig.add_trace(go.Scatter(x=[x_min, x_max], y=[
									y_min, y_max], mode='lines', line=dict(
									color='red', width=2), name='Identity Line'))
								print(471)
								fig.update_layout({'title':
									f'QQ Plot for {selected_column}', 'xaxis':
									{'title': 'Theoretical Quantiles',
									'zeroline': False}, 'yaxis': {'title':
									'Sample Quantiles'}, 'showlegend': False,
									'width': 800, 'height': 700})
								print(472)
								st.plotly_chart(fig)
								print(473)
						
						print(474)
					else:
						st.write('DataFrame not found.')
						print(475)
					print(476)
				except ZeroDivisionError:
					pass
				except Exception as e:
					st.error(e)
					print(477)
					st.subheader('⚠️Please upload a file⚠️')
					print(478)
					pass
				confirm_change = st.button('Confirm Change')
				if confirm_change:
					if 'tran' in st.session_state:
						tran_no = st.session_state.tran
					pipeline7 = {
						"method": f'{transformation_method}',
						"column_name": f'{selected_column}',
					}

					# Save the pipeline to a serialized object
					# try:
					# if type(pipe_) == 'dict':
					
					with open(f"{tran_no}_apply_transformation_on_col.pkl", "wb") as f:
						joblib.dump(pipeline7, f)
					# else:
					st.session_state.full_flow.append(f"{preprocessing_action}->`{selected_column}`")
					# joblib.dump(pipe_, f"{tran_no}_apply_transformation_on_col.pkl")
					st.session_state.df_pre = modified_df
					if tran_no == 0:
						st.session_state.pre_act.append(preprocessing_action)
					tran_no += 1
					st.session_state.tran = tran_no

					st.rerun()


		elif preprocessing_action == 'Dummy Variables :green[(Full DF)]':
			# encoding_method = st.selectbox('Select encoding method', [
			# 	'One-Hot Encoding', 'Label Encoding'])
			st.markdown(
				"""
				<style>
				.green-text {
					color: green;
				}
				</style>
				""",
				unsafe_allow_html=True
			)

			# Using st.markdown with custom CSS class
			st.markdown('- <h2 class="green-text">One-Hot Encoding:</h2>', unsafe_allow_html=True)
			opt = st.toggle('drop_first')
			if st.button('Apply', on_click=callback
				) or st.session_state.button_clicked:
				
				modified_df = df.copy()
				# modified_df = create_dummy_variables(modified_df, encoding_method, opt)
				if opt == True:
					opt = 'first'
				elif opt == False:
					opt = None
				modified_df, encoder_pipeline = create_dummy_variables(modified_df, drop=opt)
				# st.write('Modified DataFrame:')                                                   # PROBLEM
				print(480)
				st.write(modified_df)
				print(481)
				
				confirm_change = st.button('Confirm Change')
				if confirm_change:
					# pipeline8 = {
					# 	"method": f'{encoding_method}',
					# 	"param": opt
					# }

					# Save the pipeline to a serialized object
					# with open("apply_encoding_on_df.pkl", "wb") as f:
					st.session_state.pre_act.append(preprocessing_action)
					st.session_state.full_flow.append(f"{preprocessing_action}")
					joblib.dump(encoder_pipeline, "apply_encoding_on_df.pkl")
					st.session_state.df_pre = modified_df
					st.rerun()
					print(537)

		elif preprocessing_action == 'Dummy Variable :orange[(For Col)]':
			encoding_method = st.selectbox('Select encoding method', [
				'One-Hot Encoding', 'Label Encoding'])
			opt = st.toggle('drop_first')
			if st.button('Apply', on_click=callback
				) or st.session_state.button_clicked:
				modified_df = df.copy()
							# def create_dummy_variables_for_col(df, column_name, drop='first'):
				# modified_df = create_dummy_variables_for_col(modified_df, selected_column, encoding_method, opt)
				if encoding_method == 'One-Hot Encoding':
					encoding_method = 'onehot'
				elif encoding_method == 'Label Encoding':
					encoding_method = 'label'
				if opt == True:
					opt = 'first'
				elif opt == False:
					opt = None
				
				modified_df, encoder_col_pipeline = create_dummy_variables_for_col(df, selected_column, encoding_method, drop=opt)
				# if encoding_method == 'label':
				# 	modified_df, encoder_pipeline, le = create_dummy_variables_for_col(df, selected_column, encoding_method, drop=opt)
				# else:
				# 	modified_df, encoder_pipeline = create_dummy_variables_for_col(df, selected_column, encoding_method, drop=opt)

					
				st.write('Modified DataFrame:')
				print(538)
				st.write(modified_df)
				print(539)
				
				confirm_change = st.button('Confirm Change')
				if confirm_change:
					if 'col_dum' in st.session_state:
						col_dum_no = st.session_state.col_dum
					pipeline9 = {
						"method": encoder_col_pipeline,
						"column_name": f'{selected_column}',
						"param": opt
					}

					# Save the pipeline to a serialized object
					# with open(f"{col_dum_no}_apply_encoding_on_col.pkl", "wb") as f:
					
					joblib.dump(pipeline9, f"{col_dum_no}_apply_encoding_on_col.pkl")
					st.session_state.df_pre = modified_df
					if col_dum_no == 0:
						st.session_state.pre_act.append(preprocessing_action)
					col_dum_no += 1
					st.session_state.full_flow.append(f"{preprocessing_action}->`{selected_column}`")
					st.session_state.col_dum = col_dum_no
					st.rerun()
					print(595)


		elif preprocessing_action == 'Apply Scaling :green[(Full DF)]':
			scaling_method = st.selectbox('Select scaling method', [
				'Standardize Scaling', 'Min-Max Scaling', 'Robust Scaling'], index=1)
			if st.button('Apply', on_click=callback
				) or st.session_state.button_clicked:
				modified_df = df.copy()
				modified_df = apply_scaling(modified_df, scaling_method)
				st.write('Modified DataFrame:')
				print(596)
				st.write(modified_df)
				print(597)
				confirm_change = st.button('Confirm Change')
				if confirm_change:
					pipeline10 = {
						"method": f'{scaling_method}',
					}

					# Save the pipeline to a serialized object
					st.session_state.pre_act.append(preprocessing_action)
					with open("apply_scaling_on_df.pkl", "wb") as f:
						joblib.dump(pipeline10, f)
					st.session_state.df_pre = modified_df
					st.session_state.full_flow.append(f"{preprocessing_action}")
					# modified_df.to_csv("clean.csv", index=False)
					st.rerun()
					print(598)
		elif preprocessing_action == 'Clustering (Unsupervised Learning)':
			st.switch_page(r'pages\10_Unsupervised_learning.py')
		elif preprocessing_action == 'Discretize Variable :orange[(For Col)]':
			bins = st.slider('Select the number of bins', min_value=2,
				max_value=20, value=5)
			strategy = st.selectbox('Select the strategy for binning', [
				'uniform', 'quantile', 'kmeans'])
			if st.button('Apply', on_click=callback
				) or st.session_state.button_clicked:
				modified_df = df.copy()
				modified_df, discretizer_pipeline = discretize_output(modified_df, selected_column,
					bins, strategy)
				# st.write('Modified DataFrame:')
				print(599)
				st.write(modified_df)
				print(600)
				try:
					if modified_df[selected_column].dtype == 'object':
						col1, col2 = st.columns(2)
						with col1:
							unique_values = modified_df[selected_column].nunique()
							st.write(
								f'  - Data Type: :green[{modified_df[selected_column].dtype}]'
								)
							print(601)
							st.write(
								f'  - Number of Unique Values: :green[{unique_values}]'
								)
							print(602)
							if unique_values <= 20:
								st.write(
									f"  - Unique Values: :green[{', '.join(map(str, modified_df[selected_column].unique()))}]"
									)
								print(603)
							else:
								st.write(f'  - Top 20 Unique Values:')
								print(604)
								st.write(
									f":green[{', '.join(map(str, modified_df[selected_column].value_counts().head(20).index))}]"
									)
								print(605)
						with col2:
							plt.figure(figsize=(10, 6))
							print(606)
							try:
								sns.countplot(x=limit_unique_values(modified_df
									[selected_column]), data=modified_df, color
									='green')
								print(607)
							except:
								sns.countplot(x=modified_df[selected_column],
									data=modified_df, color='green')
								print(608)
							plt.xticks(rotation=45)
							print(609)
							st.pyplot()
							print(610)
							plt.close()
							print(611)
						with st.expander('More Info'):
							tab1, tab2 = st.tabs(['Insights', 'Donut chart'])
							with tab1:
								col7, col8, col9 = st.columns(3)
								with col7:
									st.write('## Insights')
									print(612)
									approximate_distinct_count = modified_df[
										selected_column].nunique()
									approximate_unique_percent = (
										approximate_distinct_count / len(
										modified_df) * 100)
									missing = modified_df[selected_column].isna(
										).sum()
									missing_percent = missing / len(modified_df
										) * 100
									memory_size = modified_df[selected_column
										].memory_usage(deep=True)
									st.write(
										f'Approximate Distinct Count: :green[{approximate_distinct_count}]'
										)
									print(613)
									st.write(
										f'Approximate Unique (%): :green[{approximate_unique_percent:.2f}%]'
										)
									print(614)
									st.write(f'Missing: :green[{missing}]')
									print(615)
									st.write(
										f'Missing (%): :green[{missing_percent:.2f}%]'
										)
									print(616)
									st.write(f'Memory Size: :green[{memory_size}]')
									print(617)
								with col8:
									st.write('## Mode')
									print(618)
									mode = modified_df[selected_column].mode(
										).iloc[0]
									st.write(f'Mode: :green[{mode}]')
									print(619)
								with col9:
									st.write('## First 5 Sample Rows')
									print(620)
									st.write(modified_df[selected_column].head())
									print(621)
							with tab2:
								data = limit_unique_values(modified_df[
									selected_column]).value_counts().reset_index()
								data.columns = [selected_column, 'count']
								fig = px.pie(data, values='count', names=
									selected_column, hole=0.5)
								fig.update_traces(textposition='inside',
									textinfo='percent+label')
								print(622)
								fig.update_layout(legend=dict(orientation='h',
									yanchor='bottom', y=1.02, xanchor='right', x=1)
									)
								print(623)
								st.plotly_chart(fig)
								print(624)
					elif pd.api.types.is_numeric_dtype(modified_df[selected_column]
						):
						col3, col4 = st.columns(2)
						with col3:
							st.write(
								f'  - Data Type: :green[{modified_df[selected_column].dtype}]'
								)
							print(625)
							st.write(
								f'  - Mean: :green[{modified_df[selected_column].mean()}]'
								)
							print(626)
							st.write(
								f'  - Standard Deviation: :green[{modified_df[selected_column].std():.3}]'
								)
							print(627)
							st.write(
								f'  - Min Value: :green[{modified_df[selected_column].min()}]'
								)
							print(628)
							st.write(
								f'  - Max Value: :green[{modified_df[selected_column].max()}]'
								)
							print(629)
						with col4:
							plt.figure(figsize=(10, 6))
							print(630)
							sns.histplot(modified_df[selected_column], kde=True,
								color='green')
							print(631)
							st.pyplot()
							print(632)
							plt.close()
							print(633)
						with st.expander('More Info'):
							tab1, tab2, tab3 = st.tabs(['Insights', 'Box plot',
								'QQ plot'])
							with tab1:
								col4, col5, col6 = st.columns(3)
								with col4:
									st.write('#### Basic Statistics')
									print(634)
									insights = calculate_insights(modified_df[
										selected_column])
									basic_stats = {key: value for key, value in
										insights.items() if key in ['Mean',
										'Median', 'Mode', 'Standard deviation',
										'Variance', 'Kurtosis', 'Skewness']}
									for key, value in basic_stats.items():
										st.write(f'**{key}:** :green[{value:.3f}]')
										print(635)
									st.write(
										f"**Memory size:** :green[{insights.get('Memory size', 'N/A'):.3f}]"
										)
									print(636)
									st.write(
										f"**Range:** :green[{insights.get('Range', 'N/A'):.3f}]"
										)
									print(637)
									st.write(
										f"**Interquartile range:** :green[{insights.get('Interquartile range', 'N/A'):.3f}]"
										)
									print(638)
								with col5:
									st.write('#### Percentiles')
									print(639)
									descriptive_stats = insights.get(
										'Descriptive statistics')
									if descriptive_stats is not None:
										percentiles = descriptive_stats.loc[[
											'min', '25%', '50%', '75%', 'max']]
										if '5%' in descriptive_stats.index:
											percentiles['5%'] = descriptive_stats['5%']
										if '95%' in descriptive_stats.index:
											percentiles['95%'] = descriptive_stats[
												'95%']
										st.write(percentiles)
										print(640)
								with col6:
									st.write('#### Additional Statistics')
									print(641)
									additional_stats = {key: value for key,
										value in insights.items() if key in [
										'Distinct', 'Distinct (%)', 'Missing',
										'Missing (%)', 'Zeros', 'Zeros (%)',
										'Negative', 'Negative (%)']}
									for key, value in additional_stats.items():
										st.write(f'**{key}:** :green[{value:.3f}]')
										print(642)
									st.write(
										f"**Coefficient of variation (CV):** :green[{insights.get('Coefficient of variation (CV)', 'N/A'):.3f}]"
										)
									print(643)
									st.write(
										f"**Median Absolute Deviation (MAD):** :green[{insights.get('Median Absolute Deviation (MAD)', 'N/A'):.3f}]"
										)
									print(644)
									st.write(
										f"**Sum:** :green[{insights.get('Sum', 'N/A'):.3f}]"
										)
									print(645)
							with tab2:
								fig = px.box(modified_df, y=selected_column)
								st.plotly_chart(fig)
								print(646)
							with tab3:
								plt.figure(figsize=(10, 6))
								print(647)
								qqplot_data = sm.qqplot(modified_df[
									selected_column], line='s').gca().lines
								fig = go.Figure()
								fig.add_trace({'type': 'scatter', 'x':
									qqplot_data[0].get_xdata(), 'y':
									qqplot_data[0].get_ydata(), 'mode':
									'markers', 'marker': {'color': '#19d3f3'}})
								print(648)
								fig.add_trace({'type': 'scatter', 'x':
									qqplot_data[1].get_xdata(), 'y':
									qqplot_data[1].get_ydata(), 'mode': 'lines',
									'line': {'color': '#636efa'}})
								print(649)
								x_min = min(qqplot_data[0].get_xdata())
								x_max = max(qqplot_data[0].get_xdata())
								y_min = min(qqplot_data[0].get_ydata())
								y_max = max(qqplot_data[0].get_ydata())
								fig.add_trace(go.Scatter(x=[x_min, x_max], y=[
									y_min, y_max], mode='lines', line=dict(
									color='red', width=2), name='Identity Line'))
								print(650)
								fig.update_layout({'title':
									f'QQ Plot for {selected_column}', 'xaxis':
									{'title': 'Theoretical Quantiles',
									'zeroline': False}, 'yaxis': {'title':
									'Sample Quantiles'}, 'showlegend': False,
									'width': 800, 'height': 700})
								print(651)
								st.plotly_chart(fig)
								print(652)
						
						print(653)
					else:
						st.write('DataFrame not found.')
						print(654)
					print(655)
				except ZeroDivisionError:
					pass
				except Exception as e:
					st.error(e)
					print(656)
					st.subheader('⚠️Please upload a file⚠️')
					print(657)
					pass
				confirm_change = st.button('Confirm Change')
				if confirm_change:
					if 'des_var' in st.session_state:
						des_var_no = st.session_state.des_var
					pipeline11 = {
						"column_name": f'{selected_column}',
						"bins": bins,
						"strategy": discretizer_pipeline
					}
					# with open(f"{des_var_no}_discretize_output_col.pkl", "wb") as f:
					# 	joblib.dump(pipeline11, f)
					# joblib.dump(discretizer_pipeline, f"{des_var_no}_discretize_output_col.pkl")
					
					joblib.dump(pipeline11, f"{des_var_no}_discretize_output_col.pkl")
					st.session_state.df_pre = modified_df
					
					if des_var_no == 0:
						st.session_state.pre_act.append(preprocessing_action)
					des_var_no += 1
					st.session_state.full_flow.append(f"{preprocessing_action}->`{selected_column}`")
					st.session_state.des_var = des_var_no
					st.rerun()
					print(658)

		elif preprocessing_action == 'Column Unique Value Replacement :orange[(For Col)]':
			print(43)
			print(659)
			select = st.selectbox('Convert type:', ['Convert to Str:',
				'Convert to int:', 'Convert to float:'])
			print(44)
			print(660)
			distinct_values = df[selected_column].unique()
			print(45)
			print(661)
			replacements = {}
			for value in distinct_values:
				print(46)
				print(662)
				if select == 'Convert to Str:':
					print(47)
					print(663)
					replacements[value] = st.text_input(f"Replace '{value}' with:", value=str(value))
				elif select == 'Convert to int:':
					try:
						print(48)
						print(664)
						replacements[value] = st.number_input(
							f"Replace '{value}' with:", value=int(value), step=1)
					except:
						print(49)
						print(665)
						replacements[value] = st.number_input(
							f"Replace '{value}' with:", step=1)
				elif select == 'Convert to float:':
					try:
						print(50)
						print(666)
						replacements[value] = st.number_input(f"Replace '{value}' with:", value=float(value),step=0.01)
					except:
						print(51)
						print(667)
						replacements[value] = st.number_input(f"Replace '{value}' with:", step=0.01)
			if st.button('Apply', on_click=callback
				) or st.session_state.button_clicked:
				st.warning(
					f'"{select}" is applied, but changes will not be saved until you press the \'Confirm Changes\' button below.'
					)
				print(668)
				print(52)
				print(669)
				modified_df = df.copy()
				print(53)
				print(670)
				print("replacements:=_", replacements)
				modified_df[selected_column].replace(replacements, inplace=True)
				print(671)
				print(54)
				print(672)
				st.write('Modified DataFrame:')
				print(673)
				print(55)
				print(674)
				st.write(modified_df)
				print(675)
				print(56)
				print(676)
				if select == 'Convert to int:':
					print(57)
					print(677)
					modified_df[selected_column] = modified_df[selected_column].astype(pd.Int64Dtype())
					print(58)
					print(678)
				elif select == 'Convert to float:':
					print(59)
					print(679)
					modified_df[selected_column] = modified_df[selected_column
						].astype(float)
				try:
					print(60)
					print(680)
					if modified_df[selected_column].dtype == 'object':
						print(61)
						print(681)
						col1, col2 = st.columns(2)
						with col1:
							print(62)
							print(682)
							unique_values = modified_df[selected_column].nunique()
							print(63)
							print(683)
							st.write(
								f'  - Data Type: :green[{modified_df[selected_column].dtype}]'
								)
							print(684)
							print(64)
							print(685)
							st.write(
								f'  - Number of Unique Values: :green[{unique_values}]'
								)
							print(686)
							print(65)
							print(687)
							if unique_values <= 20:
								print(66)
								print(688)
								st.write(
									f"  - Unique Values: :green[{', '.join(map(str, modified_df[selected_column].unique()))}]"
									)
								print(689)
								print(67)
								print(690)
							else:
								print(68)
								print(691)
								st.write(f'  - Top 20 Unique Values:')
								print(692)
								print(69)
								print(693)
								st.write(
									f":green[{', '.join(map(str, modified_df[selected_column].value_counts().head(20).index))}]"
									)
								print(694)
								print(70)
								print(695)
						with col2:
							print(71)
							print(696)
							plt.figure(figsize=(10, 6))
							print(697)
							try:
								print(72)
								print(698)
								sns.countplot(x=limit_unique_values(modified_df
									[selected_column]), data=modified_df, color
									='green')
								print(699)
							except:
								print(73)
								print(700)
								sns.countplot(x=modified_df[selected_column],
									data=modified_df, color='green')
								print(701)
							plt.xticks(rotation=45)
							print(702)
							st.pyplot()
							print(703)
							plt.close()
							print(704)
							print(74)
							print(705)
					elif pd.api.types.is_numeric_dtype(modified_df[selected_column]
						):
						print(83)
						print(706)
						col3, col4 = st.columns(2)
						with col3:
							print(84)
							print(707)
							st.write(
								f'  - Data Type: :green[{modified_df[selected_column].dtype}]'
								)
							print(708)
							st.write(
								f'  - Mean: :green[{modified_df[selected_column].mean()}]'
								)
							print(709)
							st.write(
								f'  - Standard Deviation: :green[{modified_df[selected_column].std():.3}]'
								)
							print(710)
							st.write(
								f'  - Min Value: :green[{modified_df[selected_column].min()}]'
								)
							print(711)
							st.write(
								f'  - Max Value: :green[{modified_df[selected_column].max()}]'
								)
							print(712)
						with col4:
							print(85)
							print(713)
							plt.figure(figsize=(10, 6))
							print(714)
							sns.histplot(modified_df[selected_column], kde=True,
								color='green')
							print(715)
							st.pyplot()
							print(716)
							plt.close()
							print(717)
						
						print(718)
						print(106)
						print(719)
					else:
						print(107)
						print(720)
						st.write('DataFrame not found.')
						print(721)
						print(108)
						print(722)
					print(723)
					print(109)
					print(724)
				except ZeroDivisionError:
					print(110)
					print(725)
					pass
				except Exception as e:
					print(111)
					print(726)
					st.error(e)
					print(727)
					st.subheader('⚠️Please upload a file⚠️')
					print(728)
					pass
				confirm_change = st.button('Confirm Change')
				if confirm_change:
					if 'out_rplac' in st.session_state:
						out_rplac_no = st.session_state.out_rplac
					pipeline12 = {
						"column_name": f'{selected_column}',
						"replacements": replacements,
						"select": select
					}

					# Save the pipeline to a serialized object
					
					with open(f"{out_rplac_no}_column_unique_value_replacement.pkl", "wb") as f:
						joblib.dump(pipeline12, f)
					st.session_state.df_pre = modified_df
					if out_rplac_no == 0:
						st.session_state.pre_act.append(preprocessing_action)
					out_rplac_no += 1
					st.session_state.full_flow.append(f"{preprocessing_action}->`{selected_column}`")
					st.session_state.out_rplac = out_rplac_no
					st.rerun()
					print(729)


	except Exception as e:
		print('e:=_0', e)
		print(730)
		st.error(e)
		print(731)
elif sup_unsup == 'Supervised Learning':
	# st.warning("Hi")
	try:
		# st.write('Session State:->', st.session_state['shared'])
		preprocessing_action = st.sidebar.radio('Select preprocessing action', [
		'Select Output Variable ↗️ & Select Pre-Processing Step⬇️', 'Drop Column :orange[(For Col)]', 'Treat Missing :orange[(For Col)]',
		'Treat Missing :green[(Full DF)]', 'Change Data Type :orange[(For Col)]',
		'Treat Outliers :orange[(For Col)]', 'Treat Outliers :green[(Full DF)]', 'Drop Duplicates :green[(Full DF)]', 'Apply Transformation :orange[(For Col)]',
		'Column Unique Value Replacement :orange[(For Col)]', 'Discretize Variable :orange[(For Col)]',
		'Dummy Variables :green[(Full DF)]', 'Dummy Variable :orange[(For Col)]', 'Apply Scaling :green[(Full DF)]', 'Split Data'], help=":orange[(For Col)] means -> operation on particular Column, :green[(Full DF)] means -> operation on full DataFrame")
		st.sidebar.write(":orange[Confirmed Changes] ⤵️")
		if len(st.session_state.full_flow) == 0:
			non = ["Not Confirmed any Changes Yet"]
			st.sidebar.write(non)
		else:
			st.sidebar.write(st.session_state.full_flow)
		print(54)
		if 'df_pre' in st.session_state:
			# oragnial_df = st.session_state.df
			
			# df_to_pre = st.session_state.df_pre
			
			# df = df_to_pre
			
			# selected_column = st.selectbox('Select a column to Pre-Process', df.columns)
			
			# Filter out 'Variable 2' from the list


			# to_select = st.selectbox("Select Data Frame (Recommended: DF to Pre-Process)", ["df_Y", "df_to_pre_process"], index=1)
			# if to_select == "df_Y":
			# 	# df = oragnial_df
			# 	df_to_pre = st.session_state.df_pre
			# 	y_var = st.session_state.y_var
			# 	df = pd.DataFrame(df_to_pre[y_var],)
			# elif to_select == "df_to_pre_process":
			
			df_to_pre = st.session_state.df_pre
			df = df_to_pre


			if preprocessing_action == 'Select Output Variable ↗️ & Select Pre-Processing Step⬇️':
				# selected_column = st.selectbox('Select a column to Pre-Process', df.columns)
				dlist = list(df.columns)
				if "y_var" not in st.session_state:
					selected_column = st.selectbox("⚠️ Select Y variable⚠️(Select Output Variable)⚠️:", dlist, index=len(dlist)-1)

					y_var = selected_column
					st.session_state.y_var = y_var
				else:
					y_var = st.session_state.y_var
					selected_column = st.selectbox("⚠️ Select Y variable⚠️(Select Output Variable)⚠️:", dlist, index=dlist.index(y_var))
					y_var = selected_column
					st.session_state.y_var = y_var
				if y_var is not None:
					
					y_nul = df[y_var].isnull().sum()
					if y_nul >= 1:
						st.sidebar.warning("The Y variable contains missing values. Make sure to treat it with the `Treat Missing :orange[(For Col)]`", icon="🚨")
			
			
			if preprocessing_action == 'Drop Column :orange[(For Col)]':

				y_var = st.session_state.y_var
				dlist = list(df.columns)
				filtered_dlist = [var for var in dlist if var != y_var]
				if len(filtered_dlist) == 0:
					st.warning("Can Not Drop Y variable")
					st.stop()
				selected_column = st.selectbox("DF without Y variable:", filtered_dlist, index=0)

			elif preprocessing_action == 'Treat Missing :orange[(For Col)]':

				selected_column = st.selectbox('Select a column to Pre-Process', df.columns)

			elif preprocessing_action == 'Treat Missing :green[(Full DF)]':
				
				selected_column = st.selectbox('Select a column to Pre-Process', df.columns)

			elif preprocessing_action == 'Change Data Type :orange[(For Col)]':
				
				selected_column = st.selectbox('Select a column to Pre-Process', df.columns)

			elif preprocessing_action == 'Drop Duplicates :green[(Full DF)]':

				y_var = st.session_state.y_var
				dlist = list(df.columns)
				filtered_dlist = [var for var in dlist if var != y_var]
				if len(filtered_dlist) == 0:
					st.warning("Can Not Drop Y variable")
					st.stop()

				selected_column = st.selectbox('Select a column to Pre-Process', df.columns)

			elif preprocessing_action == 'Treat Outliers :orange[(For Col)]':

				selected_column = st.selectbox('Select a column to Pre-Process', df.columns)

			elif preprocessing_action == 'Treat Outliers :green[(Full DF)]':

				selected_column = st.selectbox('Select a column to Pre-Process', df.columns)

			elif preprocessing_action == 'Apply Transformation :orange[(For Col)]':

				selected_column = st.selectbox('Select a column to Pre-Process', df.columns)

			elif preprocessing_action == 'Column Unique Value Replacement :orange[(For Col)]':

				selected_column = st.selectbox('Select a column to Pre-Process', df.columns)

			elif preprocessing_action == 'Discretize Variable :orange[(For Col)]':

				selected_column = st.selectbox('Select a column to Pre-Process', df.columns)
			
			elif preprocessing_action == 'Dummy Variables :green[(Full DF)]':

				selected_column = st.selectbox('Select a column to Pre-Process', df.columns)

			elif preprocessing_action == 'Dummy Variable :orange[(For Col)]':

				selected_column = st.selectbox('Select a column to Pre-Process', df.columns)

			elif preprocessing_action == 'Apply Scaling :green[(Full DF)]':

				selected_column = st.selectbox('Select a column to Pre-Process', df.columns)			
			

			# st.warning("kl")
			# to_select = st.selectbox("Select Data Frame (Recommended: DF to Pre-Process)", ["oragnial_df", "df_to_pre_process"], index=1)
			# if to_select == "oragnial_df":
			# 	df = oragnial_df
			# elif to_select == "df_to_pre_process":
			
			
			print(1)
			print(55)
			
			print(2)
			print(56)
			html_content = (
				f"<div class='column-header'>Insights for column:<code>{selected_column}</code></div>"
				)
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
			st.markdown(css_style, unsafe_allow_html=True)
			print(57)
			st.markdown(html_content, unsafe_allow_html=True)
			print(58)
			print(3)
			print(59)
			if df[selected_column].dtype == 'object':
				col1, col2 = st.columns(2)
				print(4)
				print(60)
				with col1:
					unique_values = df[selected_column].nunique()
					print(5)
					print(61)
					st.write(f'  - Data Type: :green[{df[selected_column].dtype}]')
					print(62)
					print(6)
					print(63)
					st.write(
						f'  - Number of Unique Values: :green[{unique_values}]')
					print(64)
					print(7)
					print(65)
					if unique_values <= 20:
						print(8)
						print(66)
						st.write(
							f"  - Unique Values: :green[{', '.join(map(str, df[selected_column].unique()))}]"
							)
						print(67)
					else:
						print(9)
						print(68)
						st.write(f'  - Top 20 Unique Values:')
						print(69)
						print(10)
						print(70)
						st.write(
							f":green[{', '.join(map(str, df[selected_column].value_counts().head(20).index))}]"
							)
						print(71)
				with col2:
					print(10)
					print(72)
					plt.figure(figsize=(10, 6))
					print(73)
					print(11)
					print(74)
					try:
						print(12)
						print(75)
						sns.countplot(x=limit_unique_values(df[selected_column]
							), data=df, color='green')
						print(76)
					except:
						print(13)
						print(77)
						sns.countplot(x=df[selected_column], data=df, color='green'
							)
						print(78)
					plt.xticks(rotation=45)
					print(79)
					st.pyplot()
					print(80)
					plt.close()
					print(81)
				with st.expander('More Info'):
					print(14)
					print(82)
					tab1, tab2 = st.tabs(['Insights', 'Donut chart'])
					print(15)
					print(83)
					with tab1:
						print(16)
						print(84)
						col7, col8, col9 = st.columns(3)
						with col7:
							print(17)
							print(85)
							st.write('## Insights')
							print(86)
							approximate_distinct_count = df[selected_column
								].nunique()
							approximate_unique_percent = (
								approximate_distinct_count / len(df) * 100)
							missing = df[selected_column].isna().sum()
							missing_percent = missing / len(df) * 100
							memory_size = df[selected_column].memory_usage(deep
								=True)
							st.write(
								f'Approximate Distinct Count: :green[{approximate_distinct_count}]'
								)
							print(87)
							st.write(
								f'Approximate Unique (%): :green[{approximate_unique_percent:.2f}%]'
								)
							print(88)
							st.write(f'Missing: :green[{missing}]')
							print(89)
							st.write(f'Missing (%): :green[{missing_percent:.2f}%]'
								)
							print(90)
							st.write(f'Memory Size: :green[{memory_size}]')
							print(91)
							print(18)
							print(92)
						with col8:
							print(19)
							print(93)
							st.write('## Mode')
							print(94)
							mode = df[selected_column].mode().iloc[0]
							st.write(f'Mode: :green[{mode}]')
							print(95)
							print(20)
							print(96)
						with col9:
							print(21)
							print(97)
							st.write('## First 5 Sample Rows')
							print(98)
							st.write(df[selected_column].head())
							print(99)
							print(22)
							print(100)
					with tab2:
						print(23)
						print(101)
						data = limit_unique_values(df[selected_column]
							).value_counts().reset_index()
						data.columns = [selected_column, 'count']
						fig = px.pie(data, values='count', names=
							selected_column, hole=0.5)
						fig.update_traces(textposition='inside', textinfo=
							'percent+label')
						print(102)
						fig.update_layout(legend=dict(orientation='h', yanchor=
							'bottom', y=1.02, xanchor='right', x=1))
						print(103)
						st.plotly_chart(fig)
						print(104)
						print(24)
						print(105)
			elif pd.api.types.is_numeric_dtype(df[selected_column]):
				print(25)
				print(106)
				col3, col4 = st.columns(2)
				print(26)
				print(107)
				with col3:
					print(27)
					print(108)
					st.write(f'  - Data Type: :green[{df[selected_column].dtype}]')
					print(109)
					st.write(f'  - Mean: :green[{df[selected_column].mean()}]')
					print(110)
					st.write(
						f'  - Standard Deviation: :green[{df[selected_column].std()}]'
						)
					print(111)
					st.write(f'  - Min Value: :green[{df[selected_column].min()}]')
					print(112)
					st.write(f'  - Max Value: :green[{df[selected_column].max()}]')
					print(113)
					print(28)
					print(114)
				with col4:
					plt.figure(figsize=(10, 6))
					print(115)
					sns.histplot(df[selected_column], kde=True, color='green')
					print(116)
					st.pyplot()
					print(117)
					plt.close()
					print(118)
					print(29)
					print(119)
				with st.expander('More Info'):
					print(30)
					print(120)
					tab1, tab2, tab3 = st.tabs(['Insights', 'Box plot', 'QQ plot'])
					with tab1:
						print(31)
						print(121)
						col4, col5, col6 = st.columns(3)
						with col4:
							print(32)
							print(122)
							st.write('#### Basic Statistics')
							print(123)
							insights = calculate_insights(df[selected_column])
							basic_stats = {key: value for key, value in
								insights.items() if key in ['Mean', 'Median',
								'Mode', 'Standard deviation', 'Variance',
								'Kurtosis', 'Skewness']}
							for key, value in basic_stats.items():
								st.write(f'**{key}:** :green[{value:.3f}]')
								print(124)
							st.write(
								f"**Memory size:** :green[{insights.get('Memory size', 'N/A'):.3f}]"
								)
							print(125)
							st.write(
								f"**Range:** :green[{insights.get('Range', 'N/A'):.3f}]"
								)
							print(126)
							st.write(
								f"**Interquartile range:** :green[{insights.get('Interquartile range', 'N/A'):.3f}]"
								)
							print(127)
							print(33)
							print(128)
						with col5:
							print(34)
							print(129)
							st.write('#### Percentiles')
							print(130)
							descriptive_stats = insights.get(
								'Descriptive statistics')
							if descriptive_stats is not None:
								percentiles = descriptive_stats.loc[['min',
									'25%', '50%', '75%', 'max']]
								if '5%' in descriptive_stats.index:
									percentiles['5%'] = descriptive_stats['5%']
								if '95%' in descriptive_stats.index:
									percentiles['95%'] = descriptive_stats['95%']
								st.write(percentiles)
								print(131)
							print(35)
							print(132)
						with col6:
							print(36)
							print(133)
							st.write('#### Additional Statistics')
							print(134)
							additional_stats = {key: value for key, value in
								insights.items() if key in ['Distinct',
								'Distinct (%)', 'Missing', 'Missing (%)',
								'Zeros', 'Zeros (%)', 'Negative', 'Negative (%)']}
							for key, value in additional_stats.items():
								st.write(f'**{key}:** :green[{value:.3f}]')
								print(135)
							st.write(
								f"**Coefficient of variation (CV):** :green[{insights.get('Coefficient of variation (CV)', 'N/A'):.3f}]"
								)
							print(136)
							st.write(
								f"**Median Absolute Deviation (MAD):** :green[{insights.get('Median Absolute Deviation (MAD)', 'N/A'):.3f}]"
								)
							print(137)
							st.write(
								f"**Sum:** :green[{insights.get('Sum', 'N/A'):.3f}]"
								)
							print(138)
							print(37)
							print(139)
					with tab2:
						print(38)
						print(140)
						fig = px.box(df, y=selected_column)
						st.plotly_chart(fig)
						print(141)
						print(39)
						print(142)
					with tab3:
						print(40)
						print(143)
						plt.figure(figsize=(10, 6))
						print(144)
						qqplot_data = sm.qqplot(df[selected_column], line='s').gca(
							).lines
						fig = go.Figure()
						fig.add_trace({'type': 'scatter', 'x': qqplot_data[0].
							get_xdata(), 'y': qqplot_data[0].get_ydata(),
							'mode': 'markers', 'marker': {'color': '#19d3f3'}})
						print(145)
						fig.add_trace({'type': 'scatter', 'x': qqplot_data[1].
							get_xdata(), 'y': qqplot_data[1].get_ydata(),
							'mode': 'lines', 'line': {'color': '#636efa'}})
						print(146)
						x_min = min(qqplot_data[0].get_xdata())
						x_max = max(qqplot_data[0].get_xdata())
						y_min = min(qqplot_data[0].get_ydata())
						y_max = max(qqplot_data[0].get_ydata())
						fig.add_trace(go.Scatter(x=[x_min, x_max], y=[y_min,
							y_max], mode='lines', line=dict(color='red', width=
							2), name='Identity Line'))
						print(147)
						fig.update_layout({'title':
							f'QQ Plot for {selected_column}', 'xaxis': {'title':
							'Theoretical Quantiles', 'zeroline': False},
							'yaxis': {'title': 'Sample Quantiles'},
							'showlegend': False, 'width': 800, 'height': 700})
						print(148)
						st.plotly_chart(fig)
						print(149)
						print(41)
						print(150)
		else:
			st.write('DataFrame not found.')
			print(151)
	except ZeroDivisionError:
		pass
	except Exception as e:
		st.error(e)
		print(152)
		st.subheader('⚠️Please upload a file⚠️')
		print(153)
		pass
	print(42)
	print(154)
	try:
		if preprocessing_action == 'Select Output Variable ↗️ & Select Pre-Processing Step⬇️':
			print('You have successfuly reached Pre-Proceaaing Phase')
			st.dataframe(df)
			print(155)
		elif preprocessing_action == 'Change Data Type :orange[(For Col)]':
			dtype_options = ['int', 'int32', 'Int32', 'int64', 'Int64', 'float',
				'float32', 'Float32', 'float64', 'Float64', 'str', 'bool']
			new_dtype = st.selectbox('Select new data type', dtype_options)
			if st.button('Apply', on_click=callback
				) or st.session_state.button_clicked:
				modified_df = df.copy()
				modified_df = change_dtype(modified_df, selected_column, new_dtype)
				print(156)
				st.write('Modified DataFrame:')
				print(157)
				st.write(modified_df)
				print(158)
				try:
					if modified_df[selected_column].dtype == 'object':
						col1, col2 = st.columns(2)
						with col1:
							unique_values = modified_df[selected_column].nunique()
							st.write(
								f'  - Data Type: :green[{modified_df[selected_column].dtype}]'
								)
							print(159)
							st.write(
								f'  - Number of Unique Values: :green[{unique_values}]'
								)
							print(160)
							if unique_values <= 20:
								st.write(
									f"  - Unique Values: :green[{', '.join(map(str, modified_df[selected_column].unique()))}]"
									)
								print(161)
							else:
								st.write(f'  - Top 20 Unique Values:')
								print(162)
								st.write(
									f":green[{', '.join(map(str, modified_df[selected_column].value_counts().head(20).index))}]"
									)
								print(163)
						with col2:
							plt.figure(figsize=(10, 6))
							print(164)
							try:
								sns.countplot(x=limit_unique_values(modified_df
									[selected_column]), data=modified_df, color
									='green')
								print(165)
							except:
								sns.countplot(x=modified_df[selected_column],
									data=modified_df, color='green')
								print(166)
							plt.xticks(rotation=45)
							print(167)
							st.pyplot()
							print(168)
							plt.close()
							print(169)
						with st.expander('More Info'):
							tab1, tab2 = st.tabs(['Insights', 'Donut chart'])
							with tab1:
								col7, col8, col9 = st.columns(3)
								with col7:
									st.write('## Insights')
									print(170)
									approximate_distinct_count = modified_df[
										selected_column].nunique()
									approximate_unique_percent = (
										approximate_distinct_count / len(
										modified_df) * 100)
									missing = modified_df[selected_column].isna(
										).sum()
									missing_percent = missing / len(modified_df
										) * 100
									memory_size = modified_df[selected_column
										].memory_usage(deep=True)
									st.write(
										f'Approximate Distinct Count: :green[{approximate_distinct_count}]'
										)
									print(171)
									st.write(
										f'Approximate Unique (%): :green[{approximate_unique_percent:.2f}%]'
										)
									print(172)
									st.write(f'Missing: :green[{missing}]')
									print(173)
									st.write(
										f'Missing (%): :green[{missing_percent:.2f}%]'
										)
									print(174)
									st.write(f'Memory Size: :green[{memory_size}]')
									print(175)
								with col8:
									st.write('## Mode')
									print(176)
									mode = modified_df[selected_column].mode(
										).iloc[0]
									st.write(f'Mode: :green[{mode}]')
									print(177)
								with col9:
									st.write('## First 5 Sample Rows')
									print(178)
									st.write(modified_df[selected_column].head())
									print(179)
							with tab2:
								data = limit_unique_values(modified_df[
									selected_column]).value_counts().reset_index()
								data.columns = [selected_column, 'count']
								fig = px.pie(data, values='count', names=
									selected_column, hole=0.5)
								fig.update_traces(textposition='inside',
									textinfo='percent+label')
								print(180)
								fig.update_layout(legend=dict(orientation='h',
									yanchor='bottom', y=1.02, xanchor='right', x=1)
									)
								print(181)
								st.plotly_chart(fig)
								print(182)
					elif pd.api.types.is_numeric_dtype(modified_df[selected_column]
						):
						col3, col4 = st.columns(2)
						with col3:
							st.write(
								f'  - Data Type: :green[{modified_df[selected_column].dtype}]'
								)
							print(183)
							st.write(
								f'  - Mean: :green[{modified_df[selected_column].mean()}]'
								)
							print(184)
							st.write(
								f'  - Standard Deviation: :green[{modified_df[selected_column].std():.3}]'
								)
							print(185)
							st.write(
								f'  - Min Value: :green[{modified_df[selected_column].min()}]'
								)
							print(186)
							st.write(
								f'  - Max Value: :green[{modified_df[selected_column].max()}]'
								)
							print(187)
						with col4:
							plt.figure(figsize=(10, 6))
							print(188)
							sns.histplot(modified_df[selected_column], kde=True,
								color='green')
							print(189)
							st.pyplot()
							print(190)
							plt.close()
							print(191)
						with st.expander('More Info'):
							tab1, tab2, tab3 = st.tabs(['Insights', 'Box plot',
								'QQ plot'])
							with tab1:
								col4, col5, col6 = st.columns(3)
								with col4:
									st.write('#### Basic Statistics')
									print(192)
									try:
										insights = calculate_insights(modified_df[selected_column])
									
										basic_stats = {key: value for key, value in
											insights.items() if key in ['Mean',
											'Median', 'Mode', 'Standard deviation',
											'Variance', 'Kurtosis', 'Skewness']}
										for key, value in basic_stats.items():
											st.write(f'**{key}:** :green[{value:.3f}]')
											print(193)
										st.write(
											f"**Memory size:** :green[{insights.get('Memory size', 'N/A'):.3f}]"
											)
										print(194)
										st.write(
											f"**Range:** :green[{insights.get('Range', 'N/A'):.3f}]"
											)
										print(195)
										st.write(
											f"**Interquartile range:** :green[{insights.get('Interquartile range', 'N/A'):.3f}]"
											)
										print(196)
									except Exception as e:
										pass
								with col5:
									st.write('#### Percentiles')
									print(197)
									descriptive_stats = insights.get(
										'Descriptive statistics')
									if descriptive_stats is not None:
										percentiles = descriptive_stats.loc[[
											'min', '25%', '50%', '75%', 'max']]
										if '5%' in descriptive_stats.index:
											percentiles['5%'] = descriptive_stats['5%']
										if '95%' in descriptive_stats.index:
											percentiles['95%'] = descriptive_stats[
												'95%']
										st.write(percentiles)
										print(198)
								with col6:
									st.write('#### Additional Statistics')
									print(199)
									additional_stats = {key: value for key,
										value in insights.items() if key in [
										'Distinct', 'Distinct (%)', 'Missing',
										'Missing (%)', 'Zeros', 'Zeros (%)',
										'Negative', 'Negative (%)']}
									for key, value in additional_stats.items():
										st.write(f'**{key}:** :green[{value:.3f}]')
										print(200)
									st.write(
										f"**Coefficient of variation (CV):** :green[{insights.get('Coefficient of variation (CV)', 'N/A'):.3f}]"
										)
									print(201)
									st.write(
										f"**Median Absolute Deviation (MAD):** :green[{insights.get('Median Absolute Deviation (MAD)', 'N/A'):.3f}]"
										)
									print(202)
									st.write(
										f"**Sum:** :green[{insights.get('Sum', 'N/A'):.3f}]"
										)
									print(203)
							with tab2:
								fig = px.box(modified_df, y=selected_column)
								st.plotly_chart(fig)
								print(204)
							with tab3:
								plt.figure(figsize=(10, 6))
								print(205)
								qqplot_data = sm.qqplot(modified_df[
									selected_column], line='s').gca().lines
								fig = go.Figure()
								fig.add_trace({'type': 'scatter', 'x':
									qqplot_data[0].get_xdata(), 'y':
									qqplot_data[0].get_ydata(), 'mode':
									'markers', 'marker': {'color': '#19d3f3'}})
								print(206)
								fig.add_trace({'type': 'scatter', 'x':
									qqplot_data[1].get_xdata(), 'y':
									qqplot_data[1].get_ydata(), 'mode': 'lines',
									'line': {'color': '#636efa'}})
								print(207)
								x_min = min(qqplot_data[0].get_xdata())
								x_max = max(qqplot_data[0].get_xdata())
								y_min = min(qqplot_data[0].get_ydata())
								y_max = max(qqplot_data[0].get_ydata())
								fig.add_trace(go.Scatter(x=[x_min, x_max], y=[
									y_min, y_max], mode='lines', line=dict(
									color='red', width=2), name='Identity Line'))
								print(208)
								fig.update_layout({'title':
									f'QQ Plot for {selected_column}', 'xaxis':
									{'title': 'Theoretical Quantiles',
									'zeroline': False}, 'yaxis': {'title':
									'Sample Quantiles'}, 'showlegend': False,
									'width': 800, 'height': 700})
								print(209)
								st.plotly_chart(fig)
								print(210)
						
						print(211)
					else:
						st.write('DataFrame not found.')
						print(212)
					print(213)
				except ZeroDivisionError:
					pass
				except Exception as e:
					# st.error(e)
					print(214)
					st.subheader('⚠️Please upload a file⚠️')
					print(215)
					pass
				confirm_change = st.button('Confirm Change')
				if confirm_change:
					if 'type_ch' in st.session_state:
						type_ch_no = st.session_state.type_ch
					st.session_state.df_pre = modified_df
					pipeline = {'column': f'{selected_column}', 'new_dtype':
						f'{new_dtype}'}
					with open(f'{type_ch_no}_datatype_pipeline.pkl', 'wb') as f:
						joblib.dump(pipeline, f)
						print(216)
					
					
					if type_ch_no == 0:
						st.session_state.pre_act.append(preprocessing_action)
					type_ch_no += 1
					st.session_state.full_flow.append(f"{preprocessing_action}->`{selected_column}`")
					st.session_state.type_ch = type_ch_no
					st.rerun()
					print(217)

		elif preprocessing_action == 'Drop Column :orange[(For Col)]':
			if st.button('Apply', on_click=callback
				) or st.session_state.button_clicked:
				modified_df = df.copy()
				drop_column(modified_df, selected_column)
				print(218)
				st.write('Modified DataFrame:')
				print(219)
				st.write(modified_df)
				print(220)
				try:
					if modified_df[selected_column].dtype == 'object':
						col1, col2 = st.columns(2)
						with col1:
							unique_values = modified_df[selected_column].nunique()
							st.write(
								f'  - Data Type: :green[{modified_df[selected_column].dtype}]'
								)
							print(221)
							st.write(
								f'  - Number of Unique Values: :green[{unique_values}]'
								)
							print(222)
							if unique_values <= 20:
								st.write(
									f"  - Unique Values: :green[{', '.join(map(str, modified_df[selected_column].unique()))}]"
									)
								print(223)
							else:
								st.write(f'  - Top 20 Unique Values:')
								print(224)
								st.write(
									f":green[{', '.join(map(str, modified_df[selected_column].value_counts().head(20).index))}]"
									)
								print(225)
						with col2:
							plt.figure(figsize=(10, 6))
							print(226)
							try:
								sns.countplot(x=limit_unique_values(modified_df
									[selected_column]), data=modified_df, color
									='green')
								print(227)
							except:
								sns.countplot(x=modified_df[selected_column],
									data=modified_df, color='green')
								print(228)
							plt.xticks(rotation=45)
							print(229)
							st.pyplot()
							print(230)
							plt.close()
							print(231)
						with st.expander('More Info'):
							tab1, tab2 = st.tabs(['Insights', 'Donut chart'])
							with tab1:
								col7, col8, col9 = st.columns(3)
								with col7:
									st.write('## Insights')
									print(232)
									approximate_distinct_count = modified_df[
										selected_column].nunique()
									approximate_unique_percent = (
										approximate_distinct_count / len(
										modified_df) * 100)
									missing = modified_df[selected_column].isna(
										).sum()
									missing_percent = missing / len(modified_df
										) * 100
									memory_size = modified_df[selected_column
										].memory_usage(deep=True)
									st.write(
										f'Approximate Distinct Count: :green[{approximate_distinct_count}]'
										)
									print(233)
									st.write(
										f'Approximate Unique (%): :green[{approximate_unique_percent:.2f}%]'
										)
									print(234)
									st.write(f'Missing: :green[{missing}]')
									print(235)
									st.write(
										f'Missing (%): :green[{missing_percent:.2f}%]'
										)
									print(236)
									st.write(f'Memory Size: :green[{memory_size}]')
									print(237)
								with col8:
									st.write('## Mode')
									print(238)
									mode = modified_df[selected_column].mode(
										).iloc[0]
									st.write(f'Mode: :green[{mode}]')
									print(239)
								with col9:
									st.write('## First 5 Sample Rows')
									print(240)
									st.write(modified_df[selected_column].head())
									print(241)
							with tab2:
								data = limit_unique_values(modified_df[
									selected_column]).value_counts().reset_index()
								data.columns = [selected_column, 'count']
								fig = px.pie(data, values='count', names=
									selected_column, hole=0.5)
								fig.update_traces(textposition='inside',
									textinfo='percent+label')
								print(242)
								fig.update_layout(legend=dict(orientation='h',
									yanchor='bottom', y=1.02, xanchor='right', x=1)
									)
								print(243)
								st.plotly_chart(fig)
								print(244)
					elif pd.api.types.is_numeric_dtype(modified_df[selected_column]
						):
						col3, col4 = st.columns(2)
						with col3:
							st.write(
								f'  - Data Type: :green[{modified_df[selected_column].dtype}]'
								)
							print(245)
							st.write(
								f'  - Mean: :green[{modified_df[selected_column].mean()}]'
								)
							print(246)
							st.write(
								f'  - Standard Deviation: :green[{modified_df[selected_column].std():.3}]'
								)
							print(247)
							st.write(
								f'  - Min Value: :green[{modified_df[selected_column].min()}]'
								)
							print(248)
							st.write(
								f'  - Max Value: :green[{modified_df[selected_column].max()}]'
								)
							print(249)
						with col4:
							plt.figure(figsize=(10, 6))
							print(250)
							sns.histplot(modified_df[selected_column], kde=True,
								color='green')
							print(251)
							st.pyplot()
							print(252)
							plt.close()
							print(253)
						with st.expander('More Info'):
							tab1, tab2, tab3 = st.tabs(['Insights', 'Box plot',
								'QQ plot'])
							with tab1:
								col4, col5, col6 = st.columns(3)
								with col4:
									st.write('#### Basic Statistics')
									print(254)
									insights = calculate_insights(modified_df[
										selected_column])
									basic_stats = {key: value for key, value in
										insights.items() if key in ['Mean',
										'Median', 'Mode', 'Standard deviation',
										'Variance', 'Kurtosis', 'Skewness']}
									for key, value in basic_stats.items():
										st.write(f'**{key}:** :green[{value:.3f}]')
										print(255)
									st.write(
										f"**Memory size:** :green[{insights.get('Memory size', 'N/A'):.3f}]"
										)
									print(256)
									st.write(
										f"**Range:** :green[{insights.get('Range', 'N/A'):.3f}]"
										)
									print(257)
									st.write(
										f"**Interquartile range:** :green[{insights.get('Interquartile range', 'N/A'):.3f}]"
										)
									print(258)
								with col5:
									st.write('#### Percentiles')
									print(259)
									descriptive_stats = insights.get(
										'Descriptive statistics')
									if descriptive_stats is not None:
										percentiles = descriptive_stats.loc[[
											'min', '25%', '50%', '75%', 'max']]
										if '5%' in descriptive_stats.index:
											percentiles['5%'] = descriptive_stats['5%']
										if '95%' in descriptive_stats.index:
											percentiles['95%'] = descriptive_stats[
												'95%']
										st.write(percentiles)
										print(260)
								with col6:
									st.write('#### Additional Statistics')
									print(261)
									additional_stats = {key: value for key,
										value in insights.items() if key in [
										'Distinct', 'Distinct (%)', 'Missing',
										'Missing (%)', 'Zeros', 'Zeros (%)',
										'Negative', 'Negative (%)']}
									for key, value in additional_stats.items():
										st.write(f'**{key}:** :green[{value:.3f}]')
										print(262)
									st.write(
										f"**Coefficient of variation (CV):** :green[{insights.get('Coefficient of variation (CV)', 'N/A'):.3f}]"
										)
									print(263)
									st.write(
										f"**Median Absolute Deviation (MAD):** :green[{insights.get('Median Absolute Deviation (MAD)', 'N/A'):.3f}]"
										)
									print(264)
									st.write(
										f"**Sum:** :green[{insights.get('Sum', 'N/A'):.3f}]"
										)
									print(265)
							with tab2:
								fig = px.box(modified_df, y=selected_column)
								st.plotly_chart(fig)
								print(266)
							with tab3:
								plt.figure(figsize=(10, 6))
								print(267)
								qqplot_data = sm.qqplot(modified_df[
									selected_column], line='s').gca().lines
								fig = go.Figure()
								fig.add_trace({'type': 'scatter', 'x':
									qqplot_data[0].get_xdata(), 'y':
									qqplot_data[0].get_ydata(), 'mode':
									'markers', 'marker': {'color': '#19d3f3'}})
								print(268)
								fig.add_trace({'type': 'scatter', 'x':
									qqplot_data[1].get_xdata(), 'y':
									qqplot_data[1].get_ydata(), 'mode': 'lines',
									'line': {'color': '#636efa'}})
								print(269)
								x_min = min(qqplot_data[0].get_xdata())
								x_max = max(qqplot_data[0].get_xdata())
								y_min = min(qqplot_data[0].get_ydata())
								y_max = max(qqplot_data[0].get_ydata())
								fig.add_trace(go.Scatter(x=[x_min, x_max], y=[
									y_min, y_max], mode='lines', line=dict(
									color='red', width=2), name='Identity Line'))
								print(270)
								fig.update_layout({'title':
									f'QQ Plot for {selected_column}', 'xaxis':
									{'title': 'Theoretical Quantiles',
									'zeroline': False}, 'yaxis': {'title':
									'Sample Quantiles'}, 'showlegend': False,
									'width': 800, 'height': 700})
								print(271)
								st.plotly_chart(fig)
								print(272)
						
						print(273)
					else:
						st.write('DataFrame not found.')
						print(274)
					print(275)
				except ZeroDivisionError:
					pass
				except Exception as e:
					pass
				confirm_change = st.button('Confirm Change')
				if confirm_change:
					if 'drop' in st.session_state:
						drop_no = st.session_state.drop
					pipeline1 = {'column': f'{selected_column}'}
					with open(f'{drop_no}_drop_pipeline.pkl', 'wb') as f:
						joblib.dump(pipeline1, f)
						print(276)
					st.session_state.df_pre = modified_df
					
					if drop_no == 0:
						st.session_state.pre_act.append(preprocessing_action)
					drop_no += 1
					st.session_state.full_flow.append(f"{preprocessing_action} -> `{selected_column}`")
					st.session_state.drop = drop_no
					# modified_df.to_csv("clean.csv", index=False)
					st.rerun()
					print(277)
		elif preprocessing_action == 'Drop Duplicates :green[(Full DF)]':
			if st.button('Apply', on_click=callback
				) or st.session_state.button_clicked:
				modified_df = df.copy()
				drop_duplicates(modified_df)
				print(278)
				st.write('Modified DataFrame:')
				print(279)
				st.write(modified_df)
				print(280)
				confirm_change = st.button('Confirm Change')
				if confirm_change:
					pipeline2 = {'action': 'drop_duplicates'}
					with open('drop_dup_pipeline.pkl', 'wb') as f:
						joblib.dump(pipeline2, f)
						print(281)
					st.session_state.pre_act.append(preprocessing_action)
					st.session_state.full_flow.append(f"{preprocessing_action}")
					st.session_state.df_pre = modified_df
					st.rerun()
					print(282)
		elif preprocessing_action == 'Treat Outliers :orange[(For Col)]':
			# outlier_method = st.radio('Select Outlier Treatment Method', [
			# 	'Delete Outliers', 'Winsorization'])
			# Custom CSS for green text
			st.markdown(
				"""
				<style>
				.green-text {
					color: green;
				}
				</style>
				""",
				unsafe_allow_html=True
			)

			# Using st.markdown with custom CSS class
			st.markdown('- <h2 class="green-text">Winsorization:</h2>', unsafe_allow_html=True)
			if st.button('Apply', on_click=callback
				) or st.session_state.button_clicked:
				modified_df = df.copy()
				# if outlier_method == 'Winsorization':
				# 	lower_limit = st.slider('Select Lower Limit', min_value=0.0,
				# 		max_value=0.1, step=0.01, value=0.05)
				# 	upper_limit = st.slider('Select Upper Limit', min_value=0.0,
				# 		max_value=0.1, step=0.01, value=0.05)
				# else:
				# 	lower_limit = 0.05
				# 	upper_limit = 0.05
				# if outlier_method == 'Delete Outliers':
				# 	z_score_threshold = st.slider('Select Z-Score Threshold',
				# 		min_value=1.0, max_value=5.0, step=0.1, value=3.0)
				# else:
				# 	z_score_threshold = 3.0
				# modified_df = apply_winsorization_col(modified_df, selected_column, outlier_method, z_score_threshold, lower_limit, upper_limit)

				default_capping_method = 'iqr'
				default_tail = 'both'
				default_fold = 1.5

				# User inputs for configuring Winsorizer
				capping_method = st.selectbox('Capping Method', ['iqr', 'gaussian', 'percentile', 'mad'], index=0)
				tail = st.selectbox('Tail', ['both', 'left', 'right'], index=0)
				fold = st.number_input('Fold', min_value=0.1, max_value=10.0, value=default_fold, step=0.1)
				
				y_var = st.session_state.y_var
				preprocessor, modified_df = apply_winsorization_col_sup(modified_df, selected_column, y_var, capping_method, tail, fold)
				st.write('Modified DataFrame:')
				print(283)
				st.write(modified_df)
				print(284)
				try:
					if modified_df[selected_column].dtype == 'object':
						col1, col2 = st.columns(2)
						with col1:
							unique_values = modified_df[selected_column].nunique()
							st.write(
								f'  - Data Type: :green[{modified_df[selected_column].dtype}]'
								)
							print(285)
							st.write(
								f'  - Number of Unique Values: :green[{unique_values}]'
								)
							print(286)
							if unique_values <= 20:
								st.write(
									f"  - Unique Values: :green[{', '.join(map(str, modified_df[selected_column].unique()))}]"
									)
								print(287)
							else:
								st.write(f'  - Top 20 Unique Values:')
								print(288)
								st.write(
									f":green[{', '.join(map(str, modified_df[selected_column].value_counts().head(20).index))}]"
									)
								print(289)
						with col2:
							plt.figure(figsize=(10, 6))
							print(290)
							try:
								sns.countplot(x=limit_unique_values(modified_df
									[selected_column]), data=modified_df, color
									='green')
								print(291)
							except:
								sns.countplot(x=modified_df[selected_column],
									data=modified_df, color='green')
								print(292)
							plt.xticks(rotation=45)
							print(293)
							st.pyplot()
							print(294)
							plt.close()
							print(295)
						with st.expander('More Info'):
							tab1, tab2 = st.tabs(['Insights', 'Donut chart'])
							with tab1:
								col7, col8, col9 = st.columns(3)
								with col7:
									st.write('## Insights')
									print(296)
									approximate_distinct_count = modified_df[
										selected_column].nunique()
									approximate_unique_percent = (
										approximate_distinct_count / len(
										modified_df) * 100)
									missing = modified_df[selected_column].isna(
										).sum()
									missing_percent = missing / len(modified_df
										) * 100
									memory_size = modified_df[selected_column
										].memory_usage(deep=True)
									st.write(
										f'Approximate Distinct Count: :green[{approximate_distinct_count}]'
										)
									print(297)
									st.write(
										f'Approximate Unique (%): :green[{approximate_unique_percent:.2f}%]'
										)
									print(298)

									st.write(f'Missing: :green[{missing}]')
									print(299)
									st.write(
										f'Missing (%): :green[{missing_percent:.2f}%]'
										)
									print(300)
									st.write(f'Memory Size: :green[{memory_size}]')
									print(301)
								with col8:
									st.write('## Mode')
									print(302)
									mode = modified_df[selected_column].mode(
										).iloc[0]
									st.write(f'Mode: :green[{mode}]')
									print(303)
								with col9:
									st.write('## First 5 Sample Rows')
									print(304)
									st.write(modified_df[selected_column].head())
									print(305)
							with tab2:
								data = limit_unique_values(modified_df[
									selected_column]).value_counts().reset_index()
								data.columns = [selected_column, 'count']
								fig = px.pie(data, values='count', names=
									selected_column, hole=0.5)
								fig.update_traces(textposition='inside',
									textinfo='percent+label')
								print(306)
								fig.update_layout(legend=dict(orientation='h',
									yanchor='bottom', y=1.02, xanchor='right', x=1)
									)
								print(307)
								st.plotly_chart(fig)
								print(308)
					elif pd.api.types.is_numeric_dtype(modified_df[selected_column]
						):
						col3, col4 = st.columns(2)
						with col3:
							st.write(
								f'  - Data Type: :green[{modified_df[selected_column].dtype}]'
								)
							print(309)
							st.write(
								f'  - Mean: :green[{modified_df[selected_column].mean()}]'
								)
							print(310)
							st.write(
								f'  - Standard Deviation: :green[{modified_df[selected_column].std():.3}]'
								)
							print(311)
							st.write(
								f'  - Min Value: :green[{modified_df[selected_column].min()}]'
								)
							print(312)
							st.write(
								f'  - Max Value: :green[{modified_df[selected_column].max()}]'
								)
							print(313)
						with col4:
							plt.figure(figsize=(10, 6))
							print(314)
							sns.histplot(modified_df[selected_column], kde=True,
								color='green')
							print(315)
							st.pyplot()
							print(316)
							plt.close()
							print(317)
						with st.expander('More Info'):
							tab1, tab2, tab3 = st.tabs(['Insights', 'Box plot',
								'QQ plot'])
							with tab1:
								col4, col5, col6 = st.columns(3)
								with col4:
									st.write('#### Basic Statistics')
									print(318)
									insights = calculate_insights(modified_df[
										selected_column])
									basic_stats = {key: value for key, value in
										insights.items() if key in ['Mean',
										'Median', 'Mode', 'Standard deviation',
										'Variance', 'Kurtosis', 'Skewness']}
									for key, value in basic_stats.items():
										st.write(f'**{key}:** :green[{value:.3f}]')
										print(319)
									st.write(
										f"**Memory size:** :green[{insights.get('Memory size', 'N/A'):.3f}]"
										)
									print(320)
									st.write(
										f"**Range:** :green[{insights.get('Range', 'N/A'):.3f}]"
										)
									print(321)
									st.write(
										f"**Interquartile range:** :green[{insights.get('Interquartile range', 'N/A'):.3f}]"
										)
									print(322)
								with col5:
									st.write('#### Percentiles')
									print(323)
									descriptive_stats = insights.get(
										'Descriptive statistics')
									if descriptive_stats is not None:
										percentiles = descriptive_stats.loc[[
											'min', '25%', '50%', '75%', 'max']]
										if '5%' in descriptive_stats.index:
											percentiles['5%'] = descriptive_stats['5%']
										if '95%' in descriptive_stats.index:
											percentiles['95%'] = descriptive_stats[
												'95%']
										st.write(percentiles)
										print(324)
								with col6:
									st.write('#### Additional Statistics')
									print(325)
									additional_stats = {key: value for key,
										value in insights.items() if key in [
										'Distinct', 'Distinct (%)', 'Missing',
										'Missing (%)', 'Zeros', 'Zeros (%)',
										'Negative', 'Negative (%)']}
									for key, value in additional_stats.items():
										st.write(f'**{key}:** :green[{value:.3f}]')
										print(326)
									st.write(
										f"**Coefficient of variation (CV):** :green[{insights.get('Coefficient of variation (CV)', 'N/A'):.3f}]"
										)
									print(327)
									st.write(
										f"**Median Absolute Deviation (MAD):** :green[{insights.get('Median Absolute Deviation (MAD)', 'N/A'):.3f}]"
										)
									print(328)
									st.write(
										f"**Sum:** :green[{insights.get('Sum', 'N/A'):.3f}]"
										)
									print(329)
							with tab2:
								fig = px.box(modified_df, y=selected_column)
								st.plotly_chart(fig)
								print(330)
							with tab3:
								plt.figure(figsize=(10, 6))
								print(331)
								qqplot_data = sm.qqplot(modified_df[
									selected_column], line='s').gca().lines
								fig = go.Figure()
								fig.add_trace({'type': 'scatter', 'x':
									qqplot_data[0].get_xdata(), 'y':
									qqplot_data[0].get_ydata(), 'mode':
									'markers', 'marker': {'color': '#19d3f3'}})
								print(332)
								fig.add_trace({'type': 'scatter', 'x':
									qqplot_data[1].get_xdata(), 'y':
									qqplot_data[1].get_ydata(), 'mode': 'lines',
									'line': {'color': '#636efa'}})
								print(333)
								x_min = min(qqplot_data[0].get_xdata())
								x_max = max(qqplot_data[0].get_xdata())
								y_min = min(qqplot_data[0].get_ydata())
								y_max = max(qqplot_data[0].get_ydata())
								fig.add_trace(go.Scatter(x=[x_min, x_max], y=[
									y_min, y_max], mode='lines', line=dict(
									color='red', width=2), name='Identity Line'))
								print(334)
								fig.update_layout({'title':
									f'QQ Plot for {selected_column}', 'xaxis':
									{'title': 'Theoretical Quantiles',
									'zeroline': False}, 'yaxis': {'title':
									'Sample Quantiles'}, 'showlegend': False,
									'width': 800, 'height': 700})
								print(335)
								st.plotly_chart(fig)
								print(336)
						
						print(337)
					else:
						st.write('DataFrame not found.')
						print(338)
					print(339)
				except ZeroDivisionError:
					pass
				except Exception as e:
					st.error(e)
					print(340)
					st.subheader('⚠️Please upload a file⚠️')
					print(341)
					pass
				confirm_change = st.button('Confirm Change')
				if confirm_change:
					if 'treat_out' in st.session_state:
						treat_out_no = st.session_state.treat_out
					joblib.dump(preprocessor, f'{treat_out_no}_outliers_for_col_pipeline.pkl')
					print(342)
					st.session_state.df_pre = modified_df
					if treat_out_no == 0:
						st.session_state.pre_act.append(preprocessing_action)
					treat_out_no += 1
					st.session_state.full_flow.append(f"{preprocessing_action}->`{selected_column}`")
					st.session_state.treat_out = treat_out_no
					st.rerun()
					print(343)
					
					
		elif preprocessing_action == 'Treat Outliers :green[(Full DF)]':
			# method = st.selectbox('Select Method to Treat Outliers', [
			# 	'Delete Outliers', 'Winsorization', 'IQR'])
			st.markdown(
				"""
				<style>
				.green-text {
					color: green;
				}
				</style>
				""",
				unsafe_allow_html=True
			)

			# Using st.markdown with custom CSS class
			st.markdown('- <h2 class="green-text">Winsorization on Full DataFrame:</h2>', unsafe_allow_html=True)
			if st.button('Apply', on_click=callback
				) or st.session_state.button_clicked:
				modified_df = df.copy()
				# if method == 'Delete Outliers':
				# 	z_score_threshold = st.slider('Select Z-Score Threshold',
				# 		min_value=1.0, max_value=5.0, step=0.1, value=3.0)
				# else:
				# 	z_score_threshold = 3.0
				# if method == 'Winsorization':
				# 	lower_limit = st.slider('Select Lower Limit:', min_value=
				# 		0.0, max_value=0.1, step=0.01, value=0.05)
				# 	upper_limit = st.slider('Select Upper Limit:', min_value=
				# 		0.0, max_value=0.1, step=0.01, value=0.05)
				# else:
				# 	lower_limit = upper_limit = 0.05
				# modified_df = treat_outliers_full_df(modified_df, method,
				# 	z_score_threshold=3, lower_limit=lower_limit, upper_limit=
				# 	upper_limit)

					# Default parameter values
				default_capping_method = 'iqr'
				default_tail = 'both'
				default_fold = 1.5

				# User inputs for configuring Winsorizer
				capping_method = st.selectbox('Capping Method', ['iqr', 'gaussian', 'percentile', 'mad'], index=0)
				tail = st.selectbox('Tail', ['both', 'left', 'right'], index=0)
				fold = st.number_input('Fold', min_value=0.1, max_value=10.0, value=default_fold, step=0.1)

				y_var = st.session_state.y_var
				pipeline, modified_df = apply_winsorization_for_full_df(modified_df, y_var, capping_method, tail, fold)
				st.write('Modified DataFrame:')
				print(344)
				st.write(modified_df)
				print(345)
				confirm_change = st.button('Confirm Change')
				if confirm_change:
					st.session_state.df_pre = modified_df
					# pipeline4 = {'method': f'{method}', 'lower_limit':
					# 	f'{lower_limit}', 'upper_limit': f'{upper_limit}',
					# 	'z_score_threshold': f'{z_score_threshold}'}
					st.session_state.pre_act.append(preprocessing_action)
					st.session_state.full_flow.append(f"{preprocessing_action}")
					joblib.dump(pipeline, 'outliers_for_full_df_pipeline.pkl')
					# with open('outliers_for_full_df_pipeline.pkl', 'wb') as f:
					# 	joblib.dump(pipeline4, f)
					print(346)
					st.rerun()
					print(347)
		elif preprocessing_action == 'Treat Missing :orange[(For Col)]':
			missing_method = st.radio('Select Missing Value Treatment Method',
				['Mean Imputation',
				'Median Imputation', 'Mode Imputation'])
			if st.button('Apply', on_click=callback
				) or st.session_state.button_clicked:
				modified_df = df.copy()
				modified_df, Imputer = treat_missing_values(modified_df, selected_column,
					missing_method)
				st.write('Modified DataFrame:')
				print(348)
				st.write(modified_df)
				print(349)
				try:
					if modified_df[selected_column].dtype == 'object':
						col1, col2 = st.columns(2)
						with col1:
							unique_values = modified_df[selected_column].nunique()
							st.write(
								f'  - Data Type: :green[{modified_df[selected_column].dtype}]'
								)
							print(350)
							st.write(
								f'  - Number of Unique Values: :green[{unique_values}]'
								)
							print(351)
							if unique_values <= 20:
								st.write(
									f"  - Unique Values: :green[{', '.join(map(str, modified_df[selected_column].unique()))}]"
									)
								print(352)
							else:
								st.write(f'  - Top 20 Unique Values:')
								print(353)
								st.write(
									f":green[{', '.join(map(str, modified_df[selected_column].value_counts().head(20).index))}]"
									)
								print(354)
						with col2:
							plt.figure(figsize=(10, 6))
							print(355)
							try:
								sns.countplot(x=limit_unique_values(modified_df
									[selected_column]), data=modified_df, color
									='green')
								print(356)
							except:
								sns.countplot(x=modified_df[selected_column],
									data=modified_df, color='green')
								print(357)
							plt.xticks(rotation=45)
							print(358)
							st.pyplot()
							print(359)
							plt.close()
							print(360)
						with st.expander('More Info'):
							tab1, tab2 = st.tabs(['Insights', 'Donut chart'])
							with tab1:
								col7, col8, col9 = st.columns(3)
								with col7:
									st.write('## Insights')
									print(361)
									approximate_distinct_count = modified_df[
										selected_column].nunique()
									approximate_unique_percent = (
										approximate_distinct_count / len(
										modified_df) * 100)
									missing = modified_df[selected_column].isna(
										).sum()
									missing_percent = missing / len(modified_df
										) * 100
									memory_size = modified_df[selected_column
										].memory_usage(deep=True)
									st.write(
										f'Approximate Distinct Count: :green[{approximate_distinct_count}]'
										)
									print(362)
									st.write(
										f'Approximate Unique (%): :green[{approximate_unique_percent:.2f}%]'
										)
									print(363)
									st.write(f'Missing: :green[{missing}]')
									print(364)
									st.write(
										f'Missing (%): :green[{missing_percent:.2f}%]'
										)
									print(365)
									st.write(f'Memory Size: :green[{memory_size}]')
									print(366)
								with col8:
									st.write('## Mode')
									print(367)
									mode = modified_df[selected_column].mode(
										).iloc[0]
									st.write(f'Mode: :green[{mode}]')
									print(368)
								with col9:
									st.write('## First 5 Sample Rows')
									print(369)
									st.write(modified_df[selected_column].head())
									print(370)
							with tab2:
								data = limit_unique_values(modified_df[
									selected_column]).value_counts().reset_index()
								data.columns = [selected_column, 'count']
								fig = px.pie(data, values='count', names=
									selected_column, hole=0.5)
								fig.update_traces(textposition='inside',
									textinfo='percent+label')
								print(371)
								fig.update_layout(legend=dict(orientation='h',
									yanchor='bottom', y=1.02, xanchor='right', x=1)
									)
								print(372)
								st.plotly_chart(fig)
								print(373)
					elif pd.api.types.is_numeric_dtype(modified_df[selected_column]
						):
						col3, col4 = st.columns(2)
						with col3:
							st.write(
								f'  - Data Type: :green[{modified_df[selected_column].dtype}]'
								)
							print(374)
							st.write(
								f'  - Mean: :green[{modified_df[selected_column].mean()}]'
								)
							print(375)
							st.write(
								f'  - Standard Deviation: :green[{modified_df[selected_column].std():.3}]'
								)
							print(376)
							st.write(
								f'  - Min Value: :green[{modified_df[selected_column].min()}]'
								)
							print(377)
							st.write(
								f'  - Max Value: :green[{modified_df[selected_column].max()}]'
								)
							print(378)
						with col4:
							plt.figure(figsize=(10, 6))
							print(379)
							sns.histplot(modified_df[selected_column], kde=True,
								color='green')
							print(380)
							st.pyplot()
							print(381)
							plt.close()
							print(382)
						with st.expander('More Info'):
							tab1, tab2, tab3 = st.tabs(['Insights', 'Box plot',
								'QQ plot'])
							with tab1:
								col4, col5, col6 = st.columns(3)
								with col4:
									st.write('#### Basic Statistics')
									print(383)
									insights = calculate_insights(modified_df[
										selected_column])
									basic_stats = {key: value for key, value in
										insights.items() if key in ['Mean',
										'Median', 'Mode', 'Standard deviation',
										'Variance', 'Kurtosis', 'Skewness']}
									for key, value in basic_stats.items():
										st.write(f'**{key}:** :green[{value:.3f}]')
										print(384)
									st.write(
										f"**Memory size:** :green[{insights.get('Memory size', 'N/A'):.3f}]"
										)
									print(385)
									st.write(
										f"**Range:** :green[{insights.get('Range', 'N/A'):.3f}]"
										)
									print(386)
									st.write(
										f"**Interquartile range:** :green[{insights.get('Interquartile range', 'N/A'):.3f}]"
										)
									print(387)
								with col5:
									st.write('#### Percentiles')
									print(388)
									descriptive_stats = insights.get(
										'Descriptive statistics')
									if descriptive_stats is not None:
										percentiles = descriptive_stats.loc[[
											'min', '25%', '50%', '75%', 'max']]
										if '5%' in descriptive_stats.index:
											percentiles['5%'] = descriptive_stats['5%']
										if '95%' in descriptive_stats.index:
											percentiles['95%'] = descriptive_stats[
												'95%']
										st.write(percentiles)
										print(389)
								with col6:
									st.write('#### Additional Statistics')
									print(390)
									additional_stats = {key: value for key,
										value in insights.items() if key in [
										'Distinct', 'Distinct (%)', 'Missing',
										'Missing (%)', 'Zeros', 'Zeros (%)',
										'Negative', 'Negative (%)']}
									for key, value in additional_stats.items():
										st.write(f'**{key}:** :green[{value:.3f}]')
										print(391)
									st.write(
										f"**Coefficient of variation (CV):** :green[{insights.get('Coefficient of variation (CV)', 'N/A'):.3f}]"
										)
									print(392)
									st.write(
										f"**Median Absolute Deviation (MAD):** :green[{insights.get('Median Absolute Deviation (MAD)', 'N/A'):.3f}]"
										)
									print(393)
									st.write(
										f"**Sum:** :green[{insights.get('Sum', 'N/A'):.3f}]"
										)
									print(394)
							with tab2:
								fig = px.box(modified_df, y=selected_column)
								st.plotly_chart(fig)
								print(395)
							with tab3:
								plt.figure(figsize=(10, 6))
								print(396)
								qqplot_data = sm.qqplot(modified_df[
									selected_column], line='s').gca().lines
								fig = go.Figure()
								fig.add_trace({'type': 'scatter', 'x':
									qqplot_data[0].get_xdata(), 'y':
									qqplot_data[0].get_ydata(), 'mode':
									'markers', 'marker': {'color': '#19d3f3'}})
								print(397)
								fig.add_trace({'type': 'scatter', 'x':
									qqplot_data[1].get_xdata(), 'y':
									qqplot_data[1].get_ydata(), 'mode': 'lines',
									'line': {'color': '#636efa'}})
								print(398)
								x_min = min(qqplot_data[0].get_xdata())
								x_max = max(qqplot_data[0].get_xdata())
								y_min = min(qqplot_data[0].get_ydata())
								y_max = max(qqplot_data[0].get_ydata())
								fig.add_trace(go.Scatter(x=[x_min, x_max], y=[
									y_min, y_max], mode='lines', line=dict(
									color='red', width=2), name='Identity Line'))
								print(399)
								fig.update_layout({'title':
									f'QQ Plot for {selected_column}', 'xaxis':
									{'title': 'Theoretical Quantiles',
									'zeroline': False}, 'yaxis': {'title':
									'Sample Quantiles'}, 'showlegend': False,
									'width': 800, 'height': 700})
								print(400)
								st.plotly_chart(fig)
								print(401)
						
						print(402)
					else:
						st.write('DataFrame not found.')
						print(403)
					print(404)
				except ZeroDivisionError:
					pass
				except Exception as e:
					st.error(e)
					print(405)
					st.subheader('⚠️Please upload a file⚠️')
					print(406)
					pass
				confirm_change = st.button('Confirm Change')
				if confirm_change:
					if 'miss' in st.session_state:
						miss_no = st.session_state.miss
					# if missing_method == 'Delete Missing Values':
						pipeline5 = {'method': Imputer, 'column':f'{selected_column}'}
						
					# 	with open(f'{miss_no}_treat_missing_vaues_for_col.pkl', 'wb') as f:
					# 		joblib.dump(pipeline5, f)
					# 		print(407)
					# 	st.session_state.df_pre = modified_df
					# 	miss_no += 1
					# 	st.session_state.miss = miss_no
					# 	st.rerun()
					# 	print(408)
					# else:
						joblib.dump(pipeline5, f'{miss_no}_treat_missing_vaues_for_col.pkl')
						st.session_state.df_pre = modified_df
						if miss_no == 0:
							st.session_state.pre_act.append(preprocessing_action)
						miss_no += 1
						st.session_state.full_flow.append(f"{preprocessing_action}->`{selected_column}`")
						st.session_state.miss = miss_no
						st.rerun()


		elif preprocessing_action == 'Treat Missing :green[(Full DF)]':
			print(409)
			numeric_columns = df.select_dtypes(include=np.number).columns
			categorical_columns = df.select_dtypes(include=['object', 'category']
				).columns
			numeric_treatment = st.selectbox(
				'Select treatment for missing values in numeric columns', [
				'Mean', 'Median', 'Mode', 'Random'])
			categorical_treatment = st.selectbox(
				'Select treatment for missing values in categorical columns', [
				'Mode', 'Random'])
			if st.button('Apply', on_click=callback
				) or st.session_state.button_clicked:
				modified_df = df.copy()
				y_var = st.session_state.y_var
				transformed_df, numeric_pipeline, categorical_pipeline = create_full_df_impute_pipelines_for_sup(modified_df, y_var, numeric_treatment, categorical_treatment)
				st.warning("Missing Value Treatment has been applied on full DF, but the changes will not be saved until you press the 'Confirm Changes' button below.")
				print(415)
				st.write('Modified DataFrame:')
				print(416)
				st.write(modified_df)
				# numeric_pipeline, categorical_pipeline = pipelines
				print(417)
				confirm_change = st.button('Confirm Change')
				if confirm_change:
					# pipeline6 = {'numeric_treatment': f'{numeric_treatment}',
					# 	'numeric_strategy': f'{numeric_strategy}',
					# 	'categorical_treatment': f'{categorical_treatment}',
					# 	'categorical_strategy': f'{categorical_strategy}'}
					# Saving the pipelines
					st.session_state.pre_act.append(preprocessing_action)

					joblib.dump(numeric_pipeline, 'numeric_treat_missing_vaues_in_full_df.pkl')
					joblib.dump(categorical_pipeline, 'categorical_treat_missing_vaues_in_full_df.pkl')
					st.session_state.full_flow.append(f"{preprocessing_action}")
					# with open('treat_missing_vaues_in_full_df.pkl', 'wb') as f:
					# 	joblib.dump(pipeline6, f)
					print(418)
					st.session_state.df_pre = modified_df
					st.rerun()
					print(419)
				

		elif preprocessing_action == 'Apply Transformation :orange[(For Col)]':
			# transformation_method = st.selectbox('Select Transformation Technique',
			# 	['Log Transformation', 'Exponential Transformation',
			# 	'Square Root Transformation', 'Box-Cox Transformation',
			# 	'Yeo-Johnson Transformation'])
			transformation_method = st.selectbox('Select Transformation Technique',
				['Log Transformation', 'Exponential Transformation',
				'Square Root Transformation'])
			if st.button('Apply', on_click=callback
				) or st.session_state.button_clicked:
				modified_df = df.copy()
				modified_df, pipe_ = apply_transformation(modified_df, selected_column, transformation_method)
				st.write('Modified DataFrame:')
				print(420)
				st.write(modified_df)
				print(421)
				try:
					if modified_df[selected_column].dtype == 'object':
						col1, col2 = st.columns(2)
						with col1:
							unique_values = modified_df[selected_column].nunique()
							st.write(
								f'  - Data Type: :green[{modified_df[selected_column].dtype}]'
								)
							print(422)
							st.write(
								f'  - Number of Unique Values: :green[{unique_values}]'
								)
							print(423)
							if unique_values <= 20:
								st.write(
									f"  - Unique Values: :green[{', '.join(map(str, modified_df[selected_column].unique()))}]"
									)
								print(424)
							else:
								st.write(f'  - Top 20 Unique Values:')
								print(425)
								st.write(
									f":green[{', '.join(map(str, modified_df[selected_column].value_counts().head(20).index))}]"
									)
								print(426)
						with col2:
							plt.figure(figsize=(10, 6))
							print(427)
							try:
								sns.countplot(x=limit_unique_values(modified_df
									[selected_column]), data=modified_df, color
									='green')
								print(428)
							except:
								sns.countplot(x=modified_df[selected_column],
									data=modified_df, color='green')
								print(429)
							plt.xticks(rotation=45)
							print(430)
							st.pyplot()
							print(431)
							plt.close()
							print(432)
						with st.expander('More Info'):
							tab1, tab2 = st.tabs(['Insights', 'Donut chart'])
							with tab1:
								col7, col8, col9 = st.columns(3)
								with col7:
									st.write('## Insights')
									print(433)
									approximate_distinct_count = modified_df[
										selected_column].nunique()
									approximate_unique_percent = (
										approximate_distinct_count / len(
										modified_df) * 100)
									missing = modified_df[selected_column].isna(
										).sum()
									missing_percent = missing / len(modified_df
										) * 100
									memory_size = modified_df[selected_column
										].memory_usage(deep=True)
									st.write(
										f'Approximate Distinct Count: :green[{approximate_distinct_count}]'
										)
									print(434)
									st.write(
										f'Approximate Unique (%): :green[{approximate_unique_percent:.2f}%]'
										)
									print(435)
									st.write(f'Missing: :green[{missing}]')
									print(436)
									st.write(
										f'Missing (%): :green[{missing_percent:.2f}%]'
										)
									print(437)
									st.write(f'Memory Size: :green[{memory_size}]')
									print(438)
								with col8:
									st.write('## Mode')
									print(439)
									mode = modified_df[selected_column].mode(
										).iloc[0]
									st.write(f'Mode: :green[{mode}]')
									print(440)
								with col9:
									st.write('## First 5 Sample Rows')
									print(441)
									st.write(modified_df[selected_column].head())
									print(442)
							with tab2:
								data = limit_unique_values(modified_df[
									selected_column]).value_counts().reset_index()
								data.columns = [selected_column, 'count']
								fig = px.pie(data, values='count', names=
									selected_column, hole=0.5)
								fig.update_traces(textposition='inside',
									textinfo='percent+label')
								print(443)
								fig.update_layout(legend=dict(orientation='h',
									yanchor='bottom', y=1.02, xanchor='right', x=1)
									)
								print(444)
								st.plotly_chart(fig)
								print(445)
					elif pd.api.types.is_numeric_dtype(modified_df[selected_column]
						):
						col3, col4 = st.columns(2)
						with col3:
							st.write(
								f'  - Data Type: :green[{modified_df[selected_column].dtype}]'
								)
							print(446)
							st.write(
								f'  - Mean: :green[{modified_df[selected_column].mean()}]'
								)
							print(447)
							st.write(
								f'  - Standard Deviation: :green[{modified_df[selected_column].std():.3}]'
								)
							print(448)
							st.write(
								f'  - Min Value: :green[{modified_df[selected_column].min()}]'
								)
							print(449)
							st.write(
								f'  - Max Value: :green[{modified_df[selected_column].max()}]'
								)
							print(450)
						with col4:
							plt.figure(figsize=(10, 6))
							print(451)
							sns.histplot(modified_df[selected_column], kde=True,
								color='green')
							print(452)
							st.pyplot()
							print(453)
							plt.close()
							print(454)
						with st.expander('More Info'):
							tab1, tab2, tab3 = st.tabs(['Insights', 'Box plot',
								'QQ plot'])
							with tab1:
								col4, col5, col6 = st.columns(3)
								with col4:
									st.write('#### Basic Statistics')
									print(455)
									insights = calculate_insights(modified_df[
										selected_column])
									basic_stats = {key: value for key, value in
										insights.items() if key in ['Mean',
										'Median', 'Mode', 'Standard deviation',
										'Variance', 'Kurtosis', 'Skewness']}
									for key, value in basic_stats.items():
										st.write(f'**{key}:** :green[{value:.3f}]')
										print(456)
									st.write(
										f"**Memory size:** :green[{insights.get('Memory size', 'N/A'):.3f}]"
										)
									print(457)
									st.write(
										f"**Range:** :green[{insights.get('Range', 'N/A'):.3f}]"
										)
									print(458)
									st.write(
										f"**Interquartile range:** :green[{insights.get('Interquartile range', 'N/A'):.3f}]"
										)
									print(459)
								with col5:
									st.write('#### Percentiles')
									print(460)
									descriptive_stats = insights.get(
										'Descriptive statistics')
									if descriptive_stats is not None:
										percentiles = descriptive_stats.loc[[
											'min', '25%', '50%', '75%', 'max']]
										if '5%' in descriptive_stats.index:
											percentiles['5%'] = descriptive_stats['5%']
										if '95%' in descriptive_stats.index:
											percentiles['95%'] = descriptive_stats[
												'95%']
										st.write(percentiles)
										print(461)
								with col6:
									st.write('#### Additional Statistics')
									print(462)
									additional_stats = {key: value for key,
										value in insights.items() if key in [
										'Distinct', 'Distinct (%)', 'Missing',
										'Missing (%)', 'Zeros', 'Zeros (%)',
										'Negative', 'Negative (%)']}
									for key, value in additional_stats.items():
										st.write(f'**{key}:** :green[{value:.3f}]')
										print(463)
									st.write(
										f"**Coefficient of variation (CV):** :green[{insights.get('Coefficient of variation (CV)', 'N/A'):.3f}]"
										)
									print(464)
									st.write(
										f"**Median Absolute Deviation (MAD):** :green[{insights.get('Median Absolute Deviation (MAD)', 'N/A'):.3f}]"
										)
									print(465)
									st.write(
										f"**Sum:** :green[{insights.get('Sum', 'N/A'):.3f}]"
										)
									print(466)
							with tab2:
								fig = px.box(modified_df, y=selected_column)
								st.plotly_chart(fig)
								print(467)
							with tab3:
								plt.figure(figsize=(10, 6))
								print(468)
								qqplot_data = sm.qqplot(modified_df[
									selected_column], line='s').gca().lines
								fig = go.Figure()
								fig.add_trace({'type': 'scatter', 'x':
									qqplot_data[0].get_xdata(), 'y':
									qqplot_data[0].get_ydata(), 'mode':
									'markers', 'marker': {'color': '#19d3f3'}})
								print(469)
								fig.add_trace({'type': 'scatter', 'x':
									qqplot_data[1].get_xdata(), 'y':
									qqplot_data[1].get_ydata(), 'mode': 'lines',
									'line': {'color': '#636efa'}})
								print(470)
								x_min = min(qqplot_data[0].get_xdata())
								x_max = max(qqplot_data[0].get_xdata())
								y_min = min(qqplot_data[0].get_ydata())
								y_max = max(qqplot_data[0].get_ydata())
								fig.add_trace(go.Scatter(x=[x_min, x_max], y=[
									y_min, y_max], mode='lines', line=dict(
									color='red', width=2), name='Identity Line'))
								print(471)
								fig.update_layout({'title':
									f'QQ Plot for {selected_column}', 'xaxis':
									{'title': 'Theoretical Quantiles',
									'zeroline': False}, 'yaxis': {'title':
									'Sample Quantiles'}, 'showlegend': False,
									'width': 800, 'height': 700})
								print(472)
								st.plotly_chart(fig)
								print(473)
						
						print(474)
					else:
						st.write('DataFrame not found.')
						print(475)
					print(476)
				except ZeroDivisionError:
					pass
				except Exception as e:
					st.error(e)
					print(477)
					st.subheader('⚠️Please upload a file⚠️')
					print(478)
					pass
				confirm_change = st.button('Confirm Change')
				if confirm_change:
					if 'tran' in st.session_state:
						tran_no = st.session_state.tran
					pipeline7 = {
						"method": f'{transformation_method}',
						"column_name": f'{selected_column}',
					}

					# Save the pipeline to a serialized object
					# try:
					# if type(pipe_) == 'dict':
					
					with open(f"{tran_no}_apply_transformation_on_col.pkl", "wb") as f:
						joblib.dump(pipeline7, f)
					# else:
					# joblib.dump(pipe_, f"{tran_no}_apply_transformation_on_col.pkl")
					st.session_state.df_pre = modified_df
					if tran_no == 0:
						st.session_state.pre_act.append(preprocessing_action)
					tran_no += 1
					st.session_state.full_flow.append(f"{preprocessing_action}->`{selected_column}`")
					st.session_state.tran = tran_no

					st.rerun()


		elif preprocessing_action == 'Dummy Variables :green[(Full DF)]':
			# encoding_method = st.selectbox('Select encoding method', [
			# 	'One-Hot Encoding', 'Label Encoding'])
			st.markdown(
				"""
				<style>
				.green-text {
					color: green;
				}
				</style>
				""",
				unsafe_allow_html=True
			)

			# Using st.markdown with custom CSS class
			st.markdown('- <h2 class="green-text">One-Hot Encoding:</h2>', unsafe_allow_html=True)
			opt = st.toggle('drop_first')
			if st.button('Apply', on_click=callback
				) or st.session_state.button_clicked:
				
				modified_df = df.copy()
				# modified_df = create_dummy_variables(modified_df, encoding_method, opt)
				if opt == True:
					opt = 'first'
				elif opt == False:
					opt = None
				
				y_var = st.session_state.y_var
				modified_df, encoder_pipeline = create_dummy_variables_for_sup(modified_df, y_var, drop=opt)
				# st.write('Modified DataFrame:')                                                   # PROBLEM
				print(480)
				st.write(modified_df)
				print(481)
				
				confirm_change = st.button('Confirm Change')
				if confirm_change:
					# pipeline8 = {
					# 	"method": f'{encoding_method}',
					# 	"param": opt
					# }

					# Save the pipeline to a serialized object
					# with open("apply_encoding_on_df.pkl", "wb") as f:
					st.session_state.pre_act.append(preprocessing_action)
					joblib.dump(encoder_pipeline, "apply_encoding_on_df.pkl")
					st.session_state.full_flow.append(f"{preprocessing_action}")
					st.session_state.df_pre = modified_df
					st.rerun()
					print(537)

		elif preprocessing_action == 'Dummy Variable :orange[(For Col)]':
			encoding_method = st.selectbox('Select encoding method', [
				'One-Hot Encoding', 'Label Encoding'])
			opt = None
			if st.button('Apply', on_click=callback
				) or st.session_state.button_clicked:
				modified_df = df.copy()
							# def create_dummy_variables_for_col(df, column_name, drop='first'):
				# modified_df = create_dummy_variables_for_col(modified_df, selected_column, encoding_method, opt)
				if encoding_method == 'One-Hot Encoding':
					encoding_method = 'onehot'
					opt = st.toggle('drop_first')
				elif encoding_method == 'Label Encoding':
					encoding_method = 'label'
				if opt == True:
					opt = 'first'
				elif opt == False:
					opt = None
				y_var = st.session_state.y_var
				modified_df, encoder_col_pipeline = create_dummy_variables_for_col_sup(df, selected_column, y_var, encoding_method, drop=opt)
				st.warning(f"Encoding is applied to the {selected_column}, but the changes will not be saved until you press the 'Confirm Changes' button below.")
				# if encoding_method == 'label':
				# 	modified_df, encoder_pipeline, le = create_dummy_variables_for_col(df, selected_column, encoding_method, drop=opt)
				# else:
				# 	modified_df, encoder_pipeline = create_dummy_variables_for_col(df, selected_column, encoding_method, drop=opt)

					
				st.write('Modified DataFrame:')
				print(538)
				st.write(modified_df)
				
				confirm_change = st.button('Confirm Change')
				if confirm_change:
					if 'col_dum' in st.session_state:
						col_dum_no = st.session_state.col_dum
					pipeline9 = {
						"method": encoder_col_pipeline,
						"column_name": f'{selected_column}',
						"param": opt
					}

					

					# Save the pipeline to a serialized object
					# with open(f"{col_dum_no}_apply_encoding_on_col.pkl", "wb") as f:
					joblib.dump(pipeline9, f"{col_dum_no}_apply_encoding_on_col.pkl")
					st.session_state.full_flow.append(f"{preprocessing_action}->`{selected_column}`")
					st.session_state.df_pre = modified_df
					if col_dum_no == 0:
						st.session_state.pre_act.append(preprocessing_action)
					col_dum_no += 1
					st.session_state.col_dum = col_dum_no
					st.rerun()
					print(595)


		elif preprocessing_action == 'Apply Scaling :green[(Full DF)]':
			scaling_method = st.selectbox('Select scaling method', [
				'Standardize Scaling', 'Min-Max Scaling', 'Robust Scaling'], index=1)
			if st.button('Apply', on_click=callback
				) or st.session_state.button_clicked:
				modified_df = df.copy()
				y_var = st.session_state.y_var
				modified_df = apply_scaling_for_sup(df, y_var, scaling_method)
				st.warning(f"{scaling_method} is applied on full df except y variable, but changes will not be saved until the 'Confirm Changes' button is pressed.")
				st.write('Modified DataFrame:')
				print(596)
				st.write(modified_df)
				print(597)
				confirm_change = st.button('Confirm Change')
				if confirm_change:
					pipeline10 = {
						"method": f'{scaling_method}',
					}

					# Save the pipeline to a serialized object
					st.session_state.pre_act.append(preprocessing_action)
					with open("apply_scaling_on_df.pkl", "wb") as f:
						joblib.dump(pipeline10, f)
					st.session_state.df_pre = modified_df
					st.session_state.full_flow.append(f"{preprocessing_action}")
					# modified_df.to_csv("clean.csv", index=False)
					st.rerun()
					print(598)
		elif preprocessing_action == 'Split Data':
			st.switch_page(r'pages/6_Split_Data.py')
		elif preprocessing_action == 'Discretize Variable :orange[(For Col)]':
			bins = st.slider('Select the number of bins', min_value=2,
				max_value=20, value=5)
			strategy = st.selectbox('Select the strategy for binning', [
				'uniform', 'quantile', 'kmeans'])
			if st.button('Apply', on_click=callback
				) or st.session_state.button_clicked:
				modified_df = df.copy()
				modified_df, discretizer_pipeline = discretize_output(modified_df, selected_column,
					bins, strategy)
				# st.write('Modified DataFrame:')
				print(599)
				st.write(modified_df)
				print(600)
				try:
					if modified_df[selected_column].dtype == 'object':
						col1, col2 = st.columns(2)
						with col1:
							unique_values = modified_df[selected_column].nunique()
							st.write(
								f'  - Data Type: :green[{modified_df[selected_column].dtype}]'
								)
							print(601)
							st.write(
								f'  - Number of Unique Values: :green[{unique_values}]'
								)
							print(602)
							if unique_values <= 20:
								st.write(
									f"  - Unique Values: :green[{', '.join(map(str, modified_df[selected_column].unique()))}]"
									)
								print(603)
							else:
								st.write(f'  - Top 20 Unique Values:')
								print(604)
								st.write(
									f":green[{', '.join(map(str, modified_df[selected_column].value_counts().head(20).index))}]"
									)
								print(605)
						with col2:
							plt.figure(figsize=(10, 6))
							print(606)
							try:
								sns.countplot(x=limit_unique_values(modified_df
									[selected_column]), data=modified_df, color
									='green')
								print(607)
							except:
								sns.countplot(x=modified_df[selected_column],
									data=modified_df, color='green')
								print(608)
							plt.xticks(rotation=45)
							print(609)
							st.pyplot()
							print(610)
							plt.close()
							print(611)
						with st.expander('More Info'):
							tab1, tab2 = st.tabs(['Insights', 'Donut chart'])
							with tab1:
								col7, col8, col9 = st.columns(3)
								with col7:
									st.write('## Insights')
									print(612)
									approximate_distinct_count = modified_df[
										selected_column].nunique()
									approximate_unique_percent = (
										approximate_distinct_count / len(
										modified_df) * 100)
									missing = modified_df[selected_column].isna(
										).sum()
									missing_percent = missing / len(modified_df
										) * 100
									memory_size = modified_df[selected_column
										].memory_usage(deep=True)
									st.write(
										f'Approximate Distinct Count: :green[{approximate_distinct_count}]'
										)
									print(613)
									st.write(
										f'Approximate Unique (%): :green[{approximate_unique_percent:.2f}%]'
										)
									print(614)
									st.write(f'Missing: :green[{missing}]')
									print(615)
									st.write(
										f'Missing (%): :green[{missing_percent:.2f}%]'
										)
									print(616)
									st.write(f'Memory Size: :green[{memory_size}]')
									print(617)
								with col8:
									st.write('## Mode')
									print(618)
									mode = modified_df[selected_column].mode(
										).iloc[0]
									st.write(f'Mode: :green[{mode}]')
									print(619)
								with col9:
									st.write('## First 5 Sample Rows')
									print(620)
									st.write(modified_df[selected_column].head())
									print(621)
							with tab2:
								data = limit_unique_values(modified_df[
									selected_column]).value_counts().reset_index()
								data.columns = [selected_column, 'count']
								fig = px.pie(data, values='count', names=
									selected_column, hole=0.5)
								fig.update_traces(textposition='inside',
									textinfo='percent+label')
								print(622)
								fig.update_layout(legend=dict(orientation='h',
									yanchor='bottom', y=1.02, xanchor='right', x=1)
									)
								print(623)
								st.plotly_chart(fig)
								print(624)
					elif pd.api.types.is_numeric_dtype(modified_df[selected_column]
						):
						col3, col4 = st.columns(2)
						with col3:
							st.write(
								f'  - Data Type: :green[{modified_df[selected_column].dtype}]'
								)
							print(625)
							st.write(
								f'  - Mean: :green[{modified_df[selected_column].mean()}]'
								)
							print(626)
							st.write(
								f'  - Standard Deviation: :green[{modified_df[selected_column].std():.3}]'
								)
							print(627)
							st.write(
								f'  - Min Value: :green[{modified_df[selected_column].min()}]'
								)
							print(628)
							st.write(
								f'  - Max Value: :green[{modified_df[selected_column].max()}]'
								)
							print(629)
						with col4:
							plt.figure(figsize=(10, 6))
							print(630)
							sns.histplot(modified_df[selected_column], kde=True,
								color='green')
							print(631)
							st.pyplot()
							print(632)
							plt.close()
							print(633)
						with st.expander('More Info'):
							tab1, tab2, tab3 = st.tabs(['Insights', 'Box plot',
								'QQ plot'])
							with tab1:
								col4, col5, col6 = st.columns(3)
								with col4:
									st.write('#### Basic Statistics')
									print(634)
									insights = calculate_insights(modified_df[
										selected_column])
									basic_stats = {key: value for key, value in
										insights.items() if key in ['Mean',
										'Median', 'Mode', 'Standard deviation',
										'Variance', 'Kurtosis', 'Skewness']}
									for key, value in basic_stats.items():
										st.write(f'**{key}:** :green[{value:.3f}]')
										print(635)
									st.write(
										f"**Memory size:** :green[{insights.get('Memory size', 'N/A'):.3f}]"
										)
									print(636)
									st.write(
										f"**Range:** :green[{insights.get('Range', 'N/A'):.3f}]"
										)
									print(637)
									st.write(
										f"**Interquartile range:** :green[{insights.get('Interquartile range', 'N/A'):.3f}]"
										)
									print(638)
								with col5:
									st.write('#### Percentiles')
									print(639)
									descriptive_stats = insights.get(
										'Descriptive statistics')
									if descriptive_stats is not None:
										percentiles = descriptive_stats.loc[[
											'min', '25%', '50%', '75%', 'max']]
										if '5%' in descriptive_stats.index:
											percentiles['5%'] = descriptive_stats['5%']
										if '95%' in descriptive_stats.index:
											percentiles['95%'] = descriptive_stats[
												'95%']
										st.write(percentiles)
										print(640)
								with col6:
									st.write('#### Additional Statistics')
									print(641)
									additional_stats = {key: value for key,
										value in insights.items() if key in [
										'Distinct', 'Distinct (%)', 'Missing',
										'Missing (%)', 'Zeros', 'Zeros (%)',
										'Negative', 'Negative (%)']}
									for key, value in additional_stats.items():
										st.write(f'**{key}:** :green[{value:.3f}]')
										print(642)
									st.write(
										f"**Coefficient of variation (CV):** :green[{insights.get('Coefficient of variation (CV)', 'N/A'):.3f}]"
										)
									print(643)
									st.write(
										f"**Median Absolute Deviation (MAD):** :green[{insights.get('Median Absolute Deviation (MAD)', 'N/A'):.3f}]"
										)
									print(644)
									st.write(
										f"**Sum:** :green[{insights.get('Sum', 'N/A'):.3f}]"
										)
									print(645)
							with tab2:
								fig = px.box(modified_df, y=selected_column)
								st.plotly_chart(fig)
								print(646)
							with tab3:
								plt.figure(figsize=(10, 6))
								print(647)
								qqplot_data = sm.qqplot(modified_df[
									selected_column], line='s').gca().lines
								fig = go.Figure()
								fig.add_trace({'type': 'scatter', 'x':
									qqplot_data[0].get_xdata(), 'y':
									qqplot_data[0].get_ydata(), 'mode':
									'markers', 'marker': {'color': '#19d3f3'}})
								print(648)
								fig.add_trace({'type': 'scatter', 'x':
									qqplot_data[1].get_xdata(), 'y':
									qqplot_data[1].get_ydata(), 'mode': 'lines',
									'line': {'color': '#636efa'}})
								print(649)
								x_min = min(qqplot_data[0].get_xdata())
								x_max = max(qqplot_data[0].get_xdata())
								y_min = min(qqplot_data[0].get_ydata())
								y_max = max(qqplot_data[0].get_ydata())
								fig.add_trace(go.Scatter(x=[x_min, x_max], y=[
									y_min, y_max], mode='lines', line=dict(
									color='red', width=2), name='Identity Line'))
								print(650)
								fig.update_layout({'title':
									f'QQ Plot for {selected_column}', 'xaxis':
									{'title': 'Theoretical Quantiles',
									'zeroline': False}, 'yaxis': {'title':
									'Sample Quantiles'}, 'showlegend': False,
									'width': 800, 'height': 700})
								print(651)
								st.plotly_chart(fig)
								print(652)
						
						print(653)
					else:
						st.write('DataFrame not found.')
						print(654)
					print(655)
				except ZeroDivisionError:
					pass
				except Exception as e:
					st.error(e)
					print(656)
					st.subheader('⚠️Please upload a file⚠️')
					print(657)
					pass
				confirm_change = st.button('Confirm Change')
				if confirm_change:
					if 'des_var' in st.session_state:
						des_var_no = st.session_state.des_var
					pipeline11 = {
						"column_name": f'{selected_column}',
						"bins": bins,
						"strategy": discretizer_pipeline
					}
					# with open(f"{des_var_no}_discretize_output_col.pkl", "wb") as f:
					# 	joblib.dump(pipeline11, f)
					# joblib.dump(discretizer_pipeline, f"{des_var_no}_discretize_output_col.pkl")
					
					joblib.dump(pipeline11, f"{des_var_no}_discretize_output_col.pkl")
					st.session_state.df_pre = modified_df
					if des_var_no == 0:
						st.session_state.pre_act.append(preprocessing_action)
					des_var_no += 1
					st.session_state.full_flow.append(f"{preprocessing_action}->`{selected_column}`")
					st.session_state.des_var = des_var_no
					st.rerun()
					print(658)

		elif preprocessing_action == 'Column Unique Value Replacement :orange[(For Col)]':
			print(43)
			print(659)
			select = st.selectbox('Convert type:', ['Convert to Str:',
				'Convert to int:', 'Convert to float:'])
			print(44)
			print(660)
			distinct_values = df[selected_column].unique()
			print(45)
			print(661)
			replacements = {}
			for value in distinct_values:
				print(46)
				print(662)
				if select == 'Convert to Str:':
					print(47)
					print(663)
					replacements[value] = st.text_input(f"Replace '{value}' with:", value=str(value))
				elif select == 'Convert to int:':
					try:
						print(48)
						print(664)
						replacements[value] = st.number_input(
							f"Replace '{value}' with:", value=int(value), step=1)
					except:
						print(49)
						print(665)
						replacements[value] = st.number_input(
							f"Replace '{value}' with:", step=1)
				elif select == 'Convert to float:':
					try:
						print(50)
						print(666)
						replacements[value] = st.number_input(f"Replace '{value}' with:", value=float(value),step=0.01)
					except:
						print(51)
						print(667)
						replacements[value] = st.number_input(f"Replace '{value}' with:", step=0.01)
			if st.button('Apply', on_click=callback
				) or st.session_state.button_clicked:
				st.warning(
					f'"{select}" is applied, but changes will not be saved until you press the \'Confirm Changes\' button below.'
					)
				print(668)
				print(52)
				print(669)
				modified_df = df.copy()
				print(53)
				print(670)
				print("replacements:=_", replacements)
				modified_df[selected_column].replace(replacements, inplace=True)
				print(671)
				print(54)
				print(672)
				st.write('Modified DataFrame:')
				print(673)
				print(55)
				print(674)
				st.write(modified_df)
				print(675)
				print(56)
				print(676)
				if select == 'Convert to int:':
					print(57)
					print(677)
					modified_df[selected_column] = modified_df[selected_column].astype(pd.Int64Dtype())
					print(58)
					print(678)
				elif select == 'Convert to float:':
					print(59)
					print(679)
					modified_df[selected_column] = modified_df[selected_column
						].astype(float)
				try:
					print(60)
					print(680)
					if modified_df[selected_column].dtype == 'object':
						print(61)
						print(681)
						col1, col2 = st.columns(2)
						with col1:
							print(62)
							print(682)
							unique_values = modified_df[selected_column].nunique()
							print(63)
							print(683)
							st.write(
								f'  - Data Type: :green[{modified_df[selected_column].dtype}]'
								)
							print(684)
							print(64)
							print(685)
							st.write(
								f'  - Number of Unique Values: :green[{unique_values}]'
								)
							print(686)
							print(65)
							print(687)
							if unique_values <= 20:
								print(66)
								print(688)
								st.write(
									f"  - Unique Values: :green[{', '.join(map(str, modified_df[selected_column].unique()))}]"
									)
								print(689)
								print(67)
								print(690)
							else:
								print(68)
								print(691)
								st.write(f'  - Top 20 Unique Values:')
								print(692)
								print(69)
								print(693)
								st.write(
									f":green[{', '.join(map(str, modified_df[selected_column].value_counts().head(20).index))}]"
									)
								print(694)
								print(70)
								print(695)
						with col2:
							print(71)
							print(696)
							plt.figure(figsize=(10, 6))
							print(697)
							try:
								print(72)
								print(698)
								sns.countplot(x=limit_unique_values(modified_df
									[selected_column]), data=modified_df, color
									='green')
								print(699)
							except:
								print(73)
								print(700)
								sns.countplot(x=modified_df[selected_column],
									data=modified_df, color='green')
								print(701)
							plt.xticks(rotation=45)
							print(702)
							st.pyplot()
							print(703)
							plt.close()
							print(704)
							print(74)
							print(705)
					elif pd.api.types.is_numeric_dtype(modified_df[selected_column]
						):
						print(83)
						print(706)
						col3, col4 = st.columns(2)
						with col3:
							print(84)
							print(707)
							st.write(
								f'  - Data Type: :green[{modified_df[selected_column].dtype}]'
								)
							print(708)
							st.write(
								f'  - Mean: :green[{modified_df[selected_column].mean()}]'
								)
							print(709)
							st.write(
								f'  - Standard Deviation: :green[{modified_df[selected_column].std():.3}]'
								)
							print(710)
							st.write(
								f'  - Min Value: :green[{modified_df[selected_column].min()}]'
								)
							print(711)
							st.write(
								f'  - Max Value: :green[{modified_df[selected_column].max()}]'
								)
							print(712)
						with col4:
							print(85)
							print(713)
							plt.figure(figsize=(10, 6))
							print(714)
							sns.histplot(modified_df[selected_column], kde=True,
								color='green')
							print(715)
							st.pyplot()
							print(716)
							plt.close()
							print(717)
						
						print(718)
						print(106)
						print(719)
					else:
						print(107)
						print(720)
						st.write('DataFrame not found.')
						print(721)
						print(108)
						print(722)
					print(723)
					print(109)
					print(724)
				except ZeroDivisionError:
					print(110)
					print(725)
					pass
				except Exception as e:
					print(111)
					print(726)
					st.error(e)
					print(727)
					st.subheader('⚠️Please upload a file⚠️')
					print(728)
					pass
				confirm_change = st.button('Confirm Change')
				if confirm_change:
					if 'out_rplac' in st.session_state:
						out_rplac_no = st.session_state.out_rplac
					pipeline12 = {
						"column_name": f'{selected_column}',
						"replacements": replacements,
						"select": select
					}

					# Save the pipeline to a serialized object
					
					with open(f"{out_rplac_no}_column_unique_value_replacement.pkl", "wb") as f:
						joblib.dump(pipeline12, f)
					st.session_state.df_pre = modified_df
					if out_rplac_no == 0:
						st.session_state.pre_act.append(preprocessing_action)
					out_rplac_no += 1
					st.session_state.full_flow.append(f"{preprocessing_action}->`{selected_column}`")
					st.session_state.out_rplac = out_rplac_no
					st.rerun()
					print(729)


	except Exception as e:
		print('e:=_0', e)
		print(730)
		st.error(e)
		print(731)