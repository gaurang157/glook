import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
st.set_page_config(
		page_title="ML-Automation", 
		page_icon="🏗️", 
		layout="wide", 
		initial_sidebar_state = "expanded")



# Initialize state
if "button_clicked" not in st.session_state:
	st.session_state.button_clicked = False
def callback():
	# Button was clicked!
	st.session_state.button_clicked = True
	
def split_data(df, target_column, test_size, random_state):
    try:
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Splitting into initial train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        # Further split the training set into train-validation split
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=random_state)
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    except Exception as e:
        # Handle exception
        print(e)

# st.write(modified_df)


try:
	# st.write("Session State:->", st.session_state["shared"])
	# Streamlit UI for data splitting
	st.title("Data Splitting Page")

	# Display the modified DataFrame
	st.subheader("Modified DataFrame")
	if "df_pre" in st.session_state:
		# df = st.session_state.df
		df0 = st.session_state.df0
		oragnial_df = st.session_state.df
		df_to_pre = st.session_state.df_pre
		# st.warning("kl")
		# to_select = st.selectbox("Select Data Frame (Recommended: Pre-Processed DF)", ["oragnial_df", "pre_processed_df"], index=1)
		# if to_select == "oragnial_df":
		# 	df = oragnial_df
		# elif to_select == "pre_processed_df":
		df = df_to_pre
		# Assuming df is your DataFrame
		# df.to_csv('your_file.csv', index=False)
		# Data split options
		st.write(df)
		
		try:
			y_var = st.session_state.y_var
			ind = df.columns.get_loc(y_var)
			target_column = st.sidebar.selectbox("Select the target column:", df.columns, index=ind)
		except:
			st.sidebar.info("Pre-Processing is not Done.")
			target_column = st.sidebar.selectbox("Select the target column:", df.columns)
			st.session_state.y_var = str(target_column)
		test_size = st.sidebar.slider("Select the test size:", 0.1, 0.5, step=0.05)
		random_state = st.sidebar.number_input("Enter the random state:", min_value=0, max_value=10000, value=42)
		st.sidebar.warning("Note: Split Data Before Moving to Model Building Page ===>")
	else:
		pass
	if (
		st.button("Apply", on_click=callback)
			or st.session_state.button_clicked
			):
		try:
			# X_train, X_test, y_train, y_test = split_data(df, target_column, test_size, random_state)
			X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, target_column, test_size, random_state)
			X_train1, X_val1, X_test1, y_train1, y_val1, y_test1 = split_data(oragnial_df, target_column, test_size, random_state)
			# try:
			# X_train0, X_val0, X_test0, y_train0, y_val0, y_test0 = split_data(df0, target_column, test_size, random_state)
			# except:
			# 	try:
			# 		cat_columns = df0.select_dtypes(include=['object']).columns
			# 		encoded_df = pd.get_dummies(df0, columns=cat_columns, drop_first=True)
			# 		X_train0, X_test0, y_train0, y_test0 = split_data(encoded_df, target_column, test_size, random_state)
			# 	except:
			# 		cat_columns = df0.select_dtypes(include=['object']).columns
			# 		encoded_df1 = pd.get_dummies(df0, columns=cat_columns, drop_first=False)
			# 		X_train0, X_test0, y_train0, y_test0 = split_data(encoded_df1, target_column, test_size, random_state)
					

			# st.success("Data split successfully.")
			# st.success("Data split successfully.")
			st.warning("Changes are applied, but they will not be saved until the 'Confirm Changes' button below is pressed.")
			st.write(":green[Training Data:]")
			col1, col2 = st.columns([4,1])
			with col1:
				st.write("X_train")
				st.write(X_train)
			with col2:
				st.write("y_train")
				st.write(y_train)

			st.write(":green[Testing Data:]")
			col3, col4 = st.columns([4,1])
			with col3:
				st.write("X_test")
				st.write(X_test)

			with col4:
				st.write("y_test")
				st.write(y_test)

			st.write(":green[Validation Data:]")
			st.write(X_val)
			# X_test0 = X_test0.reset_index(drop=True)
			# dota = pd.concat([X_test0, y_test0], axis=1)
			# dota = dota.reset_index(drop=True)
			# dota.to_csv("Masked AviationSocial Media Data.csv")
			# st.write(dota)
			# Store the train and test sets in session state
			# st.session_state.X_train = X_train
			# st.session_state.X_test = X_test
			# st.session_state.y_train = y_train
			# st.session_state.y_test = y_test
			# st.write(y_test.name)
			# st.write(X_val.describe())
			y_name = y_test.name
		except Exception as e:
			print(e)
			st.error(f"Error occurred during data split: {e}")
except:
	pass
# Undo functionality
confirm_change = st.button(
	'Confirm Change and move to Model Building Page', 
	use_container_width=True
	)
gg = st.sidebar.button("Confirm Change and Swith to Model Building")

if gg:
	st.session_state.X_train = X_train
	st.session_state.X_test = X_test
	st.session_state.y_train = y_train
	st.session_state.y_test = y_test
	st.session_state.X_test0 = X_val
	st.session_state.y_name = y_name
	st.session_state.X_val1 = X_val1
	st.session_state.y_var = target_column
	st.switch_page("pages/8_Supervised_Learning.py")

if confirm_change:
	# st.session_state.df = modified_df
	st.session_state.X_train = X_train
	st.session_state.X_test = X_test
	st.session_state.y_train = y_train
	st.session_state.y_test = y_test
	st.session_state.X_test0 = X_val
	st.session_state.y_name = y_name
	st.session_state.X_val1 = X_val1
	st.session_state.y_var = target_column
	# st.switch_page("pages/7_Model_Building.py")
	st.switch_page("pages/8_Supervised_Learning.py")
	