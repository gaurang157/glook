import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
	GradientBoostingClassifier,
	  BaggingClassifier,
		AdaBoostClassifier,
		  RandomForestClassifier,
			RandomForestRegressor,
			  GradientBoostingRegressor
)
from xgboost import XGBClassifier , XGBRegressor
# from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error
# Define a function to plot the evaluation metrics
def plot_evaluation_metrics(train_metrics, test_metrics, test_pred_y, train_pred_y, y_test):
	# Set style to dark background
	# plt.style.use('dark_background')
	# plt.style.use('light_background')

	# Plot bar plot for accuracy, precision, recall, f1-score
	metrics_names = ['accuracy', 'roc_auc', 'f1_score', 'recall', 'precision', 'rmse', 'r2_score']
	train_values = [train_metrics[name] for name in metrics_names]
	test_values = [test_metrics[name] for name in metrics_names]

	plt.figure(figsize=(10, 6))
	plt.bar(metrics_names, train_values, color='blue', alpha=0.5, label='Train')
	plt.bar(metrics_names, test_values, color='yellow', alpha=0.5, label='Test')
	plt.xlabel('Metrics')
	plt.ylabel('Value')
	plt.title('Evaluation Metrics')
	plt.legend()
	st.pyplot()

	# Plot ROC curve
	fpr, tpr, thresholds = roc_curve(y_test, test_pred_y)
	plt.figure(figsize=(8, 6))
	plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = {:.2f})'.format(test_metrics['roc_auc']))
	plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver Operating Characteristic (ROC) Curve')
	plt.legend(loc="lower right")
	st.pyplot()

	# Plot Precision-Recall curve
	precision, recall, _ = precision_recall_curve(y_test, test_pred_y)
	plt.figure(figsize=(8, 6))
	plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve')
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.title('Precision-Recall Curve')
	plt.legend(loc="lower left")
	st.pyplot()

	# Plot Confusion Matrix
	cm = confusion_matrix(y_test, test_pred_y)
	plt.figure(figsize=(8, 6))
	sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', cbar=False)
	plt.xlabel('Predicted labels')
	plt.ylabel('True labels')
	plt.title('Confusion Matrix')
	st.pyplot()






# # Create the base classifier
# dc = DecisionTreeClassifier()
# # Create the BaggingClassifier
# model = BaggingClassifier(base_estimator=dc, n_estimators=21)
from sklearn.metrics import (
	accuracy_score, f1_score, mean_squared_error, r2_score, roc_auc_score,
	recall_score, precision_score, roc_curve, auc, confusion_matrix,
	classification_report
)
# Function to train Linear Regression model
def train_bagging_classifier(train_X, train_y, test_X, test_y):
	# # Create the base classifier
	dc = DecisionTreeClassifier()
	# Create the BaggingClassifier
	bc = BaggingClassifier(base_estimator=dc, n_estimators=21)
	
	# Fit to training set
	bc.fit(train_X, train_y)
	
	# Predict on test set
	test_pred_y = bc.predict(test_X)
	
	# Predict on the training set
	train_pred_y = bc.predict(train_X)
	
	# Calculate evaluation metrics for test set
	test_metrics = {
		'accuracy': float("{:.3f}".format(accuracy_score(test_y, test_pred_y))),
		'roc_auc': float("{:.3f}".format(roc_auc_score(test_y, test_pred_y))),
		'f1_score': float("{:.3f}".format(f1_score(test_y, test_pred_y, average='weighted'))),
		'recall': float("{:.3f}".format(recall_score(test_y, test_pred_y, average='weighted'))),
		'precision': float("{:.3f}".format(precision_score(test_y, test_pred_y, average='weighted'))),
		'rmse': float("{:.3f}".format(np.sqrt(mean_squared_error(test_y, test_pred_y)))),
		'r2_score': float("{:.3f}".format(r2_score(test_y, test_pred_y)))
	}
	
	# Calculate evaluation metrics for training set
	train_metrics = {
		'accuracy': float("{:.3f}".format(accuracy_score(train_y, train_pred_y))),
		'roc_auc': float("{:.3f}".format(roc_auc_score(train_y, train_pred_y))),
		'f1_score': float("{:.3f}".format(f1_score(train_y, train_pred_y, average='weighted'))),
		'recall': float("{:.3f}".format(recall_score(train_y, train_pred_y, average='weighted'))),
		'precision': float("{:.3f}".format(precision_score(train_y, train_pred_y, average='weighted'))),
		'rmse': float("{:.3f}".format(np.sqrt(mean_squared_error(train_y, train_pred_y)))),
		'r2_score': float("{:.3f}".format(r2_score(train_y, train_pred_y)))
	}
	
	return bc, train_metrics, test_metrics, test_pred_y, train_pred_y


def train_gradient_boosting_classifier(train_X, train_y, test_X, test_y):
	# Initialize Gradient Boosting Classifier
	# , n_estimators=300, learning_rate=0.05, max_features=5, random_state=100
	# gbc = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_features=max_features, random_state=random_state)
	gbc = GradientBoostingClassifier()
	
	# Fit to training set
	gbc.fit(train_X, train_y)
	
	# Predict on test set
	test_pred_y = gbc.predict(test_X)
	
	# Predict on the training set
	train_pred_y = gbc.predict(train_X)
	
	# Calculate evaluation metrics for test set
	test_metrics = {
		'accuracy': float("{:.3f}".format(accuracy_score(test_y, test_pred_y))),
		'roc_auc': float("{:.3f}".format(roc_auc_score(test_y, test_pred_y))),
		'f1_score': float("{:.3f}".format(f1_score(test_y, test_pred_y, average='weighted'))),
		'recall': float("{:.3f}".format(recall_score(test_y, test_pred_y, average='weighted'))),
		'precision': float("{:.3f}".format(precision_score(test_y, test_pred_y, average='weighted'))),
		'rmse': float("{:.3f}".format(np.sqrt(mean_squared_error(test_y, test_pred_y)))),
		'r2_score': float("{:.3f}".format(r2_score(test_y, test_pred_y)))
	}
	
	# Calculate evaluation metrics for training set
	train_metrics = {
		'accuracy': float("{:.3f}".format(accuracy_score(train_y, train_pred_y))),
		'roc_auc': float("{:.3f}".format(roc_auc_score(train_y, train_pred_y))),
		'f1_score': float("{:.3f}".format(f1_score(train_y, train_pred_y, average='weighted'))),
		'recall': float("{:.3f}".format(recall_score(train_y, train_pred_y, average='weighted'))),
		'precision': float("{:.3f}".format(precision_score(train_y, train_pred_y, average='weighted'))),
		'rmse': float("{:.3f}".format(np.sqrt(mean_squared_error(train_y, train_pred_y)))),
		'r2_score': float("{:.3f}".format(r2_score(train_y, train_pred_y)))
	}
	
	return gbc, train_metrics, test_metrics, test_pred_y, train_pred_y


# def train_multinomial_naive_bayes(train_X, train_y, test_X, test_y):
# 	# # Create the base classifier
# 	nb = MultinomialNB(alpha=5)
	
# 	# Fit to training set
# 	nb.fit(train_X, train_y)
	
# 	# Predict on test set
# 	test_pred_y = nb.predict(test_X)
	
# 	# Predict on the training set
# 	train_pred_y = nb.predict(train_X)
	
# 	# Calculate evaluation metrics for test set
# 	test_metrics = {
# 		'accuracy': float("{:.3f}".format(accuracy_score(test_y, test_pred_y))),
# 		'roc_auc': float("{:.3f}".format(roc_auc_score(test_y, test_pred_y))),
# 		'f1_score': float("{:.3f}".format(f1_score(test_y, test_pred_y, average='weighted'))),
# 		'recall': float("{:.3f}".format(recall_score(test_y, test_pred_y, average='weighted'))),
# 		'precision': float("{:.3f}".format(precision_score(test_y, test_pred_y, average='weighted'))),
# 		'rmse': float("{:.3f}".format(np.sqrt(mean_squared_error(test_y, test_pred_y)))),
# 		'r2_score': float("{:.3f}".format(r2_score(test_y, test_pred_y)))
# 	}
	
# 	# Calculate evaluation metrics for training set
# 	train_metrics = {
# 		'accuracy': float("{:.3f}".format(accuracy_score(train_y, train_pred_y))),
# 		'roc_auc': float("{:.3f}".format(roc_auc_score(train_y, train_pred_y))),
# 		'f1_score': float("{:.3f}".format(f1_score(train_y, train_pred_y, average='weighted'))),
# 		'recall': float("{:.3f}".format(recall_score(train_y, train_pred_y, average='weighted'))),
# 		'precision': float("{:.3f}".format(precision_score(train_y, train_pred_y, average='weighted'))),
# 		'rmse': float("{:.3f}".format(np.sqrt(mean_squared_error(train_y, train_pred_y)))),
# 		'r2_score': float("{:.3f}".format(r2_score(train_y, train_pred_y)))
# 	}
	
# 	return nb, train_metrics, test_metrics, test_pred_y, train_pred_y


def train_logistic_regression(train_X, train_y, test_X, test_y):
	# # Create the base classifier
	LR = LogisticRegression()
	
	# Fit to training set
	LR.fit(train_X, train_y)
	
	# Predict on test set
	test_pred_y = LR.predict(test_X)
	
	# Predict on the training set
	train_pred_y = LR.predict(train_X)
	
	# Calculate evaluation metrics for test set
	test_metrics = {
		'accuracy': float("{:.3f}".format(accuracy_score(test_y, test_pred_y))),
		'roc_auc': float("{:.3f}".format(roc_auc_score(test_y, test_pred_y))),
		'f1_score': float("{:.3f}".format(f1_score(test_y, test_pred_y, average='weighted'))),
		'recall': float("{:.3f}".format(recall_score(test_y, test_pred_y, average='weighted'))),
		'precision': float("{:.3f}".format(precision_score(test_y, test_pred_y, average='weighted'))),
		'rmse': float("{:.3f}".format(np.sqrt(mean_squared_error(test_y, test_pred_y)))),
		'r2_score': float("{:.3f}".format(r2_score(test_y, test_pred_y)))
	}
	
	# Calculate evaluation metrics for training set
	train_metrics = {
		'accuracy': float("{:.3f}".format(accuracy_score(train_y, train_pred_y))),
		'roc_auc': float("{:.3f}".format(roc_auc_score(train_y, train_pred_y))),
		'f1_score': float("{:.3f}".format(f1_score(train_y, train_pred_y, average='weighted'))),
		'recall': float("{:.3f}".format(recall_score(train_y, train_pred_y, average='weighted'))),
		'precision': float("{:.3f}".format(precision_score(train_y, train_pred_y, average='weighted'))),
		'rmse': float("{:.3f}".format(np.sqrt(mean_squared_error(train_y, train_pred_y)))),
		'r2_score': float("{:.3f}".format(r2_score(train_y, train_pred_y)))
	}
	
	return LR, train_metrics, test_metrics, test_pred_y, train_pred_y


def train_ada_boost_classifier(train_X, train_y, test_X, test_y):
	# # Create the base classifier
	abc = AdaBoostClassifier()
	
	# Fit to training set
	abc.fit(train_X, train_y)
	
	# Predict on test set
	test_pred_y = abc.predict(test_X)
	
	# Predict on the training set
	train_pred_y = abc.predict(train_X)
	
	# Calculate evaluation metrics for test set
	test_metrics = {
		'accuracy': float("{:.3f}".format(accuracy_score(test_y, test_pred_y))),
		'roc_auc': float("{:.3f}".format(roc_auc_score(test_y, test_pred_y))),
		'f1_score': float("{:.3f}".format(f1_score(test_y, test_pred_y, average='weighted'))),
		'recall': float("{:.3f}".format(recall_score(test_y, test_pred_y, average='weighted'))),
		'precision': float("{:.3f}".format(precision_score(test_y, test_pred_y, average='weighted'))),
		'rmse': float("{:.3f}".format(np.sqrt(mean_squared_error(test_y, test_pred_y)))),
		'r2_score': float("{:.3f}".format(r2_score(test_y, test_pred_y)))
	}
	
	# Calculate evaluation metrics for training set
	train_metrics = {
		'accuracy': float("{:.3f}".format(accuracy_score(train_y, train_pred_y))),
		'roc_auc': float("{:.3f}".format(roc_auc_score(train_y, train_pred_y))),
		'f1_score': float("{:.3f}".format(f1_score(train_y, train_pred_y, average='weighted'))),
		'recall': float("{:.3f}".format(recall_score(train_y, train_pred_y, average='weighted'))),
		'precision': float("{:.3f}".format(precision_score(train_y, train_pred_y, average='weighted'))),
		'rmse': float("{:.3f}".format(np.sqrt(mean_squared_error(train_y, train_pred_y)))),
		'r2_score': float("{:.3f}".format(r2_score(train_y, train_pred_y)))
	}
	
	return abc, train_metrics, test_metrics, test_pred_y, train_pred_y


def train_random_forest_classifier(train_X, train_y, test_X, test_y):
	# # Create the base classifier
	# rfc = RandomForestClassifier(n_estimators= 10, criterion="entropy")
	rfc = RandomForestClassifier()	
	# Fit to training set
	rfc.fit(train_X, train_y)
	
	# Predict on test set
	test_pred_y = rfc.predict(test_X)
	
	# Predict on the training set
	train_pred_y = rfc.predict(train_X)
	
	# Calculate evaluation metrics for test set
	test_metrics = {
		'accuracy': float("{:.3f}".format(accuracy_score(test_y, test_pred_y))),
		'roc_auc': float("{:.3f}".format(roc_auc_score(test_y, test_pred_y))),
		'f1_score': float("{:.3f}".format(f1_score(test_y, test_pred_y, average='weighted'))),
		'recall': float("{:.3f}".format(recall_score(test_y, test_pred_y, average='weighted'))),
		'precision': float("{:.3f}".format(precision_score(test_y, test_pred_y, average='weighted'))),
		'rmse': float("{:.3f}".format(np.sqrt(mean_squared_error(test_y, test_pred_y)))),
		'r2_score': float("{:.3f}".format(r2_score(test_y, test_pred_y)))
	}
	
	# Calculate evaluation metrics for training set
	train_metrics = {
		'accuracy': float("{:.3f}".format(accuracy_score(train_y, train_pred_y))),
		'roc_auc': float("{:.3f}".format(roc_auc_score(train_y, train_pred_y))),
		'f1_score': float("{:.3f}".format(f1_score(train_y, train_pred_y, average='weighted'))),
		'recall': float("{:.3f}".format(recall_score(train_y, train_pred_y, average='weighted'))),
		'precision': float("{:.3f}".format(precision_score(train_y, train_pred_y, average='weighted'))),
		'rmse': float("{:.3f}".format(np.sqrt(mean_squared_error(train_y, train_pred_y)))),
		'r2_score': float("{:.3f}".format(r2_score(train_y, train_pred_y)))
	}
	
	return rfc, train_metrics, test_metrics, test_pred_y, train_pred_y


def train_decision_tree_classifier(train_X, train_y, test_X, test_y):
	# # Create the base classifier
	# dtc = DecisionTreeClassifier(criterion='entropy', random_state=0)
	dtc = DecisionTreeClassifier()
	# Fit to training set
	dtc.fit(train_X, train_y)
	
	# Predict on test set
	test_pred_y = dtc.predict(test_X)
	
	# Predict on the training set
	train_pred_y = dtc.predict(train_X)
	
	# Calculate evaluation metrics for test set
	test_metrics = {
		'accuracy': float("{:.3f}".format(accuracy_score(test_y, test_pred_y))),
		'roc_auc': float("{:.3f}".format(roc_auc_score(test_y, test_pred_y))),
		'f1_score': float("{:.3f}".format(f1_score(test_y, test_pred_y, average='weighted'))),
		'recall': float("{:.3f}".format(recall_score(test_y, test_pred_y, average='weighted'))),
		'precision': float("{:.3f}".format(precision_score(test_y, test_pred_y, average='weighted'))),
		'rmse': float("{:.3f}".format(np.sqrt(mean_squared_error(test_y, test_pred_y)))),
		'r2_score': float("{:.3f}".format(r2_score(test_y, test_pred_y)))
	}
	
	# Calculate evaluation metrics for training set
	train_metrics = {
		'accuracy': float("{:.3f}".format(accuracy_score(train_y, train_pred_y))),
		'roc_auc': float("{:.3f}".format(roc_auc_score(train_y, train_pred_y))),
		'f1_score': float("{:.3f}".format(f1_score(train_y, train_pred_y, average='weighted'))),
		'recall': float("{:.3f}".format(recall_score(train_y, train_pred_y, average='weighted'))),
		'precision': float("{:.3f}".format(precision_score(train_y, train_pred_y, average='weighted'))),
		'rmse': float("{:.3f}".format(np.sqrt(mean_squared_error(train_y, train_pred_y)))),
		'r2_score': float("{:.3f}".format(r2_score(train_y, train_pred_y)))
	}
	
	return dtc, train_metrics, test_metrics, test_pred_y, train_pred_y



def train_support_vector_classifier(train_X, train_y, test_X, test_y):
	# # Create the base classifier
	# dtc = DecisionTreeClassifier(criterion='entropy', random_state=0)
	svc = DecisionTreeClassifier()
	# Fit to training set
	svc.fit(train_X, train_y)
	
	# Predict on test set
	test_pred_y = svc.predict(test_X)
	
	# Predict on the training set
	train_pred_y = svc.predict(train_X)
	
	# Calculate evaluation metrics for test set
	test_metrics = {
		'accuracy': float("{:.3f}".format(accuracy_score(test_y, test_pred_y))),
		'roc_auc': float("{:.3f}".format(roc_auc_score(test_y, test_pred_y))),
		'f1_score': float("{:.3f}".format(f1_score(test_y, test_pred_y, average='weighted'))),
		'recall': float("{:.3f}".format(recall_score(test_y, test_pred_y, average='weighted'))),
		'precision': float("{:.3f}".format(precision_score(test_y, test_pred_y, average='weighted'))),
		'rmse': float("{:.3f}".format(np.sqrt(mean_squared_error(test_y, test_pred_y)))),
		'r2_score': float("{:.3f}".format(r2_score(test_y, test_pred_y)))
	}
	
	# Calculate evaluation metrics for training set
	train_metrics = {
		'accuracy': float("{:.3f}".format(accuracy_score(train_y, train_pred_y))),
		'roc_auc': float("{:.3f}".format(roc_auc_score(train_y, train_pred_y))),
		'f1_score': float("{:.3f}".format(f1_score(train_y, train_pred_y, average='weighted'))),
		'recall': float("{:.3f}".format(recall_score(train_y, train_pred_y, average='weighted'))),
		'precision': float("{:.3f}".format(precision_score(train_y, train_pred_y, average='weighted'))),
		'rmse': float("{:.3f}".format(np.sqrt(mean_squared_error(train_y, train_pred_y)))),
		'r2_score': float("{:.3f}".format(r2_score(train_y, train_pred_y)))
	}
	
	return svc, train_metrics, test_metrics, test_pred_y, train_pred_y


def train_k_neighbors_classifier(train_X, train_y, test_X, test_y):
	# # Create the base classifier
	# dtc = DecisionTreeClassifier(criterion='entropy', random_state=0)
	knc = DecisionTreeClassifier()
	# Fit to training set
	knc.fit(train_X, train_y)
	
	# Predict on test set
	test_pred_y = knc.predict(test_X)
	
	# Predict on the training set
	train_pred_y = knc.predict(train_X)
	
	# Calculate evaluation metrics for test set
	test_metrics = {
		'accuracy': float("{:.3f}".format(accuracy_score(test_y, test_pred_y))),
		'roc_auc': float("{:.3f}".format(roc_auc_score(test_y, test_pred_y))),
		'f1_score': float("{:.3f}".format(f1_score(test_y, test_pred_y, average='weighted'))),
		'recall': float("{:.3f}".format(recall_score(test_y, test_pred_y, average='weighted'))),
		'precision': float("{:.3f}".format(precision_score(test_y, test_pred_y, average='weighted'))),
		'rmse': float("{:.3f}".format(np.sqrt(mean_squared_error(test_y, test_pred_y)))),
		'r2_score': float("{:.3f}".format(r2_score(test_y, test_pred_y)))
	}
	
	# Calculate evaluation metrics for training set
	train_metrics = {
		'accuracy': float("{:.3f}".format(accuracy_score(train_y, train_pred_y))),
		'roc_auc': float("{:.3f}".format(roc_auc_score(train_y, train_pred_y))),
		'f1_score': float("{:.3f}".format(f1_score(train_y, train_pred_y, average='weighted'))),
		'recall': float("{:.3f}".format(recall_score(train_y, train_pred_y, average='weighted'))),
		'precision': float("{:.3f}".format(precision_score(train_y, train_pred_y, average='weighted'))),
		'rmse': float("{:.3f}".format(np.sqrt(mean_squared_error(train_y, train_pred_y)))),
		'r2_score': float("{:.3f}".format(r2_score(train_y, train_pred_y)))
	}
	
	return knc, train_metrics, test_metrics, test_pred_y, train_pred_y


def train_k_neighbors_regressor(train_X, train_y, test_X, test_y, threshold):
	
	knr = KNeighborsRegressor(n_neighbors=5)

	knr.fit(train_X, train_y)

	# Predict on test set
	test_pred_y = knr.predict(test_X)
	test_pred_y = (test_pred_y >= threshold).astype(int)
	# Predict on the training set
	train_pred_y = knr.predict(train_X)
	train_pred_y = (train_pred_y >= threshold).astype(int)
	
	# Calculate evaluation metrics for test set
	test_metrics = {
		'accuracy': float("{:.3f}".format(accuracy_score(test_y, test_pred_y))),
		'roc_auc': float("{:.3f}".format(roc_auc_score(test_y, test_pred_y))),
		'f1_score': float("{:.3f}".format(f1_score(test_y, test_pred_y, average='weighted'))),
		'recall': float("{:.3f}".format(recall_score(test_y, test_pred_y, average='weighted'))),
		'precision': float("{:.3f}".format(precision_score(test_y, test_pred_y, average='weighted'))),
		'rmse': float("{:.3f}".format(np.sqrt(mean_squared_error(test_y, test_pred_y)))),
		'r2_score': float("{:.3f}".format(r2_score(test_y, test_pred_y)))
	}
	
	# Calculate evaluation metrics for training set
	train_metrics = {
		'accuracy': float("{:.3f}".format(accuracy_score(train_y, train_pred_y))),
		'roc_auc': float("{:.3f}".format(roc_auc_score(train_y, train_pred_y))),
		'f1_score': float("{:.3f}".format(f1_score(train_y, train_pred_y, average='weighted'))),
		'recall': float("{:.3f}".format(recall_score(train_y, train_pred_y, average='weighted'))),
		'precision': float("{:.3f}".format(precision_score(train_y, train_pred_y, average='weighted'))),
		'rmse': float("{:.3f}".format(np.sqrt(mean_squared_error(train_y, train_pred_y)))),
		'r2_score': float("{:.3f}".format(r2_score(train_y, train_pred_y)))
	}
	
	return knr, train_metrics, test_metrics, test_pred_y, train_pred_y


def train_gradient_boosting_regressor(train_X, train_y, test_X, test_y, threshold):
	
	# gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=1)
	gbr = GradientBoostingRegressor()

	gbr.fit(train_X, train_y)

	# Predict on test set
	test_pred_y = gbr.predict(test_X)
	test_pred_y = (test_pred_y >= threshold).astype(int)
	# Predict on the training set
	train_pred_y = gbr.predict(train_X)
	train_pred_y = (train_pred_y >= threshold).astype(int)
	
	# Calculate evaluation metrics for test set
	test_metrics = {
		'accuracy': float("{:.3f}".format(accuracy_score(test_y, test_pred_y))),
		'roc_auc': float("{:.3f}".format(roc_auc_score(test_y, test_pred_y))),
		'f1_score': float("{:.3f}".format(f1_score(test_y, test_pred_y, average='weighted'))),
		'recall': float("{:.3f}".format(recall_score(test_y, test_pred_y, average='weighted'))),
		'precision': float("{:.3f}".format(precision_score(test_y, test_pred_y, average='weighted'))),
		'rmse': float("{:.3f}".format(np.sqrt(mean_squared_error(test_y, test_pred_y)))),
		'r2_score': float("{:.3f}".format(r2_score(test_y, test_pred_y)))
	}
	
	# Calculate evaluation metrics for training set
	train_metrics = {
		'accuracy': float("{:.3f}".format(accuracy_score(train_y, train_pred_y))),
		'roc_auc': float("{:.3f}".format(roc_auc_score(train_y, train_pred_y))),
		'f1_score': float("{:.3f}".format(f1_score(train_y, train_pred_y, average='weighted'))),
		'recall': float("{:.3f}".format(recall_score(train_y, train_pred_y, average='weighted'))),
		'precision': float("{:.3f}".format(precision_score(train_y, train_pred_y, average='weighted'))),
		'rmse': float("{:.3f}".format(np.sqrt(mean_squared_error(train_y, train_pred_y)))),
		'r2_score': float("{:.3f}".format(r2_score(train_y, train_pred_y)))
	}
	
	return gbr, train_metrics, test_metrics, test_pred_y, train_pred_y


def train_xtreme_gradient_boosting_regressor(train_X, train_y, test_X, test_y, threshold):
	# gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=1)
	xgbr = XGBRegressor()

	xgbr.fit(train_X, train_y)

	# Predict on test set
	test_pred_y = xgbr.predict(test_X)
	test_pred_y = (test_pred_y >= threshold).astype(int)
	# Predict on the training set
	train_pred_y = xgbr.predict(train_X)
	train_pred_y = (train_pred_y >= threshold).astype(int)
	
	# Calculate evaluation metrics for test set
	test_metrics = {
		'accuracy': float("{:.3f}".format(accuracy_score(test_y, test_pred_y))),
		'roc_auc': float("{:.3f}".format(roc_auc_score(test_y, test_pred_y))),
		'f1_score': float("{:.3f}".format(f1_score(test_y, test_pred_y, average='weighted'))),
		'recall': float("{:.3f}".format(recall_score(test_y, test_pred_y, average='weighted'))),
		'precision': float("{:.3f}".format(precision_score(test_y, test_pred_y, average='weighted'))),
		'rmse': float("{:.3f}".format(np.sqrt(mean_squared_error(test_y, test_pred_y)))),
		'r2_score': float("{:.3f}".format(r2_score(test_y, test_pred_y)))
	}
	
	# Calculate evaluation metrics for training set
	train_metrics = {
		'accuracy': float("{:.3f}".format(accuracy_score(train_y, train_pred_y))),
		'roc_auc': float("{:.3f}".format(roc_auc_score(train_y, train_pred_y))),
		'f1_score': float("{:.3f}".format(f1_score(train_y, train_pred_y, average='weighted'))),
		'recall': float("{:.3f}".format(recall_score(train_y, train_pred_y, average='weighted'))),
		'precision': float("{:.3f}".format(precision_score(train_y, train_pred_y, average='weighted'))),
		'rmse': float("{:.3f}".format(np.sqrt(mean_squared_error(train_y, train_pred_y)))),
		'r2_score': float("{:.3f}".format(r2_score(train_y, train_pred_y)))
	}
	
	return xgbr, train_metrics, test_metrics, test_pred_y, train_pred_y


def train_xtreme_gradient_boosting_classifier(train_X, train_y, test_X, test_y):
	# # Create the base classifier
	# dtc = DecisionTreeClassifier(criterion='entropy', random_state=0)
	xgbc = XGBClassifier()
	# Fit to training set
	xgbc.fit(train_X, train_y)
	
	# Predict on test set
	test_pred_y = xgbc.predict(test_X)
	
	# Predict on the training set
	train_pred_y = xgbc.predict(train_X)
	
	# Calculate evaluation metrics for test set
	test_metrics = {
		'accuracy': float("{:.3f}".format(accuracy_score(test_y, test_pred_y))),
		'roc_auc': float("{:.3f}".format(roc_auc_score(test_y, test_pred_y))),
		'f1_score': float("{:.3f}".format(f1_score(test_y, test_pred_y, average='weighted'))),
		'recall': float("{:.3f}".format(recall_score(test_y, test_pred_y, average='weighted'))),
		'precision': float("{:.3f}".format(precision_score(test_y, test_pred_y, average='weighted'))),
		'rmse': float("{:.3f}".format(np.sqrt(mean_squared_error(test_y, test_pred_y)))),
		'r2_score': float("{:.3f}".format(r2_score(test_y, test_pred_y)))
	}
	
	# Calculate evaluation metrics for training set
	train_metrics = {
		'accuracy': float("{:.3f}".format(accuracy_score(train_y, train_pred_y))),
		'roc_auc': float("{:.3f}".format(roc_auc_score(train_y, train_pred_y))),
		'f1_score': float("{:.3f}".format(f1_score(train_y, train_pred_y, average='weighted'))),
		'recall': float("{:.3f}".format(recall_score(train_y, train_pred_y, average='weighted'))),
		'precision': float("{:.3f}".format(precision_score(train_y, train_pred_y, average='weighted'))),
		'rmse': float("{:.3f}".format(np.sqrt(mean_squared_error(train_y, train_pred_y)))),
		'r2_score': float("{:.3f}".format(r2_score(train_y, train_pred_y)))
	}
	
	return xgbc, train_metrics, test_metrics, test_pred_y, train_pred_y


def train_random_forest__regressor(train_X, train_y, test_X, test_y, threshold):
	# gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=1)
	rfr = RandomForestRegressor()

	rfr.fit(train_X, train_y)

	# Predict on test set
	test_pred_y = rfr.predict(test_X)
	test_pred_y = (test_pred_y >= threshold).astype(int)
	# Predict on the training set
	train_pred_y = rfr.predict(train_X)
	train_pred_y = (train_pred_y >= threshold).astype(int)
	
	# Calculate evaluation metrics for test set
	test_metrics = {
		'accuracy': float("{:.3f}".format(accuracy_score(test_y, test_pred_y))),
		'roc_auc': float("{:.3f}".format(roc_auc_score(test_y, test_pred_y))),
		'f1_score': float("{:.3f}".format(f1_score(test_y, test_pred_y, average='weighted'))),
		'recall': float("{:.3f}".format(recall_score(test_y, test_pred_y, average='weighted'))),
		'precision': float("{:.3f}".format(precision_score(test_y, test_pred_y, average='weighted'))),
		'rmse': float("{:.3f}".format(np.sqrt(mean_squared_error(test_y, test_pred_y)))),
		'r2_score': float("{:.3f}".format(r2_score(test_y, test_pred_y)))
	}
	
	# Calculate evaluation metrics for training set
	train_metrics = {
		'accuracy': float("{:.3f}".format(accuracy_score(train_y, train_pred_y))),
		'roc_auc': float("{:.3f}".format(roc_auc_score(train_y, train_pred_y))),
		'f1_score': float("{:.3f}".format(f1_score(train_y, train_pred_y, average='weighted'))),
		'recall': float("{:.3f}".format(recall_score(train_y, train_pred_y, average='weighted'))),
		'precision': float("{:.3f}".format(precision_score(train_y, train_pred_y, average='weighted'))),
		'rmse': float("{:.3f}".format(np.sqrt(mean_squared_error(train_y, train_pred_y)))),
		'r2_score': float("{:.3f}".format(r2_score(train_y, train_pred_y)))
	}
	
	return rfr, train_metrics, test_metrics, test_pred_y, train_pred_y


def train_support_vector_regressor(train_X, train_y, test_X, test_y, threshold):
	# gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=1)
	svr = SVR()

	svr.fit(train_X, train_y)

	# Predict on test set
	test_pred_y = svr.predict(test_X)
	test_pred_y = (test_pred_y >= threshold).astype(int)
	# Predict on the training set
	train_pred_y = svr.predict(train_X)
	train_pred_y = (train_pred_y >= threshold).astype(int)
	
	# Calculate evaluation metrics for test set
	test_metrics = {
		'accuracy': float("{:.3f}".format(accuracy_score(test_y, test_pred_y))),
		'roc_auc': float("{:.3f}".format(roc_auc_score(test_y, test_pred_y))),
		'f1_score': float("{:.3f}".format(f1_score(test_y, test_pred_y, average='weighted'))),
		'recall': float("{:.3f}".format(recall_score(test_y, test_pred_y, average='weighted'))),
		'precision': float("{:.3f}".format(precision_score(test_y, test_pred_y, average='weighted'))),
		'rmse': float("{:.3f}".format(np.sqrt(mean_squared_error(test_y, test_pred_y)))),
		'r2_score': float("{:.3f}".format(r2_score(test_y, test_pred_y)))
	}
	
	# Calculate evaluation metrics for training set
	train_metrics = {
		'accuracy': float("{:.3f}".format(accuracy_score(train_y, train_pred_y))),
		'roc_auc': float("{:.3f}".format(roc_auc_score(train_y, train_pred_y))),
		'f1_score': float("{:.3f}".format(f1_score(train_y, train_pred_y, average='weighted'))),
		'recall': float("{:.3f}".format(recall_score(train_y, train_pred_y, average='weighted'))),
		'precision': float("{:.3f}".format(precision_score(train_y, train_pred_y, average='weighted'))),
		'rmse': float("{:.3f}".format(np.sqrt(mean_squared_error(train_y, train_pred_y)))),
		'r2_score': float("{:.3f}".format(r2_score(train_y, train_pred_y)))
	}
	
	return svr, train_metrics, test_metrics, test_pred_y, train_pred_y



def train_decision_tree_regressor(train_X, train_y, test_X, test_y, threshold):
	dtr = DecisionTreeRegressor()

	dtr.fit(train_X, train_y)

	# Predict on test set
	test_pred_y = dtr.predict(test_X)
	test_pred_y = (test_pred_y >= threshold).astype(int)
	# Predict on the training set
	train_pred_y = dtr.predict(train_X)
	train_pred_y = (train_pred_y >= threshold).astype(int)
	
	# Calculate evaluation metrics for test set
	test_metrics = {
		'accuracy': float("{:.3f}".format(accuracy_score(test_y, test_pred_y))),
		'roc_auc': float("{:.3f}".format(roc_auc_score(test_y, test_pred_y))),
		'f1_score': float("{:.3f}".format(f1_score(test_y, test_pred_y, average='weighted'))),
		'recall': float("{:.3f}".format(recall_score(test_y, test_pred_y, average='weighted'))),
		'precision': float("{:.3f}".format(precision_score(test_y, test_pred_y, average='weighted'))),
		'rmse': float("{:.3f}".format(np.sqrt(mean_squared_error(test_y, test_pred_y)))),
		'r2_score': float("{:.3f}".format(r2_score(test_y, test_pred_y)))
	}
	
	# Calculate evaluation metrics for training set
	train_metrics = {
		'accuracy': float("{:.3f}".format(accuracy_score(train_y, train_pred_y))),
		'roc_auc': float("{:.3f}".format(roc_auc_score(train_y, train_pred_y))),
		'f1_score': float("{:.3f}".format(f1_score(train_y, train_pred_y, average='weighted'))),
		'recall': float("{:.3f}".format(recall_score(train_y, train_pred_y, average='weighted'))),
		'precision': float("{:.3f}".format(precision_score(train_y, train_pred_y, average='weighted'))),
		'rmse': float("{:.3f}".format(np.sqrt(mean_squared_error(train_y, train_pred_y)))),
		'r2_score': float("{:.3f}".format(r2_score(train_y, train_pred_y)))
	}
	
	return dtr, train_metrics, test_metrics, test_pred_y, train_pred_y



def train_linear_regression(train_X, train_y, test_X, test_y, threshold):
	slr = LinearRegression()

	slr.fit(train_X, train_y)

	# Predict on test set
	test_pred_y = slr.predict(test_X)
	test_pred_y = (test_pred_y >= threshold).astype(int)
	# Predict on the training set
	train_pred_y = slr.predict(train_X)
	train_pred_y = (train_pred_y >= threshold).astype(int)
	
	# Calculate evaluation metrics for test set
	test_metrics = {
		'accuracy': float("{:.3f}".format(accuracy_score(test_y, test_pred_y))),
		'roc_auc': float("{:.3f}".format(roc_auc_score(test_y, test_pred_y))),
		'f1_score': float("{:.3f}".format(f1_score(test_y, test_pred_y, average='weighted'))),
		'recall': float("{:.3f}".format(recall_score(test_y, test_pred_y, average='weighted'))),
		'precision': float("{:.3f}".format(precision_score(test_y, test_pred_y, average='weighted'))),
		'rmse': float("{:.3f}".format(np.sqrt(mean_squared_error(test_y, test_pred_y)))),
		'r2_score': float("{:.3f}".format(r2_score(test_y, test_pred_y)))
	}
	
	# Calculate evaluation metrics for training set
	train_metrics = {
		'accuracy': float("{:.3f}".format(accuracy_score(train_y, train_pred_y))),
		'roc_auc': float("{:.3f}".format(roc_auc_score(train_y, train_pred_y))),
		'f1_score': float("{:.3f}".format(f1_score(train_y, train_pred_y, average='weighted'))),
		'recall': float("{:.3f}".format(recall_score(train_y, train_pred_y, average='weighted'))),
		'precision': float("{:.3f}".format(precision_score(train_y, train_pred_y, average='weighted'))),
		'rmse': float("{:.3f}".format(np.sqrt(mean_squared_error(train_y, train_pred_y)))),
		'r2_score': float("{:.3f}".format(r2_score(train_y, train_pred_y)))
	}
	
	return slr, train_metrics, test_metrics, test_pred_y, train_pred_y



def train_reg_binary_(train_X, train_y, test_X, test_y, threshold, model1):
	model = model1

	model.fit(train_X, train_y)

	# Predict on test set
	test_pred_y = model.predict(test_X)
	test_pred_y = (test_pred_y >= threshold).astype(int)
	# Predict on the training set
	train_pred_y = model.predict(train_X)
	train_pred_y = (train_pred_y >= threshold).astype(int)
	
	# Calculate evaluation metrics for test set
	test_metrics = {
		'accuracy': float("{:.3f}".format(accuracy_score(test_y, test_pred_y))),
		'roc_auc': float("{:.3f}".format(roc_auc_score(test_y, test_pred_y))),
		'f1_score': float("{:.3f}".format(f1_score(test_y, test_pred_y, average='weighted'))),
		'recall': float("{:.3f}".format(recall_score(test_y, test_pred_y, average='weighted'))),
		'precision': float("{:.3f}".format(precision_score(test_y, test_pred_y, average='weighted'))),
		'rmse': float("{:.3f}".format(np.sqrt(mean_squared_error(test_y, test_pred_y)))),
		'r2_score': float("{:.3f}".format(r2_score(test_y, test_pred_y)))
	}
	
	# Calculate evaluation metrics for training set
	train_metrics = {
		'accuracy': float("{:.3f}".format(accuracy_score(train_y, train_pred_y))),
		'roc_auc': float("{:.3f}".format(roc_auc_score(train_y, train_pred_y))),
		'f1_score': float("{:.3f}".format(f1_score(train_y, train_pred_y, average='weighted'))),
		'recall': float("{:.3f}".format(recall_score(train_y, train_pred_y, average='weighted'))),
		'precision': float("{:.3f}".format(precision_score(train_y, train_pred_y, average='weighted'))),
		'rmse': float("{:.3f}".format(np.sqrt(mean_squared_error(train_y, train_pred_y)))),
		'r2_score': float("{:.3f}".format(r2_score(train_y, train_pred_y)))
	}
	
	return model, train_metrics, test_metrics, test_pred_y, train_pred_y



def train_cls_binary_(train_X, train_y, test_X, test_y, model1):
	# # Create the base classifier
	# dtc = DecisionTreeClassifier(criterion='entropy', random_state=0)
	model = model1
	# Fit to training set
	model.fit(train_X, train_y)
	
	# Predict on test set
	test_pred_y = model.predict(test_X)
	
	# Predict on the training set
	train_pred_y = model.predict(train_X)
	
	# Calculate evaluation metrics for test set
	test_metrics = {
		'accuracy': float("{:.3f}".format(accuracy_score(test_y, test_pred_y))),
		'roc_auc': float("{:.3f}".format(roc_auc_score(test_y, test_pred_y))),
		'f1_score': float("{:.3f}".format(f1_score(test_y, test_pred_y, average='weighted'))),
		'recall': float("{:.3f}".format(recall_score(test_y, test_pred_y, average='weighted'))),
		'precision': float("{:.3f}".format(precision_score(test_y, test_pred_y, average='weighted'))),
		'rmse': float("{:.3f}".format(np.sqrt(mean_squared_error(test_y, test_pred_y)))),
		'r2_score': float("{:.3f}".format(r2_score(test_y, test_pred_y)))
	}
	
	# Calculate evaluation metrics for training set
	train_metrics = {
		'accuracy': float("{:.3f}".format(accuracy_score(train_y, train_pred_y))),
		'roc_auc': float("{:.3f}".format(roc_auc_score(train_y, train_pred_y))),
		'f1_score': float("{:.3f}".format(f1_score(train_y, train_pred_y, average='weighted'))),
		'recall': float("{:.3f}".format(recall_score(train_y, train_pred_y, average='weighted'))),
		'precision': float("{:.3f}".format(precision_score(train_y, train_pred_y, average='weighted'))),
		'rmse': float("{:.3f}".format(np.sqrt(mean_squared_error(train_y, train_pred_y)))),
		'r2_score': float("{:.3f}".format(r2_score(train_y, train_pred_y)))
	}
	
	return model, train_metrics, test_metrics, test_pred_y, train_pred_y

def train_cls_mcc_(train_X, train_y, test_X, test_y, model1):
	model = model1
	model.fit(train_X, train_y)

	# Predict on test set
	test_pred_y = model.predict(test_X)
	# Predict on the training set
	train_pred_y = model.predict(train_X)

	# Calculate evaluation metrics for test set
	test_metrics = {
		'accuracy': accuracy_score(test_y, test_pred_y),
		'precision': precision_score(test_y, test_pred_y, average='weighted'),
		'recall': recall_score(test_y, test_pred_y, average='weighted'),
		'f1_score': f1_score(test_y, test_pred_y, average='weighted'),
		'confusion_matrix': confusion_matrix(test_y, test_pred_y)
	}

	# Calculate evaluation metrics for training set
	train_metrics = {
		'accuracy': accuracy_score(train_y, train_pred_y),
		'precision': precision_score(train_y, train_pred_y, average='weighted'),
		'recall': recall_score(train_y, train_pred_y, average='weighted'),
		'f1_score': f1_score(train_y, train_pred_y, average='weighted')
	}

	return model, train_metrics, test_metrics, test_pred_y, train_pred_y


def train_reg_cout_(train_X, train_y, test_X, test_y, model1):
	model = model1
	model.fit(train_X, train_y)

	# Predict on test set
	test_pred_y = model.predict(test_X)
	# Predict on the training set
	train_pred_y = model.predict(train_X)

	# Calculate evaluation metrics for test set
	test_metrics = {
		'rmse': np.sqrt(mean_squared_error(test_y, test_pred_y)),
		'r2_score': r2_score(test_y, test_pred_y),
		'mae': mean_absolute_error(test_y, test_pred_y),
		# 'msle': mean_squared_log_error(test_y, test_pred_y),
		'mape': np.mean(np.abs((test_y - test_pred_y) / test_y)) * 100
	}

	# Calculate evaluation metrics for training set
	train_metrics = {
		'rmse': np.sqrt(mean_squared_error(train_y, train_pred_y)),
		'r2_score': r2_score(train_y, train_pred_y),
		'mae': mean_absolute_error(train_y, train_pred_y),
		# 'msle': mean_squared_log_error(train_y, train_pred_y),
		'mape': np.mean(np.abs((train_y - train_pred_y) / train_y)) * 100
	}

	return model, train_metrics, test_metrics, test_pred_y, train_pred_y


st.title("Model Building")

st.write("Session State:->", st.session_state["shared"])
if "X_train" in st.session_state and "X_test" in st.session_state and "y_train" in st.session_state and "y_test" in st.session_state:
	X_train = st.session_state.X_train
	X_test = st.session_state.X_test
	y_train = st.session_state.y_train
	y_test = st.session_state.y_test
	# y_test = y_test.map({1: "Yes", 0: "No"})
	# st.write(y_test)
	# y_train = y_train.map({1: "Yes", 0: "No"})
	# X_test = X_test.map({1: "Yes", 0: "No"})
	# st.write(X_test)
	X_test.replace({True: 1, False: 0}, inplace=True)
	# st.write(X_test)
	# X_train = X_train.map({1: "Yes", 0: "No"})
	X_train.replace({True: 1, False: 0}, inplace=True)
	
	# st.write("X_train")
	# # st.dataframe(X_train)
	# # Check for missing values
	# missing_values = X_train.isna().sum()
	# st.write(missing_values)
	# st.write("X_test")
	# # st.dataframe(X_test)
	# missing_values = X_test.isna().sum()
	# st.write(missing_values)
	# st.write("y_train")
	# # st.dataframe(y_train)
	# missing_values = y_train.isna().sum()
	# st.write(missing_values)
	# st.write("y_test")
	# # st.dataframe(y_test)
	# missing_values = y_test.isna().sum()
	# st.write(missing_values)
	# print("type_=", type(y_test))
	# y_test = y_test.to_frame()
	# y_train = y_train.to_frame()
else:
	st.error('This is an error', icon="🚨")
	st.warning('Check if you have done proper preprocessing and data spliting or not!')

# Available models
models = {
	"Extreme Gradient Boosting Classifier": XGBClassifier(),
	"Extreme Gradient Boosting Regressor": XGBRegressor(),
	"Linear Regression": LinearRegression(),
	"Gradient Boosting Classifier": GradientBoostingClassifier(),
	"Gradient Boosting Regressor": GradientBoostingRegressor(),
	"Bagging Classifier": BaggingClassifier(),
	# "Multinomial Naive Bayes": MultinomialNB(),
	"Ada Boost Classifier": AdaBoostClassifier(),
	"Logistic Regression": LogisticRegression(),
	"Random Forest Classifier": RandomForestClassifier(),
	"Random Forest Regressor": RandomForestRegressor(),
	"Support Vector Classifier": SVC(),
	"Support Vector Regressor": SVR(),
	"K-Neighbors Classifier": KNeighborsClassifier(),
	"K-Neighbors Regressor": KNeighborsRegressor(),
	"Decision Tree Classifier": DecisionTreeClassifier(),
	"Decision Tree Regressor": DecisionTreeRegressor()
}

# Model selection using multiselect
# selected_models = st.multiselect("Select models to train", list(models.keys()))
# Model selection using multiselect

selected_type = st.radio("Select Model Type", ["Regression Models", "Regression Models (continuous data)", "Classification Models", "Multi Class Classification"], captions=["Binary Output Variable", "Continuous Output Variable", "Binary Output Variable", "Output Variable with more than 2 class"])
st.sidebar.write(":blue[After training the models, click below to proceed ⤵️]")
if st.sidebar.button("Supervised_Deployment_Demo"):	
	st.switch_page(r"pages/12_Supervised_Deployment_Demo.py")

if st.sidebar.button("Custom_Model_Training"):
	st.switch_page(r"pages/11_Custom_Model_Training.py")
# if selected_type == "All Models":
# 	selected_models = st.multiselect("Select models to train", list(models.keys()))
if selected_type == "Regression Models":
	# UI to select threshold
	threshold = st.slider('Threshold for Binary Prediction', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
	selected_models = st.multiselect("Select models to train", [key for key, model in models.items() if isinstance(model, (LinearRegression, XGBRegressor, GradientBoostingRegressor, RandomForestRegressor, SVR, KNeighborsRegressor, DecisionTreeRegressor))])
elif selected_type == "Regression Models (continuous data)":
	selected_models = st.multiselect("Select models to train", [key for key, model in models.items() if isinstance(model, (LinearRegression, XGBRegressor, GradientBoostingRegressor, RandomForestRegressor, SVR, KNeighborsRegressor, DecisionTreeRegressor))])
elif selected_type == "Classification Models":  # Classification Models
	selected_models = st.multiselect("Select models to train", [key for key, model in models.items() if isinstance(model, (XGBClassifier, LogisticRegression, GradientBoostingClassifier, BaggingClassifier, AdaBoostClassifier, RandomForestClassifier, SVC, KNeighborsClassifier, DecisionTreeClassifier))])
elif selected_type == "Multi Class Classification":
	selected_models = st.multiselect("Select models to train", [key for key, model in models.items() if isinstance(model, (XGBClassifier, LogisticRegression, GradientBoostingClassifier, BaggingClassifier, AdaBoostClassifier, RandomForestClassifier, SVC, KNeighborsClassifier, DecisionTreeClassifier))])
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
			model_function = models[model_name]
			if len(selected_models) <= 5 and len(str(model_function)) <= 30:
				st.write(model_function)
				# st.write(len(str(model_function)))
			else:
				st.subheader(f":violet[{model_name}]", divider=True)
			# st.write(f"Training {model_name}...")
			# if model_name == 'Linear Regression':
			# 	model_function = train_linear_regression
			# elif model_name == 'Gradient Boosting Classifier':
			# 	model_function = train_gradient_boosting_classifier
			# elif model_name == 'Bagging Classifier':
			# 	model_function = train_bagging_classifier
			# elif model_name == 'Multinomial Naive Bayes':
			# 	model_function = train_multinomial_naive_bayes
			# elif model_name == 'Logistic Regression':
			# 	model_function = train_logistic_regression
			# elif model_name == 'Ada Boost Classifier':
			# 	model_function = train_ada_boost_classifier
			# elif model_name == 'Random Forest Classifier':
			# 	model_function = train_random_forest_classifier
			# elif model_name == 'Decision Tree Classifier':
			# 	model_function = train_decision_tree_classifier
			# elif model_name == 'K-Neighbors Regressor':
			# 	model_function = train_k_neighbors_regressor
			# elif model_name == 'Gradient Boosting Regressor':
			# 	model_function = train_gradient_boosting_regressor
			# elif model_name == 'Extreme Gradient Boosting Regressor':
			# 	model_function = train_xtreme_gradient_boosting_regressor
			# elif model_name == 'Extreme Gradient Boosting Classifier':
			# 	model_function = train_xtreme_gradient_boosting_classifier
			# elif model_name == 'Random Forest Regressor':
			# 	model_function = train_random_forest__regressor
			# elif model_name == 'Support Vector Regressor':
			# 	model_function = train_support_vector_regressor
			# elif model_name == 'Decision Tree Regressor':
			# 	model_function = train_decision_tree_regressor
			# elif model_name == 'Support Vector Classifier':
			# 	model_function = train_support_vector_classifier
			# elif model_name == 'K-Neighbors Classifier':
			# 	model_function = train_k_neighbors_classifier
			# if selected_type == "Regression Models":
				# print("model_function:=", model_function)
			

			# Perform model training here
			# Example: model_instance.fit(X_train, y_train)
			# st.dataframe(X_train)
			# Define an empty list to store the metrics of each model
			# Create an empty DataFrame to store the metrics of each model
			

			try:
				if selected_type == 'Regression Models':
					model, train_metrics, test_metrics, test_pred_y, train_pred_y = train_reg_binary_(X_train, y_train, X_test, y_test, threshold, model_function)
				elif selected_type == 'Regression Models (continuous data)':
					model, train_metrics, test_metrics, test_pred_y, train_pred_y = train_reg_cout_(X_train, y_train, X_test, y_test, model_function)
				elif selected_type == 'Classification Models':
					model, train_metrics, test_metrics, test_pred_y, train_pred_y = train_cls_binary_(X_train, y_train, X_test, y_test, model_function)
				elif selected_type == 'Multi Class Classification':
					model, train_metrics, test_metrics, test_pred_y, train_pred_y = train_cls_mcc_(X_train, y_train, X_test, y_test, model_function)
				# st.success(f"{model_name} trained successfully!")
				# Display evaluation metrics in Streamlit UI
				# Append the metrics of the current model to the list
				# Append the metrics to the DataFrame
				# Append the metrics to the list
				if selected_type == "Regression Models" or selected_type == "Classification Models":
					try:
						all_metrics_data.append({
							'Model': model_name,
							'Accuracy': test_metrics['accuracy'],
							'ROC AUC': test_metrics['roc_auc'],
							'F1 Score': test_metrics['f1_score'],
							'Recall': test_metrics['recall'],
							'Precision': test_metrics['precision'],
							'RMSE': test_metrics['rmse'],
							'R2 Score': test_metrics['r2_score']
						})
					except:
						pass
				elif selected_type == "Multi Class Classification":
					try:
						all_metrics_data.append({
							'Model': model_name,
							'Accuracy': test_metrics['accuracy'],
							# 'ROC AUC': test_metrics['roc_auc'],
							'F1 Score': test_metrics['f1_score'],
							'Recall': test_metrics['recall'],
							'Precision': test_metrics['precision'],
							# 'RMSE': test_metrics['rmse'],
							# 'R2 Score': test_metrics['r2_score']
						})
					except:
						pass
				elif selected_type == "Regression Models (continuous data)":
					try:
						all_metrics_data.append({
							'Model': model_name,
							'RMSE': test_metrics['rmse'],
							'R2 Score': test_metrics['r2_score'],
							# 'MSLE': test_metrics['msle'],
							'MAE': test_metrics['mae'],
							'MAPE': test_metrics['mape'],
						})
					except:
						pass
				st.write("Testing Metrics")
				st.write(test_metrics)

				st.write("Training Metrics")
				st.write(train_metrics)
				# Call the function
				# st.write(dir(model))
				# model_name = str(model)[0:len(str(model))-2]
				st.write(model_name)
				with st.expander("Model Evalution:"):
					try:
						plot_evaluation_metrics(train_metrics, test_metrics, test_pred_y, train_pred_y, y_test)
					except:
						pass	
				model_info = {
					'model': model,  # The trained model
					'train_metrics': train_metrics,  # Metrics from the training set
					'test_metrics': test_metrics,  # Metrics from the test set
					'model_name': model_name,  # Name of the model
				}

				# Dump to a joblib file
				model_filename = f"{model_name}_model.pkl"
				model_filename = model_filename.replace(" ", "")
				# Save the dictionary containing the model and its metrics
				with open(model_filename, "wb") as f:
					joblib.dump(model_info, f)
		
			except Exception as e:
				st.write(e)
				# st.title("GAURANG")
				pass

if selected_type == "Regression Models" or selected_type == "Classification Models":
	try:
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
			fig_bar = px.bar(all_metrics_df, x='Model', y=['Accuracy', 'ROC AUC', 'F1 Score', 'Recall', 'Precision', 'RMSE', 'R2 Score'], 
							barmode='group', title='Model Performance Metrics')
			fig_bar.update_layout(xaxis_title='Model', yaxis_title='Metric Value')
			st.plotly_chart(fig_bar)
			# Violin Plot
			fig_violin = go.Figure()
			for metric in ['Accuracy', 'ROC AUC', 'F1 Score', 'Recall', 'Precision', 'RMSE', 'R2 Score']:
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
				fig_radar.add_trace(go.Scatterpolar(r=row[['Accuracy', 'ROC AUC', 'F1 Score', 'Recall', 'Precision', 'RMSE', 'R2 Score']].values,
													theta=['Accuracy', 'ROC AUC', 'F1 Score', 'Recall', 'Precision', 'RMSE', 'R2 Score'],
													fill='toself', name=row['Model']))
			fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), title='Radar Chart - Performance Metrics')
			st.plotly_chart(fig_radar)
			# Line plot
			fig_line = px.line(all_metrics_df, x='Model', y=['Accuracy', 'ROC AUC', 'F1 Score', 'Recall', 'Precision', 'RMSE', 'R2 Score'], 
							title='Model Performance Metrics')
			fig_line.update_layout(xaxis_title='Model', yaxis_title='Metric Value')
			st.plotly_chart(fig_line)
			# Display the DataFrame with all the metrics
			st.write("All Model Metrics:")
			st.write(all_metrics_df)
	except:
		pass
elif selected_type == "Multi Class Classification":
	try:
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
			fig_bar = px.bar(all_metrics_df, x='Model', y=['Accuracy',  'F1 Score', 'Recall', 'Precision'], 
							barmode='group', title='Model Performance Metrics')
			fig_bar.update_layout(xaxis_title='Model', yaxis_title='Metric Value')
			st.plotly_chart(fig_bar)
			# Violin Plot
			fig_violin = go.Figure()
			for metric in ['Accuracy', 'F1 Score', 'Recall', 'Precision']:
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
				fig_radar.add_trace(go.Scatterpolar(r=row[['Accuracy', 'F1 Score', 'Recall', 'Precision']].values,
													theta=['Accuracy', 'F1 Score', 'Recall', 'Precision'],
													fill='toself', name=row['Model']))
			fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), title='Radar Chart - Performance Metrics')
			st.plotly_chart(fig_radar)
			# Line plot
			fig_line = px.line(all_metrics_df, x='Model', y=['Accuracy', 'F1 Score', 'Recall', 'Precision'], 
							title='Model Performance Metrics')
			fig_line.update_layout(xaxis_title='Model', yaxis_title='Metric Value')
			st.plotly_chart(fig_line)
			# Display the DataFrame with all the metrics
			st.write("All Model Metrics:")
			st.write(all_metrics_df)
	except:
		pass

elif selected_type == "Regression Models (continuous data)":
	try:
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
			fig_bar = px.bar(all_metrics_df, x='Model', y=['RMSE', 'R2 Score', 'R2 Score', 'MAE', 'MAPE'], 
							barmode='group', title='Model Performance Metrics')
			fig_bar.update_layout(xaxis_title='Model', yaxis_title='Metric Value')
			st.plotly_chart(fig_bar)
			# Violin Plot
			fig_violin = go.Figure()
			for metric in ['RMSE', 'R2 Score', 'R2 Score', 'MAE', 'MAPE']:
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
				fig_radar.add_trace(go.Scatterpolar(r=row[['RMSE', 'R2 Score', 'R2 Score', 'MAE', 'MAPE']].values,
													theta=['RMSE', 'R2 Score', 'R2 Score', 'MAE', 'MAPE'],
													fill='toself', name=row['Model']))
			fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), title='Radar Chart - Performance Metrics')
			st.plotly_chart(fig_radar)
			# Line plot
			fig_line = px.line(all_metrics_df, x='Model', y=['RMSE', 'R2 Score', 'R2 Score', 'MAE', 'MAPE'], 
							title='Model Performance Metrics')
			fig_line.update_layout(xaxis_title='Model', yaxis_title='Metric Value')
			st.plotly_chart(fig_line)
			# Display the DataFrame with all the metrics
			st.write("All Model Metrics:")
			st.write(all_metrics_df)
	except:
		pass

# After the loop ends, compare the metrics of all models
# import pandas as pd
# if all_test_metrics:
# 	# Combine all test metrics into a DataFrame for easy comparison
# 	df_test_metrics = pd.DataFrame(all_test_metrics)
	
# 	# Plot comparison of test metrics
# 	st.write("Comparison of Test Metrics")
# 	st.write(df_test_metrics)
	
# 	# Combine all train metrics into a DataFrame for easy comparison
# 	df_train_metrics = pd.DataFrame(all_train_metrics)
	
# 	# Plot comparison of train metrics
# 	st.write("Comparison of Train Metrics")
# 	st.write(df_train_metrics)
				
				