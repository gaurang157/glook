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
from sklearn.naive_bayes import MultinomialNB
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error
from sklearn.cluster import (
	KMeans, DBSCAN, AgglomerativeClustering, MeanShift, Birch,
	AffinityPropagation, SpectralClustering, OPTICS, 
	MiniBatchKMeans, FeatureAgglomeration, HDBSCAN
)
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import (
	 PCA, 
	 NMF, 
	 FastICA, 
	 FactorAnalysis, 
	 DictionaryLearning, 
	 TruncatedSVD
	 )
from sklearn.metrics import (
	silhouette_score, 
	davies_bouldin_score, 
	calinski_harabasz_score,
	adjusted_rand_score, 
	adjusted_mutual_info_score, 
	completeness_score,
	homogeneity_score, 
	v_measure_score
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error
from sklearn.metrics import (
	accuracy_score, f1_score, mean_squared_error, r2_score, roc_auc_score,
	recall_score, precision_score, roc_curve, auc, confusion_matrix,
	classification_report
)
import joblib
from copy import deepcopy

def train_xtreme_gradient_boosting_regressor(train_X, train_y, test_X, test_y, params):
	# gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=1)
	xgbr = XGBRegressor(**params)

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

def train_gradient_boosting_regressor(train_X, train_y, test_X, test_y, params):
	# gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=1)
	xgbr = GradientBoostingRegressor(**params)

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

def train_linear_regression(train_X, train_y, test_X, test_y, params):
	slr = LinearRegression(**params)

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


def train_random_forest__regressor(train_X, train_y, test_X, test_y, params):
	# gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=1)
	rfr = RandomForestRegressor(**params)

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

def train_support_vector_regressor(train_X, train_y, test_X, test_y, params):
	# gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=1)
	svr = SVR(**params)

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

def train_k_neighbors_regressor(train_X, train_y, test_X, test_y, params):
	
	knr = KNeighborsRegressor(**params)

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


def train_decision_tree_regressor(train_X, train_y, test_X, test_y, params):
	dtr = DecisionTreeRegressor(**params)

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

def train_xtreme_gradient_boosting_classifier(train_X, train_y, test_X, test_y, params):
	# # Create the base classifier
	# dtc = DecisionTreeClassifier(criterion='entropy', random_state=0)
	xgbc = XGBClassifier(**params)
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


def train_gradient_boosting_classifier(train_X, train_y, test_X, test_y, params):
	
	gbc = GradientBoostingClassifier(**params)
	
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


def train_bagging_classifier(train_X, train_y, test_X, test_y, params):
	# # Create the base classifier
	dc = DecisionTreeClassifier()
	# Create the BaggingClassifier
	bc = BaggingClassifier(base_estimator=dc, **params)
	
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


def train_multinomial_naive_bayes(train_X, train_y, test_X, test_y, alpha):
	# # Create the base classifier
	nb = MultinomialNB(alpha=alpha)
	
	# Fit to training set
	nb.fit(train_X, train_y)
	
	# Predict on test set
	test_pred_y = nb.predict(test_X)
	
	# Predict on the training set
	train_pred_y = nb.predict(train_X)
	
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
	
	return nb, train_metrics, test_metrics, test_pred_y, train_pred_y

def train_ada_boost_classifier(train_X, train_y, test_X, test_y, params):
	# # Create the base classifier
	abc = AdaBoostClassifier(**params)
	
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


def train_logistic_reg_classifier(train_X, train_y, test_X, test_y, params):
	# If the penalty is 'l1' and the solver is 'lbfgs', set penalty to 'none'
	if params['penalty'] == 'l1' and params['solver'] == 'lbfgs':
		params['penalty'] = 'none'
	# # Create the base classifier
	model = LogisticRegression(**params)
	
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


def train_rand_forest_classifier(train_X, train_y, test_X, test_y, params):
	# # Create the base classifier
	model = RandomForestClassifier(**params)
	
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


def train_support_vec_classifier(train_X, train_y, test_X, test_y, params):
	# # Create the base classifier
	model = SVC(**params)
	
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


def train_k_neighbour_classifier(train_X, train_y, test_X, test_y, params):
	# # Create the base classifier
	model = KNeighborsClassifier(**params)
	
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


def train_decision_tree_classifier(train_X, train_y, test_X, test_y, params):
	# # Create the base classifier
	model = DecisionTreeClassifier(**params)
	
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


# def train_reg_cout_(train_X, train_y, test_X, test_y, model1):
# 	model = model1
# 	model.fit(train_X, train_y)

# 	# Predict on test set
# 	test_pred_y = model.predict(test_X)
# 	# Predict on the training set
# 	train_pred_y = model.predict(train_X)

# 	# Calculate evaluation metrics for test set
# 	test_metrics = {
# 		'rmse': np.sqrt(mean_squared_error(test_y, test_pred_y)),
# 		'r2_score': r2_score(test_y, test_pred_y),
# 		'mae': mean_absolute_error(test_y, test_pred_y),
# 		# 'msle': mean_squared_log_error(test_y, test_pred_y),
# 		'mape': np.mean(np.abs((test_y - test_pred_y) / test_y)) * 100
# 	}

# 	# Calculate evaluation metrics for training set
# 	train_metrics = {
# 		'rmse': np.sqrt(mean_squared_error(train_y, train_pred_y)),
# 		'r2_score': r2_score(train_y, train_pred_y),
# 		'mae': mean_absolute_error(train_y, train_pred_y),
# 		# 'msle': mean_squared_log_error(train_y, train_pred_y),
# 		'mape': np.mean(np.abs((train_y - train_pred_y) / train_y)) * 100
# 	}

# 	return model, train_metrics, test_metrics, test_pred_y, train_pred_y

def train_reg_cout_(train_X, train_y, test_X, test_y, model1):
    # Make copies of the input data
    train_X_copy = deepcopy(train_X)
    train_y_copy = deepcopy(train_y)
    test_X_copy = deepcopy(test_X)
    test_y_copy = deepcopy(test_y)

    model = model1
    model.fit(train_X_copy, train_y_copy)

    # Predict on test set
    test_pred_y = model.predict(test_X_copy)
    # Predict on the training set
    train_pred_y = model.predict(train_X_copy)

    # Calculate evaluation metrics for test set
    test_metrics = {
        'rmse': np.sqrt(mean_squared_error(test_y_copy, test_pred_y)),
        'r2_score': r2_score(test_y_copy, test_pred_y),
        'mae': mean_absolute_error(test_y_copy, test_pred_y),
        'mape': np.mean(np.abs((test_y_copy - test_pred_y) / test_y_copy)) * 100
    }

    # Calculate evaluation metrics for training set
    train_metrics = {
        'rmse': np.sqrt(mean_squared_error(train_y_copy, train_pred_y)),
        'r2_score': r2_score(train_y_copy, train_pred_y),
        'mae': mean_absolute_error(train_y_copy, train_pred_y),
        'mape': np.mean(np.abs((train_y_copy - train_pred_y) / train_y_copy)) * 100
    }

    return model, train_metrics, test_metrics, test_pred_y, train_pred_y


# def train_reg_binary_(train_X, train_y, test_X, test_y, threshold, model1):
# 	model = model1

# 	model.fit(train_X, train_y)

# 	# Predict on test set
# 	test_pred_y = model.predict(test_X)
# 	test_pred_y = (test_pred_y >= threshold).astype(int)
# 	# Predict on the training set
# 	train_pred_y = model.predict(train_X)
# 	train_pred_y = (train_pred_y >= threshold).astype(int)
	
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
	
# 	return model, train_metrics, test_metrics, test_pred_y, train_pred_y


def train_reg_binary_(train_X, train_y, test_X, test_y, threshold, model1):
    # Make copies of the input data
    train_X_copy = deepcopy(train_X)
    train_y_copy = deepcopy(train_y)
    test_X_copy = deepcopy(test_X)
    test_y_copy = deepcopy(test_y)
    
    model = model1

    model.fit(train_X_copy, train_y_copy)

    # Predict on test set
    test_pred_y = model.predict(test_X_copy)
    test_pred_y = (test_pred_y >= threshold).astype(int)
    # Predict on the training set
    train_pred_y = model.predict(train_X_copy)
    train_pred_y = (train_pred_y >= threshold).astype(int)
    
    # Calculate evaluation metrics for test set
    test_metrics = {
        'accuracy': float("{:.3f}".format(accuracy_score(test_y_copy, test_pred_y))),
        'roc_auc': float("{:.3f}".format(roc_auc_score(test_y_copy, test_pred_y))),
        'f1_score': float("{:.3f}".format(f1_score(test_y_copy, test_pred_y, average='weighted'))),
        'recall': float("{:.3f}".format(recall_score(test_y_copy, test_pred_y, average='weighted'))),
        'precision': float("{:.3f}".format(precision_score(test_y_copy, test_pred_y, average='weighted'))),
        'rmse': float("{:.3f}".format(np.sqrt(mean_squared_error(test_y_copy, test_pred_y)))),
        'r2_score': float("{:.3f}".format(r2_score(test_y_copy, test_pred_y)))
    }
    
    # Calculate evaluation metrics for training set
    train_metrics = {
        'accuracy': float("{:.3f}".format(accuracy_score(train_y_copy, train_pred_y))),
        'roc_auc': float("{:.3f}".format(roc_auc_score(train_y_copy, train_pred_y))),
        'f1_score': float("{:.3f}".format(f1_score(train_y_copy, train_pred_y, average='weighted'))),
        'recall': float("{:.3f}".format(recall_score(train_y_copy, train_pred_y, average='weighted'))),
        'precision': float("{:.3f}".format(precision_score(train_y_copy, train_pred_y, average='weighted'))),
        'rmse': float("{:.3f}".format(np.sqrt(mean_squared_error(train_y_copy, train_pred_y)))),
        'r2_score': float("{:.3f}".format(r2_score(train_y_copy, train_pred_y)))
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
		'f1_score': f1_score(train_y, train_pred_y, average='weighted'),
		# 'confusion_matrix': confusion_matrix(test_y, test_pred_y)
	}

	return model, train_metrics, test_metrics, test_pred_y, train_pred_y


# Function to train KMeans model
def train_kmeans_model(data, params):
	model = KMeans(**params)
	model.fit(data)
	
	# Calculate silhouette score (if applicable)
	if len(set(model.labels_)) > 1:
		silhouette = silhouette_score(data, model.labels_)
	else:
		silhouette = None
	
	# Calculate Davies-Bouldin score (if applicable)
	if len(set(model.labels_)) > 1:
		davies_bouldin = davies_bouldin_score(data, model.labels_)
	else:
		davies_bouldin = None
	
	# Calculate additional evaluation metrics
	calinski_harabasz = calinski_harabasz_score(data, model.labels_) if 'n_clusters' in params else None
	
	# Format evaluation metrics
	metrics = {
		'silhouette': float("{:.3f}".format(silhouette)) if silhouette is not None else None,
		'davies_bouldin': float("{:.3f}".format(davies_bouldin)) if davies_bouldin is not None else None,
		'calinski_harabasz': float("{:.3f}".format(calinski_harabasz)) if calinski_harabasz is not None else None,
	}
	
	return model, metrics


def train_birch_model(data, params):
	model = Birch(**params)
	
	# Fit the clustering model
	model.fit(data)
	
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
	if params.get('n_clusters') is not None:
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


# Function to train DBSCAN model
def train_dbscan_model(data, eps, min_samples):
	model = DBSCAN(eps=eps, min_samples=min_samples)
	
	# Fit the clustering model
	model.fit(data)
	
	# Calculate silhouette score (if applicable)
	if len(set(model.labels_)) > 1:
		silhouette = silhouette_score(data, model.labels_)
	else:
		silhouette = None
	
	# Calculate Davies-Bouldin score (if applicable)
	if len(set(model.labels_)) > 1:
		davies_bouldin = davies_bouldin_score(data, model.labels_)
	else:
		davies_bouldin = None
	
	# Calculate additional evaluation metrics
	if len(set(model.labels_)) > 1:
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

# Function to train HDBSCAN model
def train_hdbscan_model(data, min_samples, min_cluster_size, cluster_selection_epsilon):
	params = {
		'min_samples': min_samples,
		'min_cluster_size': min_cluster_size,
		'cluster_selection_epsilon': cluster_selection_epsilon
	}
	
	model = HDBSCAN(**params)
	model.fit(data)
	
	# Calculate silhouette score
	if len(set(model.labels_)) > 1:
		silhouette = silhouette_score(data, model.labels_)
	else:
		silhouette = None
	
	# Calculate Davies-Bouldin score
	if len(set(model.labels_)) > 1:
		davies_bouldin = davies_bouldin_score(data, model.labels_)
	else:
		davies_bouldin = None
	
	# Calculate Calinski-Harabasz score
	if len(set(model.labels_)) > 1:
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


# Function to train Agglomerative Clustering model
def train_agglomerative_clus_model(data, params):
	model = AgglomerativeClustering(**params)
	
	# Fit the clustering model
	model.fit(data)
	
	# Calculate silhouette score (if applicable)
	if len(set(model.labels_)) > 1:
		silhouette = silhouette_score(data, model.labels_)
	else:
		silhouette = None
	
	# Calculate Davies-Bouldin score (if applicable)
	if len(set(model.labels_)) > 1:
		davies_bouldin = davies_bouldin_score(data, model.labels_)
	else:
		davies_bouldin = None
	
	# Calculate additional evaluation metrics
	calinski_harabasz = calinski_harabasz_score(data, model.labels_)
	
	# Format evaluation metrics
	metrics = {
		'silhouette': float("{:.3f}".format(silhouette)) if silhouette is not None else None,
		'davies_bouldin': float("{:.3f}".format(davies_bouldin)) if davies_bouldin is not None else None,
		'calinski_harabasz': float("{:.3f}".format(calinski_harabasz)) if calinski_harabasz is not None else None,
	}
	
	return model, metrics

# Function to train MeanShift model
def train_mean_shift_model(data, bandwidth):
	# Define the model with bandwidth parameter
	model = MeanShift(bandwidth=bandwidth)
	
	# Fit the clustering model
	model.fit(data)
	
	# Calculate silhouette score (if applicable)
	if len(set(model.labels_)) > 1:
		silhouette = silhouette_score(data, model.labels_)
	else:
		silhouette = None
	
	# Calculate Davies-Bouldin score (if applicable)
	if len(set(model.labels_)) > 1:
		davies_bouldin = davies_bouldin_score(data, model.labels_)
	else:
		davies_bouldin = None
	
	# Calculate additional evaluation metrics
	if len(set(model.labels_)) > 1:
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


# Function to train AffinityPropagation and calculate metrics
def train_affinity_propo_model(data, params):
	model = AffinityPropagation(**params)
	model.fit(data)

	# Calculate silhouette score
	silhouette = silhouette_score(data, model.labels_)

	# Calculate Davies-Bouldin score
	davies_bouldin = davies_bouldin_score(data, model.labels_)

	# Calculate Calinski-Harabasz score
	calinski_harabasz = calinski_harabasz_score(data, model.labels_)

	# Format evaluation metrics
	metrics = {
		'silhouette': float("{:.3f}".format(silhouette)),
		'davies_bouldin': float("{:.3f}".format(davies_bouldin)),
		'calinski_harabasz': float("{:.3f}".format(calinski_harabasz))
	}

	return model, metrics


# Function to train SpectralClustering model with given hyperparameters
def train_spectral_clust_model(data, params):
	model = SpectralClustering(**params)

	# Fit the clustering model
	model.fit(data)
	
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

	# Calculate Calinski-Harabasz score (if applicable)
	if hasattr(model, 'labels_') and len(set(model.labels_)) > 1:
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


# Function to train OPTICS model
def train_optics_model(data, params):
	model = OPTICS(**params)

	# Fit the clustering model
	model.fit(data)

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
	if len(set(model.labels_)) > 1:
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

# Function to train MiniBatchKMeans and calculate evaluation metrics
def train_mini_batch_kmeans_model(data, params):
	model = MiniBatchKMeans(**params)
	
	# Fit the clustering model
	model.fit(data)
	
	# Check if we can calculate silhouette score
	if hasattr(model, 'labels_') and len(set(model.labels_)) > 1:
		silhouette = silhouette_score(data, model.labels_)
	else:
		silhouette = None
	
	# Check if we can calculate Davies-Bouldin score
	if hasattr(model, 'labels_') and len(set(model.labels_)) > 1:
		davies_bouldin = davies_bouldin_score(data, model.labels_)
	else:
		davies_bouldin = None
	
	# Check if we can calculate Calinski-Harabasz score
	calinski_harabasz = calinski_harabasz_score(data, model.labels_)
	
	# Format evaluation metrics
	metrics = {
		'silhouette': float("{:.3f}".format(silhouette)) if silhouette is not None else None,
		'davies_bouldin': float("{:.3f}".format(davies_bouldin)) if davies_bouldin is not None else None,
		'calinski_harabasz': float("{:.3f}".format(calinski_harabasz)) if calinski_harabasz is not None else None,
	}
	
	return model, metrics

# Function to train FeatureAgglomeration model with custom hyperparameters
def train_feature_agglo_model(data, params):
	model = FeatureAgglomeration(**params)

	# Fit the clustering model
	model.fit(data)
	
	# Calculate silhouette score
	silhouette = None
	if hasattr(model, 'labels_') and len(set(model.labels_)) > 1:
		silhouette = silhouette_score(data, model.labels_)

	# Calculate Davies-Bouldin score
	davies_bouldin = None
	if hasattr(model, 'labels_') and len(set(model.labels_)) > 1:
		davies_bouldin_score(data, model.labels_)

	# Calculate Calinski-Harabasz score
	calinski_harabasz = None
	if hasattr(model, 'labels_') and len(set(model.labels_)) > 1:
		calinski_harabasz = calinski_harabasz_score(data, model.labels_)

	# Format evaluation metrics
	metrics = {
		'silhouette': float("{:.3f}".format(silhouette)) if silhouette is not None else None,
		'davies_bouldin': float("{:.3f}".format(davies_bouldin)) if davies_bouldin is not None else None,
		'calinski_harabasz': float("{:.3f}".format(calinski_harabasz)) if calinski_harabasz is not None else None,
	}
	
	return model, metrics


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




st.title("Custom Model Building")

st.write("Session State:->", st.session_state["shared"])
if "X_train" in st.session_state or "X_test" in st.session_state or "y_train" in st.session_state or "y_test" in st.session_state or "df" in st.session_state:
	# df = st.session_state.df
	df_to_pre = st.session_state.df_pre
	# df = df_to_pre
	df, validation_df = train_test_split(df_to_pre, test_size=0.1, random_state=42)
	st.session_state.uns_valid = validation_df
	max_clusters = len(df) - 1
	try:
		X_train = st.session_state.X_train
		X_test = st.session_state.X_test
		y_train = st.session_state.y_train
		y_test = st.session_state.y_test
		X_test.replace({True: 1, False: 0}, inplace=True)
		X_train.replace({True: 1, False: 0}, inplace=True)
	except:
		pass

else:
	st.error('This is an error', icon="🚨")
	st.warning('Check if you have done proper preprocessing and data spliting or not!')
	
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
	"Decision Tree Regressor": DecisionTreeRegressor(),
	"Birch": Birch(),
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
}

methods = {
	"PCA": PCA(),
	"NMF": NMF(),
	"FastICA": FastICA(),
	"FactorAnalysis": FactorAnalysis(),
	"DictionaryLearning": DictionaryLearning(),
	"TruncatedSVD": TruncatedSVD()
}

selected_type = st.sidebar.radio("Select Model Type", ["Select Model Type", "Regression Models", "Regression Models (continuous)", "Classification Models", "Multi Class Classification", "Clustering", "Clustering by Decomposition"], captions=["choice ⤵️", "Binary Output Variable", "Continuous Output Variable", "Binary Output Variable", "Output Variable with more than 2 class", "Normal Clustering", "First Decomposition then Clustering"])

# option = st.sidebar.selectbox(
# 		"Choose a learning type:",
# 		("Supervised", "Unsupervised")
# 	)

# if option == "Supervised":
	# st.write("You selected Supervised Learning.")
	# Add your supervised learning code here

try:
	if selected_type == "Select Model Type":
		st.title("👈Select Model Type in Sidebar")
	elif selected_type == "Regression Models":
		# UI to select threshold
		threshold = st.slider('Threshold for Binary Prediction', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
		selected_models = st.selectbox("Select models to train", [key for key, model in models.items() if isinstance(model, (LinearRegression, XGBRegressor, GradientBoostingRegressor, RandomForestRegressor, SVR, KNeighborsRegressor, DecisionTreeRegressor))])
	elif selected_type == "Regression Models (continuous)":
		threshold = st.slider('Threshold for Binary Prediction', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
		selected_models = st.selectbox("Select models to train", [key for key, model in models.items() if isinstance(model, (LinearRegression, XGBRegressor, GradientBoostingRegressor, RandomForestRegressor, SVR, KNeighborsRegressor, DecisionTreeRegressor))])
	elif selected_type == "Classification Models":  # Classification Models
		selected_models = st.selectbox("Select models to train", [key for key, model in models.items() if isinstance(model, (LogisticRegression, GradientBoostingClassifier, BaggingClassifier, MultinomialNB, AdaBoostClassifier, RandomForestClassifier, SVC, KNeighborsClassifier, DecisionTreeClassifier, XGBClassifier))])
	elif selected_type == "Multi Class Classification":
		selected_models = st.selectbox("Select models to train", [key for key, model in models.items() if isinstance(model, (LogisticRegression, GradientBoostingClassifier, BaggingClassifier, MultinomialNB, AdaBoostClassifier, RandomForestClassifier, SVC, KNeighborsClassifier, DecisionTreeClassifier, XGBClassifier))])


	elif selected_type == "Clustering":
		# Text input for selecting the number of clusters
		# n_clusters = st.number_input('Enter the number of clusters:', min_value=2, max_value=int(max_clusters), value=2, step=1)
		selected_models = st.selectbox("Select models to train", [key for key, model in models.items() if isinstance(model, (KMeans, DBSCAN, HDBSCAN, AgglomerativeClustering, MeanShift, Birch, AffinityPropagation, SpectralClustering, OPTICS, MiniBatchKMeans, FeatureAgglomeration))])
	elif selected_type == "Clustering by Decomposition":
		# Slider for selecting the number of components
		n_comp = st.slider('Select the number of Decomposition components:', min_value=1, max_value=int(min(df.shape)), value=2)
		# Text input for selecting the number of clusters
		n_clusters = st.number_input('Enter the number of clusters:', min_value=2, max_value=int(max_clusters), value=2, step=1)
		selected_method = st.selectbox("Select Decomposition method", list(methods.keys()))
		selected_models = st.selectbox("Select Clustering models to train", [key for key, model in models.items() if isinstance(model, (KMeans, DBSCAN, HDBSCAN, AgglomerativeClustering, MeanShift, Birch, AffinityPropagation, SpectralClustering, OPTICS, MiniBatchKMeans, FeatureAgglomeration))])
		model_name = selected_method
	# all_metrics_data = []

	if selected_models == "Extreme Gradient Boosting Regressor":
		print("")
		st.header("XGBoostRegressor Hyperparameters")
		_model = models["Extreme Gradient Boosting Regressor"]
		st.write(_model)
		n_estimators = st.slider("Number of Estimators", min_value=100, max_value=1000, value=100, step=10)
		learning_rate = st.slider("Learning Rate", min_value=0.01, max_value=0.3, value=0.1, step=0.01)
		max_depth = st.slider("Max Depth", min_value=3, max_value=10, value=3, step=1)
		min_child_weight = st.slider("Min Child Weight", min_value=1, max_value=20, value=1, step=1)
		subsample = st.slider("Subsample", min_value=0.5, max_value=0.9, value=0.8, step=0.1)
		colsample_bytree = st.slider("Colsample Bytree", min_value=0.5, max_value=0.9, value=0.8, step=0.1)
		gamma = st.slider("Gamma", min_value=0.0, max_value=0.5, value=0.0, step=0.01)
		reg_alpha = st.slider("Reg Alpha", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
		reg_lambda = st.slider("Reg Lambda", min_value=0.0, max_value=1.0, value=0.0, step=0.01)

		# Train the model with the selected hyperparameters
		params = {
			'n_estimators': n_estimators,
			'learning_rate': learning_rate,
			'max_depth': max_depth,
			'min_child_weight': min_child_weight,
			'subsample': subsample,
			'colsample_bytree': colsample_bytree,
			'gamma': gamma,
			'reg_alpha': reg_alpha,
			'reg_lambda': reg_lambda
		}
		_model = _model.set_params(**params)
		if selected_type == "Regression Models":
			model, train_metrics, test_metrics, test_pred_y, train_pred_y = train_reg_binary_(X_train, y_train, X_test, y_test, threshold, _model)
		else:
			model, train_metrics, test_metrics, test_pred_y, train_pred_y = train_reg_cout_(X_train, y_train, X_test, y_test, _model)			
		st.write("HiI")
		col1, col2 = st.columns(2)

		with col1:
			st.write(":orange[Test Metrics]")
			st.write(test_metrics)

		with col2:
			st.write(":orange[Train Metrics]")
			st.write(train_metrics)
		if st.button("Register Model"):
			model_info = {
				'model': model,  # The trained model
				'train_metrics': train_metrics,  # Metrics from the training set
				'test_metrics': test_metrics,  # Metrics from the test set
				'model_name': selected_models,  # Name of the model
			}

			# Dump to a joblib file
			model_filename = f"{selected_models}_model_registered.pkl"
			model_filename = model_filename.replace(" ", "")
			# Save the dictionary containing the model and its metrics
			with open(model_filename, "wb") as f:
				joblib.dump(model_info, f)
			st.success("Model Registered Successfully ✅")

	elif selected_models == "Linear Regression":
		# print()
		st.header("Linear Regression Hyperparameters")
		_model = models["Linear Regression"]
		st.write(_model)
		fit_intercept = st.radio("Fit Intercept", [True, False], index=0)
		copy_X = st.radio("Copy X", [True, False], index=1)

		# Train the model with the selected hyperparameters
		params = {
			'fit_intercept': fit_intercept,
			'copy_X': copy_X
		}
		# print(**params)
		st.write(params)
		# model, train_metrics, test_metrics, test_pred_y, train_pred_y = train_linear_regression(X_train, y_train, X_test, y_test,params)
		_model = _model.set_params(**params)
		if selected_type == "Regression Models":
			model, train_metrics, test_metrics, test_pred_y, train_pred_y = train_reg_binary_(X_train, y_train, X_test, y_test, threshold, _model)
		else:
			model, train_metrics, test_metrics, test_pred_y, train_pred_y = train_reg_cout_(X_train, y_train, X_test, y_test, _model)
		st.write("HiI")
		col1, col2 = st.columns(2)

		with col1:
			st.write(":orange[Test Metrics]")
			st.write(test_metrics)

		with col2:
			st.write(":orange[Train Metrics]")
			st.write(train_metrics)

		if st.button("Register Model"):
			model_info = {
				'model': model,  # The trained model
				'train_metrics': train_metrics,  # Metrics from the training set
				'test_metrics': test_metrics,  # Metrics from the test set
				'model_name': selected_models,  # Name of the model
			}

			# Dump to a joblib file
			model_filename = f"{selected_models}_model_registered.pkl"
			model_filename = model_filename.replace(" ", "")
			# Save the dictionary containing the model and its metrics
			with open(model_filename, "wb") as f:
				joblib.dump(model_info, f)
			st.success("Model Registered Successfully ✅")

	elif selected_models == "Gradient Boosting Regressor":
		# st.title("XGBoostRegressor Hyperparameter Tuning")
		_model = models["Gradient Boosting Regressor"]
		# Sidebar for hyperparameter input
		# Sidebar for hyperparameter input
		st.header("GradientBoostingRegressor Hyperparameters")
		n_estimators = st.slider("Number of Estimators", min_value=100, max_value=1000, value=100, step=10)
		learning_rate = st.slider("Learning Rate", min_value=0.01, max_value=0.3, value=0.1, step=0.01)
		max_depth = st.slider("Max Depth", min_value=3, max_value=10, value=3, step=1)
		min_samples_split = st.slider("Min Samples Split", min_value=2, max_value=20, value=2, step=1)
		min_samples_leaf = st.slider("Min Samples Leaf", min_value=1, max_value=10, value=1, step=1)
		subsample = st.slider("Subsample", min_value=0.5, max_value=1.0, value=1.0, step=0.1)
		max_features = st.select_slider("Max Features", options=["sqrt", "log2", None])

		# Train the model with the selected hyperparameters
		params = {
			'n_estimators': n_estimators,
			'learning_rate': learning_rate,
			'max_depth': max_depth,
			'min_samples_split': min_samples_split,
			'min_samples_leaf': min_samples_leaf,
			'subsample': subsample,
			'max_features': max_features
		}

		# model, train_metrics, test_metrics, test_pred_y, train_pred_y = train_gradient_boosting_regressor(X_train, y_train, X_test, y_test,params)
		_model = _model.set_params(**params)
		if selected_type == "Regression Models":
			model, train_metrics, test_metrics, test_pred_y, train_pred_y = train_reg_binary_(X_train, y_train, X_test, y_test, threshold, _model)
		else:
			model, train_metrics, test_metrics, test_pred_y, train_pred_y = train_reg_cout_(X_train, y_train, X_test, y_test, _model)

		st.write("HiI")
		col1, col2 = st.columns(2)

		with col1:
			st.write(":orange[Test Metrics]")
			st.write(test_metrics)

		with col2:
			st.write(":orange[Train Metrics]")
			st.write(train_metrics)
		if st.button("Register Model"):
			model_info = {
				'model': model,  # The trained model
				'train_metrics': train_metrics,  # Metrics from the training set
				'test_metrics': test_metrics,  # Metrics from the test set
				'model_name': selected_models,  # Name of the model
			}

			# Dump to a joblib file
			model_filename = f"{selected_models}_model_registered.pkl"
			model_filename = model_filename.replace(" ", "")
			# Save the dictionary containing the model and its metrics
			with open(model_filename, "wb") as f:
				joblib.dump(model_info, f)
			st.success("Model Registered Successfully ✅")
	
	elif selected_models == "Random Forest Regressor":
		print("train_random_forest__regressor")
		# Sidebar for hyperparameter input
		_model = models["Random Forest Regressor"]
		st.header("Random Forest Regressor Hyperparameters")
		n_estimators = st.slider("Number of Estimators", min_value=100, max_value=1000, value=100, step=10)
		max_depth = st.slider("Max Depth", min_value=1, max_value=50, value=None, step=1, key='max_depth')
		min_samples_split = st.slider("Min Samples Split", min_value=2, max_value=20, value=2, step=1, key='min_samples_split')
		min_samples_leaf = st.slider("Min Samples Leaf", min_value=1, max_value=20, value=1, step=1, key='min_samples_leaf')
		max_features = st.selectbox("Max Features", options=["sqrt", "log2"], index=0)

		# Train the model with the selected hyperparameters
		params = {
			'n_estimators': n_estimators,
			'max_depth': max_depth,
			'min_samples_split': min_samples_split,
			'min_samples_leaf': min_samples_leaf,
			'max_features': max_features
		}
		
		# model, train_metrics, test_metrics, test_pred_y, train_pred_y = train_random_forest__regressor(X_train, y_train, X_test, y_test,params)
		_model = _model.set_params(**params)
		if selected_type == "Regression Models":
			model, train_metrics, test_metrics, test_pred_y, train_pred_y = train_reg_binary_(X_train, y_train, X_test, y_test, threshold, _model)
		else:
			model, train_metrics, test_metrics, test_pred_y, train_pred_y = train_reg_cout_(X_train, y_train, X_test, y_test, _model)

		st.write("HiI")
		col1, col2 = st.columns(2)

		with col1:
			st.write(":orange[Test Metrics]")
			st.write(test_metrics)

		with col2:
			st.write(":orange[Train Metrics]")
			st.write(train_metrics)
		if st.button("Register Model"):
			model_info = {
				'model': model,  # The trained model
				'train_metrics': train_metrics,  # Metrics from the training set
				'test_metrics': test_metrics,  # Metrics from the test set
				'model_name': selected_models,  # Name of the model
			}

			# Dump to a joblib file
			model_filename = f"{selected_models}_model_registered.pkl"
			model_filename = model_filename.replace(" ", "")
			# Save the dictionary containing the model and its metrics
			with open(model_filename, "wb") as f:
				joblib.dump(model_info, f)
			st.success("Model Registered Successfully ✅")

	elif selected_models == "Support Vector Regressor":
		print("train_support_vector_regressor")
		_model = models["Support Vector Regressor"]
		# Sidebar for hyperparameter input
		st.header("Support Vector Regressor Hyperparameters")
		C = st.slider("C (Regularization Parameter)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
		epsilon = st.slider("Epsilon", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
		kernel = st.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"])

		# Train the model with the selected hyperparameters
		params = {
			'C': C,
			'epsilon': epsilon,
			'kernel': kernel
		}
		
		# model, train_metrics, test_metrics, test_pred_y, train_pred_y = train_support_vector_regressor(X_train, y_train, X_test, y_test,params)
		_model = _model.set_params(**params)
		if selected_type == "Regression Models":
			model, train_metrics, test_metrics, test_pred_y, train_pred_y = train_reg_binary_(X_train, y_train, X_test, y_test, threshold, _model)
		else:
			model, train_metrics, test_metrics, test_pred_y, train_pred_y = train_reg_cout_(X_train, y_train, X_test, y_test, _model)

		st.write("HiI")
		col1, col2 = st.columns(2)

		with col1:
			st.write(":orange[Test Metrics]")
			st.write(test_metrics)

		with col2:
			st.write(":orange[Train Metrics]")
			st.write(train_metrics)
		if st.button("Register Model"):
			model_info = {
				'model': model,  # The trained model
				'train_metrics': train_metrics,  # Metrics from the training set
				'test_metrics': test_metrics,  # Metrics from the test set
				'model_name': selected_models,  # Name of the model
			}

			# Dump to a joblib file
			model_filename = f"{selected_models}_model_registered.pkl"
			model_filename = model_filename.replace(" ", "")
			# Save the dictionary containing the model and its metrics
			with open(model_filename, "wb") as f:
				joblib.dump(model_info, f)
			st.success("Model Registered Successfully ✅")

	elif selected_models == "K-Neighbors Regressor":
		print("train_k_neighbors_regressor")
		# Sidebar for hyperparameter input
		_model = models["K-Neighbors Regressor"]
		st.header("K Neighbors Regressor Hyperparameters")
		n_neighbors = st.slider("Number of Neighbors", min_value=1, max_value=20, value=5, step=1)
		weights = st.selectbox("Weights", ["uniform", "distance"])
		algorithm = st.selectbox("Algorithm", ["auto", "ball_tree", "kd_tree", "brute"])
		leaf_size = st.slider("Leaf Size", min_value=10, max_value=50, value=30, step=10)
		p = st.slider("p (Power parameter for Minkowski metric)", min_value=1, max_value=5, value=2, step=1)

		# Train the model with the selected hyperparameters
		params = {
			'n_neighbors': n_neighbors,
			'weights': weights,
			'algorithm': algorithm,
			'leaf_size': leaf_size,
			'p': p
		}
		
		# model, train_metrics, test_metrics, test_pred_y, train_pred_y = train_k_neighbors_regressor(X_train, y_train, X_test, y_test,params)
		_model = _model.set_params(**params)
		if selected_type == "Regression Models":
			model, train_metrics, test_metrics, test_pred_y, train_pred_y = train_reg_binary_(X_train, y_train, X_test, y_test, threshold, _model)
		else:
			model, train_metrics, test_metrics, test_pred_y, train_pred_y = train_reg_cout_(X_train, y_train, X_test, y_test, _model)

		st.write("HiI")
		col1, col2 = st.columns(2)

		with col1:
			st.write(":orange[Test Metrics]")
			st.write(test_metrics)

		with col2:
			st.write(":orange[Train Metrics]")
			st.write(train_metrics)
		if st.button("Register Model"):
			model_info = {
				'model': model,  # The trained model
				'train_metrics': train_metrics,  # Metrics from the training set
				'test_metrics': test_metrics,  # Metrics from the test set
				'model_name': selected_models,  # Name of the model
			}

			# Dump to a joblib file
			model_filename = f"{selected_models}_model_registered.pkl"
			model_filename = model_filename.replace(" ", "")
			# Save the dictionary containing the model and its metrics
			with open(model_filename, "wb") as f:
				joblib.dump(model_info, f)
			st.success("Model Registered Successfully ✅")

	elif selected_models == "Decision Tree Regressor":
		print("train_decision_tree_regressor")
		_model = models["Decision Tree Regressor"]
		# Sidebar for hyperparameter input
		st.header("DecisionTreeRegressor Hyperparameters")
		max_depth = st.slider("Max Depth", min_value=1, max_value=20, value=5, step=1)
		min_samples_split = st.slider("Min Samples Split", min_value=2, max_value=20, value=2, step=1)
		min_samples_leaf = st.slider("Min Samples Leaf", min_value=1, max_value=20, value=1, step=1)
		max_features = st.select_slider("Max Features", options=["auto", "sqrt", "log2", None])

		# Train the model with the selected hyperparameters
		params = {
			'max_depth': max_depth,
			'min_samples_split': min_samples_split,
			'min_samples_leaf': min_samples_leaf,
			'max_features': max_features
		}

		# model, train_metrics, test_metrics, test_pred_y, train_pred_y = train_decision_tree_regressor(X_train, y_train, X_test, y_test,params)
		_model = _model.set_params(**params)
		if selected_type == "Regression Models":
			model, train_metrics, test_metrics, test_pred_y, train_pred_y = train_reg_binary_(X_train, y_train, X_test, y_test, threshold, _model)
		else:
			model, train_metrics, test_metrics, test_pred_y, train_pred_y = train_reg_cout_(X_train, y_train, X_test, y_test, _model)

		st.write("HiI")
		col1, col2 = st.columns(2)

		with col1:
			st.write(":orange[Test Metrics]")
			st.write(test_metrics)

		with col2:
			st.write(":orange[Train Metrics]")
			st.write(train_metrics)		
		
		if st.button("Register Model"):
			model_info = {
				'model': model,  # The trained model
				'train_metrics': train_metrics,  # Metrics from the training set
				'test_metrics': test_metrics,  # Metrics from the test set
				'model_name': selected_models,  # Name of the model
			}

			# Dump to a joblib file
			model_filename = f"{selected_models}_model_registered.pkl"
			model_filename = model_filename.replace(" ", "")
			# Save the dictionary containing the model and its metrics
			with open(model_filename, "wb") as f:
				joblib.dump(model_info, f)
			st.success("Model Registered Successfully ✅")

	elif selected_models == "Extreme Gradient Boosting Classifier":
		print("train_xtreme_gradient_boosting_classifier")
		_model = models["Extreme Gradient Boosting Classifier"]
		# Sidebar for hyperparameter input
		st.header("XGBoostClassifier Hyperparameters")
		max_depth = st.slider("Max Depth", min_value=3, max_value=10, value=3, step=1)
		learning_rate = st.slider("Learning Rate", min_value=0.01, max_value=0.3, value=0.1, step=0.01)
		n_estimators = st.slider("Number of Estimators", min_value=100, max_value=1000, value=100, step=10)
		subsample = st.slider("Subsample", min_value=0.5, max_value=0.9, value=0.8, step=0.1)
		colsample_bytree = st.slider("Colsample Bytree", min_value=0.5, max_value=0.9, value=0.8, step=0.1)
		gamma = st.slider("Gamma", min_value=0.0, max_value=0.5, value=0.0, step=0.01)
		reg_alpha = st.slider("Reg Alpha", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
		reg_lambda = st.slider("Reg Lambda", min_value=0.0, max_value=1.0, value=0.0, step=0.01)

		# Train the model with the selected hyperparameters
		params = {
			'max_depth': max_depth,
			'learning_rate': learning_rate,
			'n_estimators': n_estimators,
			'subsample': subsample,
			'colsample_bytree': colsample_bytree,
			'gamma': gamma,
			'reg_alpha': reg_alpha,
			'reg_lambda': reg_lambda,
			'objective': 'binary:logistic'  # for classification
		}

		# model, train_metrics, test_metrics, test_pred_y, train_pred_y = train_xtreme_gradient_boosting_classifier(X_train, y_train, X_test, y_test, params)
		_model = _model.set_params(**params)
		if selected_type == "Classification Models":
			model, train_metrics, test_metrics, test_pred_y, train_pred_y = train_cls_binary_(X_train, y_train, X_test, y_test, _model)
		else:
			model, train_metrics, test_metrics, test_pred_y, train_pred_y = train_cls_mcc_(X_train, y_train, X_test, y_test, _model)

		st.write("HiI")
		col1, col2 = st.columns(2)

		with col1:
			st.write(":orange[Test Metrics]")
			st.write(test_metrics)

		with col2:
			st.write(":orange[Train Metrics]")
			st.write(train_metrics)
		if st.button("Register Model"):
			model_info = {
				'model': model,  # The trained model
				'train_metrics': train_metrics,  # Metrics from the training set
				'test_metrics': test_metrics,  # Metrics from the test set
				'model_name': selected_models,  # Name of the model
			}

			# Dump to a joblib file
			model_filename = f"{selected_models}_model_registered.pkl"
			model_filename = model_filename.replace(" ", "")
			# Save the dictionary containing the model and its metrics
			with open(model_filename, "wb") as f:
				joblib.dump(model_info, f)
			st.success("Model Registered Successfully ✅")

	elif selected_models == "Gradient Boosting Classifier":
		print("train_gradient_boosting_classifier")
		# Sidebar for hyperparameter input
		_model = models["Gradient Boosting Classifier"]
		st.header("GradientBoostingClassifier Hyperparameters")
		n_estimators = st.slider("Number of Estimators", min_value=50, max_value=500, value=100, step=10)
		learning_rate = st.slider("Learning Rate", min_value=0.01, max_value=0.3, value=0.1, step=0.01)
		max_depth = st.slider("Max Depth", min_value=3, max_value=10, value=3, step=1)
		min_samples_split = st.slider("Min Samples Split", min_value=2, max_value=10, value=2, step=1)
		min_samples_leaf = st.slider("Min Samples Leaf", min_value=1, max_value=10, value=1, step=1)
		subsample = st.slider("Subsample", min_value=0.5, max_value=1.0, value=1.0, step=0.1)
		max_features = st.selectbox("Max Features", options=[None, 'sqrt', 'log2'])
		random_state = st.number_input("Random State", value=42)

		# Train the model with the selected hyperparameters
		params = {
			'n_estimators': n_estimators,
			'learning_rate': learning_rate,
			'max_depth': max_depth,
			'min_samples_split': min_samples_split,
			'min_samples_leaf': min_samples_leaf,
			'subsample': subsample,
			'max_features': max_features,
			'random_state': random_state
		}

		# model, train_metrics, test_metrics, test_pred_y, train_pred_y = train_gradient_boosting_classifier(X_train, y_train, X_test, y_test, params)
		_model = _model.set_params(**params)
		if selected_type == "Classification Models":
			model, train_metrics, test_metrics, test_pred_y, train_pred_y = train_cls_binary_(X_train, y_train, X_test, y_test, _model)
		else:
			model, train_metrics, test_metrics, test_pred_y, train_pred_y = train_cls_mcc_(X_train, y_train, X_test, y_test, _model)
		st.write("HiI")
		col1, col2 = st.columns(2)

		with col1:
			st.write(":orange[Test Metrics]")
			st.write(test_metrics)

		with col2:
			st.write(":orange[Train Metrics]")
			st.write(train_metrics)
		if st.button("Register Model"):
			model_info = {
				'model': model,  # The trained model
				'train_metrics': train_metrics,  # Metrics from the training set
				'test_metrics': test_metrics,  # Metrics from the test set
				'model_name': selected_models,  # Name of the model
			}

			# Dump to a joblib file
			model_filename = f"{selected_models}_model_registered.pkl"
			model_filename = model_filename.replace(" ", "")
			# Save the dictionary containing the model and its metrics
			with open(model_filename, "wb") as f:
				joblib.dump(model_info, f)
			st.success("Model Registered Successfully ✅")

	elif selected_models == "Bagging Classifier":
		print("train_bagging_classifier")
		_model = models["Bagging Classifier"]
		# Sidebar for hyperparameter input
		st.header("Bagging Classifier Hyperparameters")
		n_estimators = st.slider("Number of Estimators", min_value=10, max_value=100, value=10, step=10)
		max_samples = st.slider("Max Samples", min_value=0.1, max_value=1.0, value=1.0, step=0.1)
		max_features = st.slider("Max Features", min_value=0.1, max_value=1.0, value=1.0, step=0.1)
		bootstrap = st.checkbox("Bootstrap", value=True)
		bootstrap_features = st.checkbox("Bootstrap Features", value=False)

		# Train the model with the selected hyperparameters
		params = {
			'n_estimators': n_estimators,
			'max_samples': max_samples,
			'max_features': max_features,
			'bootstrap': bootstrap,
			'bootstrap_features': bootstrap_features
		}

		# model, train_metrics, test_metrics, test_pred_y, train_pred_y = train_bagging_classifier(X_train, y_train, X_test, y_test, params)
		_model = _model.set_params(**params)
		if selected_type == "Classification Models":
			model, train_metrics, test_metrics, test_pred_y, train_pred_y = train_cls_binary_(X_train, y_train, X_test, y_test, _model)
		else:
			model, train_metrics, test_metrics, test_pred_y, train_pred_y = train_cls_mcc_(X_train, y_train, X_test, y_test, _model)

		st.write("HiI")
		col1, col2 = st.columns(2)

		with col1:
			st.write(":orange[Test Metrics]")
			st.write(test_metrics)

		with col2:
			st.write(":orange[Train Metrics]")
			st.write(train_metrics)
		if st.button("Register Model"):
			model_info = {
				'model': model,  # The trained model
				'train_metrics': train_metrics,  # Metrics from the training set
				'test_metrics': test_metrics,  # Metrics from the test set
				'model_name': selected_models,  # Name of the model
			}

			# Dump to a joblib file
			model_filename = f"{selected_models}_model_registered.pkl"
			model_filename = model_filename.replace(" ", "")
			# Save the dictionary containing the model and its metrics
			with open(model_filename, "wb") as f:
				joblib.dump(model_info, f)
			st.success("Model Registered Successfully ✅")

	# elif selected_models == "Multinomial Naive Bayes":
	# 	print("train_multinomial_naive_bayes")
	# 	_model = models["Multinomial Naive Bayes"]
	# 	# Sidebar for hyperparameter input
	# 	st.header("Multinomial Naive Bayes Hyperparameters")
	# 	alpha = st.slider("Alpha", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
	# 	params = {
	# 		'alpha': alpha
	# 	}
	# 	# model, train_metrics, test_metrics, test_pred_y, train_pred_y = train_multinomial_naive_bayes(X_train, y_train, X_test, y_test, alpha)
	# 	_model = _model.set_params(**params)
	# 	if selected_type == "Classification Models":
	# 		model, train_metrics, test_metrics, test_pred_y, train_pred_y = train_cls_binary_(X_train, y_train, X_test, y_test, _model)
	# 	else:
	# 		model, train_metrics, test_metrics, test_pred_y, train_pred_y = train_cls_mcc_(X_train, y_train, X_test, y_test, _model)

	# 	st.write("HiI")
	# 	col1, col2 = st.columns(2)

	# 	with col1:
	# 		st.write(":orange[Test Metrics]")
	# 		st.write(test_metrics)

	# 	with col2:
	# 		st.write(":orange[Train Metrics]")
	# 		st.write(train_metrics)
	# 	if st.button("Register Model"):
	# 		model_info = {
	# 			'model': model,  # The trained model
	# 			'train_metrics': train_metrics,  # Metrics from the training set
	# 			'test_metrics': test_metrics,  # Metrics from the test set
	# 			'model_name': selected_models,  # Name of the model
	# 		}

	# 		# Dump to a joblib file
	# 		model_filename = f"{selected_models}_model_registered.pkl"
	# 		model_filename = model_filename.replace(" ", "")
	# 		# Save the dictionary containing the model and its metrics
	# 		with open(model_filename, "wb") as f:
	# 			joblib.dump(model_info, f)
	# 		st.success("Model Registered Successfully ✅")
	
	elif selected_models == "Ada Boost Classifier":
		print("train_ada_boost_classifier")
		_model = models["Ada Boost Classifier"]
		st.header("AdaBoostClassifier Hyperparameters")
		n_estimators = st.slider("Number of Estimators", min_value=50, max_value=500, value=50, step=10)
		learning_rate = st.slider("Learning Rate", min_value=0.01, max_value=1.0, value=1.0, step=0.01)

		# Train the model with the selected hyperparameters
		params = {
			'n_estimators': n_estimators,
			'learning_rate': learning_rate
		}
		# model, train_metrics, test_metrics, test_pred_y, train_pred_y = train_ada_boost_classifier(X_train, y_train, X_test, y_test, params)
		_model = _model.set_params(**params)
		if selected_type == "Classification Models":
			model, train_metrics, test_metrics, test_pred_y, train_pred_y = train_cls_binary_(X_train, y_train, X_test, y_test, _model)
		else:
			model, train_metrics, test_metrics, test_pred_y, train_pred_y = train_cls_mcc_(X_train, y_train, X_test, y_test, _model)

		st.write("HiI")
		col1, col2 = st.columns(2)

		with col1:
			st.write(":orange[Test Metrics]")
			st.write(test_metrics)

		with col2:
			st.write(":orange[Train Metrics]")
			st.write(train_metrics)
		if st.button("Register Model"):
			model_info = {
				'model': model,  # The trained model
				'train_metrics': train_metrics,  # Metrics from the training set
				'test_metrics': test_metrics,  # Metrics from the test set
				'model_name': selected_models,  # Name of the model
			}

			# Dump to a joblib file
			model_filename = f"{selected_models}_model_registered.pkl"
			model_filename = model_filename.replace(" ", "")
			# Save the dictionary containing the model and its metrics
			with open(model_filename, "wb") as f:
				joblib.dump(model_info, f)
			st.success("Model Registered Successfully ✅")

	elif selected_models == "Logistic Regression":
		print("train_logistic_reg_classifier")
		_model = models["Logistic Regression"]
		# Sidebar for hyperparameter input
		st.header("Logistic Regression Hyperparameters")
		penalty = st.radio("Penalty", ["l1", "l2"])
		solver = st.radio("Solver", ["lbfgs", "liblinear", "saga"])
		C = st.slider("Inverse of Regularization Strength (C)", min_value=0.01, max_value=10.0, value=1.0, step=0.01)
		max_iter = st.slider("Maximum Number of Iterations", min_value=100, max_value=1000, value=100, step=100)

		# Train the model with the selected hyperparameters
		params = {
			'penalty': penalty,
			'solver': solver,
			'C': C,
			'max_iter': max_iter
		}
		# model, train_metrics, test_metrics, test_pred_y, train_pred_y = train_logistic_reg_classifier(X_train, y_train, X_test, y_test, params)
		_model = _model.set_params(**params)
		if selected_type == "Classification Models":
			model, train_metrics, test_metrics, test_pred_y, train_pred_y = train_cls_binary_(X_train, y_train, X_test, y_test, _model)
		else:
			model, train_metrics, test_metrics, test_pred_y, train_pred_y = train_cls_mcc_(X_train, y_train, X_test, y_test, _model)

		st.write("HiI")
		col1, col2 = st.columns(2)

		with col1:
			st.write(":orange[Test Metrics]")
			st.write(test_metrics)

		with col2:
			st.write(":orange[Train Metrics]")
			st.write(train_metrics)
		if st.button("Register Model"):
			model_info = {
				'model': model,  # The trained model
				'train_metrics': train_metrics,  # Metrics from the training set
				'test_metrics': test_metrics,  # Metrics from the test set
				'model_name': selected_models,  # Name of the model
			}

			# Dump to a joblib file
			model_filename = f"{selected_models}_model_registered.pkl"
			model_filename = model_filename.replace(" ", "")
			# Save the dictionary containing the model and its metrics
			with open(model_filename, "wb") as f:
				joblib.dump(model_info, f)
			st.success("Model Registered Successfully ✅")
		
	elif selected_models == "Random Forest Classifier":
		print("train_rand_forest_classifier")

		_model = models["Random Forest Classifier"]
		# Sidebar for hyperparameter input
		st.header("Random Forest Classifier Hyperparameters")
		n_estimators = st.slider("Number of Estimators", min_value=50, max_value=500, value=100, step=10)
		max_depth = st.slider("Max Depth", min_value=2, max_value=50, value=10, step=1)
		min_samples_split = st.slider("Min Samples Split", min_value=2, max_value=20, value=2, step=1)
		min_samples_leaf = st.slider("Min Samples Leaf", min_value=1, max_value=20, value=1, step=1)
		max_features = st.selectbox("Max Features", ["auto", "sqrt", "log2", None])

		# Convert max_features to int if it's 'auto'
		if max_features == "auto":
			max_features = None

		# Train the model with the selected hyperparameters
		params = {
			'n_estimators': n_estimators,
			'max_depth': max_depth,
			'min_samples_split': min_samples_split,
			'min_samples_leaf': min_samples_leaf,
			'max_features': max_features
		}

		# model, train_metrics, test_metrics, test_pred_y, train_pred_y = train_rand_forest_classifier(X_train, y_train, X_test, y_test, params)
		_model = _model.set_params(**params)
		if selected_type == "Classification Models":
			model, train_metrics, test_metrics, test_pred_y, train_pred_y = train_cls_binary_(X_train, y_train, X_test, y_test, _model)
		else:
			model, train_metrics, test_metrics, test_pred_y, train_pred_y = train_cls_mcc_(X_train, y_train, X_test, y_test, _model)

		st.write("HiI")
		col1, col2 = st.columns(2)

		with col1:
			st.write(":orange[Test Metrics]")
			st.write(test_metrics)

		with col2:
			st.write(":orange[Train Metrics]")
			st.write(train_metrics)
		if st.button("Register Model"):
			model_info = {
				'model': model,  # The trained model
				'train_metrics': train_metrics,  # Metrics from the training set
				'test_metrics': test_metrics,  # Metrics from the test set
				'model_name': selected_models,  # Name of the model
			}

			# Dump to a joblib file
			model_filename = f"{selected_models}_model_registered.pkl"
			model_filename = model_filename.replace(" ", "")
			# Save the dictionary containing the model and its metrics
			with open(model_filename, "wb") as f:
				joblib.dump(model_info, f)
			st.success("Model Registered Successfully ✅")

	elif selected_models == "Support Vector Classifier":
		print("train_support_vec_classifier")
		_model = models["Support Vector Classifier"]
		# Sidebar for hyperparameter input
		st.header("SVC Hyperparameters")
		C = st.slider("Regularization parameter (C)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
		kernel = st.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"], index=2)
		degree = st.slider("Degree (for polynomial kernel)", min_value=1, max_value=10, value=3, step=1)
		gamma = st.slider("Gamma (for rbf, poly, sigmoid kernels)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

		# Train the model with the selected hyperparameters
		params = {
			'C': C,
			'kernel': kernel,
			'degree': degree,
			'gamma': gamma
		}

		# model, train_metrics, test_metrics, test_pred_y, train_pred_y = train_support_vec_classifier(X_train, y_train, X_test, y_test, params)
		_model = _model.set_params(**params)
		if selected_type == "Classification Models":
			model, train_metrics, test_metrics, test_pred_y, train_pred_y = train_cls_binary_(X_train, y_train, X_test, y_test, _model)
		else:
			model, train_metrics, test_metrics, test_pred_y, train_pred_y = train_cls_mcc_(X_train, y_train, X_test, y_test, _model)

		st.write("HiI")
		col1, col2 = st.columns(2)

		with col1:
			st.write(":orange[Test Metrics]")
			st.write(test_metrics)

		with col2:
			st.write(":orange[Train Metrics]")
			st.write(train_metrics)
		if st.button("Register Model"):
			model_info = {
				'model': model,  # The trained model
				'train_metrics': train_metrics,  # Metrics from the training set
				'test_metrics': test_metrics,  # Metrics from the test set
				'model_name': selected_models,  # Name of the model
			}

			# Dump to a joblib file
			model_filename = f"{selected_models}_model_registered.pkl"
			model_filename = model_filename.replace(" ", "")
			# Save the dictionary containing the model and its metrics
			with open(model_filename, "wb") as f:
				joblib.dump(model_info, f)
			st.success("Model Registered Successfully ✅")

	elif selected_models == "K-Neighbors Classifier":
		print("train_k_neighbour_classifier")
		_model = models["K-Neighbors Classifier"]
		# Sidebar for hyperparameter input
		st.header("KNeighborsClassifier Hyperparameters")
		n_neighbors = st.slider("Number of Neighbors", min_value=1, max_value=20, value=5, step=1)
		weights = st.selectbox("Weights", ['uniform', 'distance'])
		algorithm = st.selectbox("Algorithm", ['auto', 'ball_tree', 'kd_tree', 'brute'])
		leaf_size = st.slider("Leaf Size", min_value=10, max_value=50, value=30, step=5)
		p = st.slider("p", min_value=1, max_value=10, value=2, step=1)

		# Train the model with the selected hyperparameters
		params = {
			'n_neighbors': n_neighbors,
			'weights': weights,
			'algorithm': algorithm,
			'leaf_size': leaf_size,
			'p': p
		}
		
		# model, train_metrics, test_metrics, test_pred_y, train_pred_y = train_k_neighbour_classifier(X_train, y_train, X_test, y_test, params)
		_model = _model.set_params(**params)
		if selected_type == "Classification Models":
			model, train_metrics, test_metrics, test_pred_y, train_pred_y = train_cls_binary_(X_train, y_train, X_test, y_test, _model)
		else:
			model, train_metrics, test_metrics, test_pred_y, train_pred_y = train_cls_mcc_(X_train, y_train, X_test, y_test, _model)

		st.write("HiI")
		col1, col2 = st.columns(2)

		with col1:
			st.write(":orange[Test Metrics]")
			st.write(test_metrics)

		with col2:
			st.write(":orange[Train Metrics]")
			st.write(train_metrics)
		if st.button("Register Model"):
			model_info = {
				'model': model,  # The trained model
				'train_metrics': train_metrics,  # Metrics from the training set
				'test_metrics': test_metrics,  # Metrics from the test set
				'model_name': selected_models,  # Name of the model
			}

			# Dump to a joblib file
			model_filename = f"{selected_models}_model_registered.pkl"
			model_filename = model_filename.replace(" ", "")
			# Save the dictionary containing the model and its metrics
			with open(model_filename, "wb") as f:
				joblib.dump(model_info, f)
			st.success("Model Registered Successfully ✅")


	elif selected_models == "Decision Tree Classifier":
		print("train_decision_tree_classifier")
		_model = models["Decision Tree Classifier"]
		# Sidebar for hyperparameter input
		st.header("Decision Tree Classifier Hyperparameters")
		max_depth = st.slider("Max Depth", min_value=1, max_value=20, value=10, step=1)
		criterion = st.selectbox("Criterion", ["gini", "entropy"])
		splitter = st.selectbox("Splitter", ["best", "random"])
		min_samples_split = st.slider("Min Samples Split", min_value=2, max_value=20, value=2, step=1)
		min_samples_leaf = st.slider("Min Samples Leaf", min_value=1, max_value=20, value=1, step=1)
		max_features = st.selectbox("Max Features", ["auto", "sqrt", "log2", None])

		# Convert max_features to appropriate type
		if max_features == "auto":
			max_features = None

		# Train the model with the selected hyperparameters
		params = {
			'max_depth': max_depth,
			'criterion': criterion,
			'splitter': splitter,
			'min_samples_split': min_samples_split,
			'min_samples_leaf': min_samples_leaf,
			'max_features': max_features
		}
		
		# model, train_metrics, test_metrics, test_pred_y, train_pred_y = train_decision_tree_classifier(X_train, y_train, X_test, y_test, params)
		_model = _model.set_params(**params)
		if selected_type == "Classification Models":
			model, train_metrics, test_metrics, test_pred_y, train_pred_y = train_cls_binary_(X_train, y_train, X_test, y_test, _model)
		else:
			model, train_metrics, test_metrics, test_pred_y, train_pred_y = train_cls_mcc_(X_train, y_train, X_test, y_test, _model)

		st.write("HiI")
		col1, col2 = st.columns(2)

		with col1:
			st.write(":orange[Test Metrics]")
			st.write(test_metrics)

		with col2:
			st.write(":orange[Train Metrics]")
			st.write(train_metrics)
		if st.button("Register Model"):
			model_info = {
				'model': model,  # The trained model
				'train_metrics': train_metrics,  # Metrics from the training set
				'test_metrics': test_metrics,  # Metrics from the test set
				'model_name': selected_models,  # Name of the model
			}

			# Dump to a joblib file
			model_filename = f"{selected_models}_model_registered.pkl"
			model_filename = model_filename.replace(" ", "")
			# Save the dictionary containing the model and its metrics
			with open(model_filename, "wb") as f:
				joblib.dump(model_info, f)
			st.success("Model Registered Successfully ✅")

	elif selected_models == "K-Means":
		print("train_decision_tree_classifier")

		# Sidebar for hyperparameter input
		st.header("KMeans Hyperparameters")
		n_clusters = st.slider("Number of Clusters", min_value=2, max_value=20, value=2, step=1)
		init = st.selectbox("Initialization Method", ["k-means++", "random"])
		n_init = st.slider("Number of Initializations", min_value=1, max_value=10, value=5, step=1)
		max_iter = st.slider("Maximum Number of Iterations", min_value=100, max_value=1000, value=300, step=100)
		tol = st.number_input("Tolerance", value=1e-4, format="%.6f", step=1e-6)
		
		# Train the model with the selected hyperparameters
		params = {
			'n_clusters': n_clusters,
			'init': init,
			'n_init': n_init,
			'max_iter': max_iter,
			'tol': tol
		}
		if selected_type == "Clustering":
			model, metrics = train_kmeans_model(df, params)
		else:
			method, transformed_data, variance_explained = decomposition(df, n_comp, model_name)
			st.write("Decomposition Method", method)
			st.write("variance_explained:", variance_explained)
			model, metrics = train_kmeans_model(transformed_data, params)
		st.write("Model:", model)
		st.write("Metrics:", metrics)
		st.write("Lables:⬇️")
		distinct_labels = np.unique(model.labels_)
		st.write(distinct_labels)
		if st.button("Register Model"):
			if selected_type == "Clustering":
				model_info = {
					'model': model,  # The trained model
					'metrics': metrics,
					'model_name': selected_models,  # Name of the model
					'labels': model.labels_,
					'distinct_labels': distinct_labels,
					'selected_type': selected_type
				}

				# Dump to a joblib file
				model_filename = f"{selected_models}_clust_model_registered.pkl"
				model_filename = model_filename.replace(" ", "")
				# Save the dictionary containing the model and its metrics
				with open(model_filename, "wb") as f:
					joblib.dump(model_info, f)
				st.success("Model Registered Successfully ✅")
			else:
				model_info = {
					'model': model,  # The trained model
					'metrics': metrics,
					'decom_method': model_name,
					'decomn_comp': n_comp,	
					'variance_explained': variance_explained,
					'model_name': selected_models,  # Name of the model
					'labels': model.labels_,
					'distinct_labels': distinct_labels,
					'selected_type': selected_type
				}

				# Dump to a joblib file
				model_filename = f"{selected_models}_decom_clust_model_registered.pkl"
				model_filename = model_filename.replace(" ", "")
				# Save the dictionary containing the model and its metrics
				with open(model_filename, "wb") as f:
					joblib.dump(model_info, f)
				st.success("Model Registered Successfully ✅")
	
	elif selected_models == "Birch":
		print("train_decision_tree_classifier")
		st.header("Birch Clustering Hyperparameters")
		threshold = st.slider("Threshold", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
		branching_factor = st.slider("Branching Factor", min_value=50, max_value=500, value=100, step=50)
		n_clusters = st.slider("Number of Clusters", min_value=2, max_value=10, value=2, step=1)

		# Train the model with the selected hyperparameters
		params = {
			'threshold': threshold,
			'branching_factor': branching_factor,
			'n_clusters': n_clusters
		}
		if selected_type == "Clustering":
			model, metrics = train_birch_model(df, params)
		else:
			method, transformed_data, variance_explained = decomposition(df, n_comp, model_name)
			st.write("Decomposition Method", method)
			st.write("variance_explained:", variance_explained)
			model, metrics = train_birch_model(transformed_data, params)
		st.write("Model:", model)
		st.write("Metrics:", metrics)
		st.write("Lables:⬇️")
		distinct_labels = np.unique(model.labels_)
		st.write(distinct_labels)
		if st.button("Register Model"):
			if selected_type == "Clustering":
				model_info = {
					'model': model,  # The trained model
					'metrics': metrics,
					'model_name': selected_models,  # Name of the model
					'labels': model.labels_,
					'distinct_labels': distinct_labels,
					'selected_type': selected_type
				}

				# Dump to a joblib file
				model_filename = f"{selected_models}_clust_model_registered.pkl"
				model_filename = model_filename.replace(" ", "")
				# Save the dictionary containing the model and its metrics
				with open(model_filename, "wb") as f:
					joblib.dump(model_info, f)
				st.success("Model Registered Successfully ✅")
			else:
				model_info = {
					'model': model,  # The trained model
					'metrics': metrics,
					'decom_method': model_name,
					'decomn_comp': n_comp,	
					'variance_explained': variance_explained,
					'model_name': selected_models,  # Name of the model
					'labels': model.labels_,
					'distinct_labels': distinct_labels,
					'selected_type': selected_type
				}

				# Dump to a joblib file
				model_filename = f"{selected_models}_decom_clust_model_registered.pkl"
				model_filename = model_filename.replace(" ", "")
				# Save the dictionary containing the model and its metrics
				with open(model_filename, "wb") as f:
					joblib.dump(model_info, f)
				st.success("Model Registered Successfully ✅")

	elif selected_models == "DBSCAN":
		print("train_decision_tree_classifier")


		# Sidebar for DBSCAN parameters
		st.header("DBSCAN Parameters")
		eps = st.slider("EPS", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
		min_samples = st.slider("Min Samples", min_value=1, max_value=100, value=5, step=1)
		if selected_type == "Clustering":
			model, metrics = train_dbscan_model(df, eps, min_samples)
		else:
			method, transformed_data, variance_explained = decomposition(df, n_comp, model_name)
			st.write("Decomposition Method", method)
			st.write("variance_explained:", variance_explained)
			model, metrics = train_dbscan_model(transformed_data, eps, min_samples)
		# Train the model with the selected parameters
		# model, metrics = train_dbscan_model(df, eps, min_samples)
		st.write("Model:", model)
		st.write("Evaluation Metrics:", metrics)
		st.write("Lables:⬇️")
		distinct_labels = np.unique(model.labels_)
		st.write(distinct_labels)
		if st.button("Register Model"):
			if selected_type == "Clustering":
				model_info = {
					'model': model,  # The trained model
					'metrics': metrics,
					'model_name': selected_models,  # Name of the model
					'labels': model.labels_,
					'distinct_labels': distinct_labels,
					'selected_type': selected_type
				}

				# Dump to a joblib file
				model_filename = f"{selected_models}_clust_model_registered.pkl"
				model_filename = model_filename.replace(" ", "")
				# Save the dictionary containing the model and its metrics
				with open(model_filename, "wb") as f:
					joblib.dump(model_info, f)
				st.success("Model Registered Successfully ✅")
			else:
				model_info = {
					'model': model,  # The trained model
					'metrics': metrics,
					'decom_method': model_name,
					'decomn_comp': n_comp,	
					'variance_explained': variance_explained,
					'model_name': selected_models,  # Name of the model
					'labels': model.labels_,
					'distinct_labels': distinct_labels,
					'selected_type': selected_type
				}

				# Dump to a joblib file
				model_filename = f"{selected_models}_decom_clust_model_registered.pkl"
				model_filename = model_filename.replace(" ", "")
				# Save the dictionary containing the model and its metrics
				with open(model_filename, "wb") as f:
					joblib.dump(model_info, f)
				st.success("Model Registered Successfully ✅")

	elif selected_models == "HDBSCAN":
		print("train_decision_tree_classifier")

		# Sidebar for parameter input
		st.header("HDBSCAN Parameters")
		min_samples = st.slider("Min Samples", min_value=1, max_value=100, value=5, step=1)
		min_cluster_size = st.slider("Min Cluster Size", min_value=1, max_value=100, value=5, step=1)
		cluster_selection_epsilon = st.slider("Cluster Selection Epsilon", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
		if selected_type == "Clustering":
			model, metrics = train_hdbscan_model(df, min_samples, min_cluster_size, cluster_selection_epsilon)
		else:
			method, transformed_data, variance_explained = decomposition(df, n_comp, model_name)
			st.write("Decomposition Method", method)
			st.write("variance_explained:", variance_explained)
			model, metrics = train_hdbscan_model(transformed_data, min_samples, min_cluster_size, cluster_selection_epsilon)
		st.write("Model:", model)
		st.write("Metrics:", metrics)
		st.write("Lables:⬇️")
		distinct_labels = np.unique(model.labels_)
		st.write(distinct_labels)
		if st.button("Register Model"):
			if selected_type == "Clustering":
				model_info = {
					'model': model,  # The trained model
					'metrics': metrics,
					'model_name': selected_models,  # Name of the model
					'labels': model.labels_,
					'distinct_labels': distinct_labels,
					'selected_type': selected_type
				}

				# Dump to a joblib file
				model_filename = f"{selected_models}_clust_model_registered.pkl"
				model_filename = model_filename.replace(" ", "")
				# Save the dictionary containing the model and its metrics
				with open(model_filename, "wb") as f:
					joblib.dump(model_info, f)
				st.success("Model Registered Successfully ✅")
			else:
				model_info = {
					'model': model,  # The trained model
					'metrics': metrics,
					'decom_method': model_name,
					'decomn_comp': n_comp,	
					'variance_explained': variance_explained,
					'model_name': selected_models,  # Name of the model
					'labels': model.labels_,
					'distinct_labels': distinct_labels,
					'selected_type': selected_type
				}

				# Dump to a joblib file
				model_filename = f"{selected_models}_decom_clust_model_registered.pkl"
				model_filename = model_filename.replace(" ", "")
				# Save the dictionary containing the model and its metrics
				with open(model_filename, "wb") as f:
					joblib.dump(model_info, f)
				st.success("Model Registered Successfully ✅")

	elif selected_models == "Agglomerative Clustering":
		print("train_decision_tree_classifier")

		# Sidebar for hyperparameter input
		st.header("Agglomerative Clustering Hyperparameters")
		n_clusters = st.slider("Number of Clusters", min_value=2, max_value=20, value=2, step=1)
		affinity = st.selectbox("Affinity", ["euclidean", "l1", "l2", "manhattan", "cosine"], index=0)
		linkage = st.selectbox("Linkage", ["ward", "complete", "average", "single"], index=0)

		# Train the model with the selected hyperparameters
		params = {
			'n_clusters': n_clusters,
			'affinity': affinity,
			'linkage': linkage
		}
		if selected_type == "Clustering":
			model, metrics = train_agglomerative_clus_model(df, params)
		else:
			method, transformed_data, variance_explained = decomposition(df, n_comp, model_name)
			st.write("Decomposition Method", method)
			st.write("variance_explained:", variance_explained)
			model, metrics = train_agglomerative_clus_model(transformed_data, params)
		st.write("Model:", model)
		st.write("Metrics:", metrics)
		st.write("Lables:⬇️")
		distinct_labels = np.unique(model.labels_)
		st.write(distinct_labels)
		if st.button("Register Model"):
			if selected_type == "Clustering":
				model_info = {
					'model': model,  # The trained model
					'metrics': metrics,
					'model_name': selected_models,  # Name of the model
					'labels': model.labels_,
					'distinct_labels': distinct_labels,
					'selected_type': selected_type
				}

				# Dump to a joblib file
				model_filename = f"{selected_models}_clust_model_registered.pkl"
				model_filename = model_filename.replace(" ", "")
				# Save the dictionary containing the model and its metrics
				with open(model_filename, "wb") as f:
					joblib.dump(model_info, f)
				st.success("Model Registered Successfully ✅")
			else:
				model_info = {
					'model': model,  # The trained model
					'metrics': metrics,
					'decom_method': model_name,
					'decomn_comp': n_comp,	
					'variance_explained': variance_explained,
					'model_name': selected_models,  # Name of the model
					'labels': model.labels_,
					'distinct_labels': distinct_labels,
					'selected_type': selected_type
				}

				# Dump to a joblib file
				model_filename = f"{selected_models}_decom_clust_model_registered.pkl"
				model_filename = model_filename.replace(" ", "")
				# Save the dictionary containing the model and its metrics
				with open(model_filename, "wb") as f:
					joblib.dump(model_info, f)
				st.success("Model Registered Successfully ✅")

	elif selected_models == "Mean Shift":
		print("train_decision_tree_classifier")


		# Sidebar for bandwidth input
		st.header("MeanShift Clustering Hyperparameters")
		bandwidth = st.slider("Bandwidth", min_value=0.1, max_value=5.0, value=1.0, step=0.1)

		if selected_type == "Clustering":
			model, metrics = train_mean_shift_model(df, bandwidth)
		else:
			method, transformed_data, variance_explained = decomposition(df, n_comp, model_name)
			st.write("Decomposition Method", method)
			st.write("variance_explained:", variance_explained)
			model, metrics = train_mean_shift_model(transformed_data, bandwidth)
		st.write("Model:", model)
		st.write("Metrics:", metrics)
		st.write("Lables:⬇️")
		distinct_labels = np.unique(model.labels_)
		st.write(distinct_labels)
		if st.button("Register Model"):
			if selected_type == "Clustering":
				model_info = {
					'model': model,  # The trained model
					'metrics': metrics,
					'model_name': selected_models,  # Name of the model
					'labels': model.labels_,
					'distinct_labels': distinct_labels,
					'selected_type': selected_type
				}

				# Dump to a joblib file
				model_filename = f"{selected_models}_clust_model_registered.pkl"
				model_filename = model_filename.replace(" ", "")
				# Save the dictionary containing the model and its metrics
				with open(model_filename, "wb") as f:
					joblib.dump(model_info, f)
				st.success("Model Registered Successfully ✅")
			else:
				model_info = {
					'model': model,  # The trained model
					'metrics': metrics,
					'decom_method': model_name,
					'decomn_comp': n_comp,	
					'variance_explained': variance_explained,
					'model_name': selected_models,  # Name of the model
					'labels': model.labels_,
					'distinct_labels': distinct_labels,
					'selected_type': selected_type
				}

				# Dump to a joblib file
				model_filename = f"{selected_models}_decom_clust_model_registered.pkl"
				model_filename = model_filename.replace(" ", "")
				# Save the dictionary containing the model and its metrics
				with open(model_filename, "wb") as f:
					joblib.dump(model_info, f)
				st.success("Model Registered Successfully ✅")

	elif selected_models == "Affinity Propagation":
		print("train_decision_tree_classifier")
		# Sidebar for hyperparameter input
		st.header("AffinityPropagation Hyperparameters")
		damping = st.slider("Damping", min_value=0.5, max_value=1.0, value=0.9, step=0.05)
		max_iter = st.slider("Max Iterations", min_value=100, max_value=1000, value=200, step=50)
		convergence_iter = st.slider("Convergence Iterations", min_value=5, max_value=50, value=15, step=5)
		preference = st.number_input("Preference (Higher: Fewer Clusters)", value=-50.0, step=5.0)
		affinity = st.selectbox("Affinity", options=["euclidean", "precomputed"], index=0)

		# Train the AffinityPropagation model with the selected hyperparameters
		params = {
			'damping': damping,
			'max_iter': max_iter,
			'convergence_iter': convergence_iter,
			'preference': preference,
			'affinity': affinity
		}

		if selected_type == "Clustering":
			model, metrics = train_affinity_propo_model(df, params)
		else:
			method, transformed_data, variance_explained = decomposition(df, n_comp, model_name)
			st.write("Decomposition Method", method)
			st.write("variance_explained:", variance_explained)
			model, metrics = train_affinity_propo_model(transformed_data, params)
		st.write("Model:", model)
		st.write("Metrics:", metrics)
		st.write("Lables:⬇️")
		distinct_labels = np.unique(model.labels_)
		st.write(distinct_labels)
		if st.button("Register Model"):
			if selected_type == "Clustering":
				model_info = {
					'model': model,  # The trained model
					'metrics': metrics,
					'model_name': selected_models,  # Name of the model
					'labels': model.labels_,
					'distinct_labels': distinct_labels,
					'selected_type': selected_type
				}

				# Dump to a joblib file
				model_filename = f"{selected_models}_clust_model_registered.pkl"
				model_filename = model_filename.replace(" ", "")
				# Save the dictionary containing the model and its metrics
				with open(model_filename, "wb") as f:
					joblib.dump(model_info, f)
				st.success("Model Registered Successfully ✅")
			else:
				model_info = {
					'model': model,  # The trained model
					'metrics': metrics,
					'decom_method': model_name,
					'decomn_comp': n_comp,	
					'variance_explained': variance_explained,
					'model_name': selected_models,  # Name of the model
					'labels': model.labels_,
					'distinct_labels': distinct_labels,
					'selected_type': selected_type
				}

				# Dump to a joblib file
				model_filename = f"{selected_models}_decom_clust_model_registered.pkl"
				model_filename = model_filename.replace(" ", "")
				# Save the dictionary containing the model and its metrics
				with open(model_filename, "wb") as f:
					joblib.dump(model_info, f)
				st.success("Model Registered Successfully ✅")

	elif selected_models == "Spectral Clustering":
		print("train_decision_tree_classifier")
		# Sidebar for hyperparameter input
		st.header("SpectralClustering Hyperparameters")
		n_clusters = st.slider("Number of Clusters", min_value=2, max_value=10, value=3, step=1)
		assign_labels = st.selectbox("Assign Labels Method", ["kmeans", "discretize"])
		affinity = st.selectbox("Affinity", ["rbf", "nearest_neighbors", "precomputed", "cosine", "poly"])
		gamma = st.slider("Gamma (only for RBF affinity)", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
		degree = st.slider("Degree (only for poly affinity)", min_value=2, max_value=5, value=3, step=1)
		coef0 = st.slider("Coefficient 0 (for poly and sigmoid affinity)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

		# Hyperparameters dictionary
		params = {
			'n_clusters': n_clusters,
			'assign_labels': assign_labels,
			'affinity': affinity,
			'gamma': gamma,
			'degree': degree,
			'coef0': coef0,
		}

		if selected_type == "Clustering":
			model, metrics = train_spectral_clust_model(df, params)
		else:
			method, transformed_data, variance_explained = decomposition(df, n_comp, model_name)
			st.write("Decomposition Method", method)
			st.write("variance_explained:", variance_explained)
			model, metrics = train_spectral_clust_model(transformed_data, params)
		st.write("Model:", model)
		st.write("Metrics:", metrics)
		st.write("Lables:⬇️")
		distinct_labels = np.unique(model.labels_)
		st.write(distinct_labels)
		if st.button("Register Model"):
			if selected_type == "Clustering":
				model_info = {
					'model': model,  # The trained model
					'metrics': metrics,
					'model_name': selected_models,  # Name of the model
					'labels': model.labels_,
					'distinct_labels': distinct_labels,
					'selected_type': selected_type
				}

				# Dump to a joblib file
				model_filename = f"{selected_models}_clust_model_registered.pkl"
				model_filename = model_filename.replace(" ", "")
				# Save the dictionary containing the model and its metrics
				with open(model_filename, "wb") as f:
					joblib.dump(model_info, f)
				st.success("Model Registered Successfully ✅")
			else:
				model_info = {
					'model': model,  # The trained model
					'metrics': metrics,
					'decom_method': model_name,
					'decomn_comp': n_comp,	
					'variance_explained': variance_explained,
					'model_name': selected_models,  # Name of the model
					'labels': model.labels_,
					'distinct_labels': distinct_labels,
					'selected_type': selected_type
				}

				# Dump to a joblib file
				model_filename = f"{selected_models}_decom_clust_model_registered.pkl"
				model_filename = model_filename.replace(" ", "")
				# Save the dictionary containing the model and its metrics
				with open(model_filename, "wb") as f:
					joblib.dump(model_info, f)
				st.success("Model Registered Successfully ✅")
		

		# st.write(metrics)

	elif selected_models == "OPTICS":
		print("train_decision_tree_classifier")
		# Sidebar for hyperparameter input
		st.header("OPTICS Hyperparameters")

		# Hyperparameter sliders for OPTICS
		min_samples = st.slider("Min Samples", min_value=2, max_value=100, value=5, step=1)
		max_eps = st.slider("Max Eps", min_value=0.1, max_value=10.0, value=2.0, step=0.1)
		cluster_method = st.selectbox("Cluster Method", ["xi", "dbscan"], index=0)
		xi = st.slider("Xi", min_value=0.01, max_value=0.3, value=0.05, step=0.01)
		predecessor_correction = st.checkbox("Predecessor Correction", value=True)

		# Parameter dictionary
		params = {
			'min_samples': min_samples,
			'max_eps': max_eps,
			'cluster_method': cluster_method,
			'xi': xi,
			'predecessor_correction': predecessor_correction,
		}

		if selected_type == "Clustering":
			model, metrics = train_optics_model(df, params)
		else:
			method, transformed_data, variance_explained = decomposition(df, n_comp, model_name)
			st.write("Decomposition Method", method)
			st.write("variance_explained:", variance_explained)
			model, metrics = train_optics_model(transformed_data, params)
		st.write("Model:", model)
		st.write("Metrics:", metrics)
		st.write("Lables:⬇️")
		distinct_labels = np.unique(model.labels_)
		st.write(distinct_labels)
		if st.button("Register Model"):
			if selected_type == "Clustering":
				model_info = {
					'model': model,  # The trained model
					'metrics': metrics,
					'model_name': selected_models,  # Name of the model
					'labels': model.labels_,
					'distinct_labels': distinct_labels,
					'selected_type': selected_type
				}

				# Dump to a joblib file
				model_filename = f"{selected_models}_clust_model_registered.pkl"
				model_filename = model_filename.replace(" ", "")
				# Save the dictionary containing the model and its metrics
				with open(model_filename, "wb") as f:
					joblib.dump(model_info, f)
				st.success("Model Registered Successfully ✅")
			else:
				model_info = {
					'model': model,  # The trained model
					'metrics': metrics,
					'decom_method': model_name,
					'decomn_comp': n_comp,	
					'variance_explained': variance_explained,
					'model_name': selected_models,  # Name of the model
					'labels': model.labels_,
					'distinct_labels': distinct_labels,
					'selected_type': selected_type
				}

				# Dump to a joblib file
				model_filename = f"{selected_models}_decom_clust_model_registered.pkl"
				model_filename = model_filename.replace(" ", "")
				# Save the dictionary containing the model and its metrics
				with open(model_filename, "wb") as f:
					joblib.dump(model_info, f)
				st.success("Model Registered Successfully ✅")



	elif selected_models == "Mini Batch K-Means":
		print("train_decision_tree_classifier")
		# Sidebar for hyperparameter input
		st.header("MiniBatchKMeans Hyperparameters")
		n_clusters = st.slider("Number of Clusters", min_value=2, max_value=50, value=8, step=1)
		batch_size = st.slider("Batch Size", min_value=10, max_value=1000, value=100, step=10)
		max_iter = st.slider("Max Iterations", min_value=10, max_value=300, value=100, step=10)
		random_state = st.number_input("Random State", min_value=0, value=0)
		reassignment_ratio = st.slider("Reassignment Ratio", min_value=0.0, max_value=1.0, value=0.01, step=0.01)
		tol = st.slider("Tolerance", min_value=1e-5, max_value=1e-1, value=1e-4, step=1e-5)

		# Prepare parameters for MiniBatchKMeans
		params = {
			'n_clusters': n_clusters,
			'batch_size': batch_size,
			'max_iter': max_iter,
			'random_state': random_state,
			'reassignment_ratio': reassignment_ratio,
			'tol': tol
		}

		if selected_type == "Clustering":
			model, metrics = train_mini_batch_kmeans_model(df, params)
		else:
			method, transformed_data, variance_explained = decomposition(df, n_comp, model_name)
			st.write("Decomposition Method", method)
			st.write("variance_explained:", variance_explained)
			model, metrics = train_mini_batch_kmeans_model(transformed_data, params)
		st.write("Model:", model)
		st.write("Metrics:", metrics)
		st.write("Lables:⬇️")
		distinct_labels = np.unique(model.labels_)
		st.write(distinct_labels)
		if st.button("Register Model"):
			if selected_type == "Clustering":
				model_info = {
					'model': model,  # The trained model
					'metrics': metrics,
					'model_name': selected_models,  # Name of the model
					'labels': model.labels_,
					'distinct_labels': distinct_labels,
					'selected_type': selected_type
				}

				# Dump to a joblib file
				model_filename = f"{selected_models}_clust_model_registered.pkl"
				model_filename = model_filename.replace(" ", "")
				# Save the dictionary containing the model and its metrics
				with open(model_filename, "wb") as f:
					joblib.dump(model_info, f)
				st.success("Model Registered Successfully ✅")
			else:
				model_info = {
					'model': model,  # The trained model
					'metrics': metrics,
					'decom_method': model_name,
					'decomn_comp': n_comp,	
					'variance_explained': variance_explained,
					'model_name': selected_models,  # Name of the model
					'labels': model.labels_,
					'distinct_labels': distinct_labels,
					'selected_type': selected_type
				}

				# Dump to a joblib file
				model_filename = f"{selected_models}_decom_clust_model_registered.pkl"
				model_filename = model_filename.replace(" ", "")
				# Save the dictionary containing the model and its metrics
				with open(model_filename, "wb") as f:
					joblib.dump(model_info, f)
				st.success("Model Registered Successfully ✅")

		
	elif selected_models == "Feature Agglomeration":
		print("train_decision_tree_classifier")


		# Sidebar for hyperparameter input
		st.header("FeatureAgglomeration Hyperparameters")
		n_clusters = st.slider("Number of Clusters", min_value=2, max_value=20, value=5, step=1)
		affinity = st.selectbox("Affinity", ["euclidean", "manhattan", "cosine", "precomputed"])
		linkage = st.selectbox("Linkage", ["ward", "complete", "average", "single"])

		# Train the model with the selected hyperparameters
		params = {
			'n_clusters': n_clusters,
			'affinity': affinity,
			'linkage': linkage
		}
		if selected_type == "Clustering":
			model, metrics = train_feature_agglo_model(df, params)
		else:
			method, transformed_data, variance_explained = decomposition(df, n_comp, model_name)
			st.write("Decomposition Method", method)
			st.write("variance_explained:", variance_explained)
			model, metrics = train_feature_agglo_model(transformed_data, params)
		st.write("Model:", model)
		st.write("Metrics:", metrics)
		st.write("Lables:⬇️")
		distinct_labels = np.unique(model.labels_)
		st.write(distinct_labels)
		if st.button("Register Model"):
			if selected_type == "Clustering":
				model_info = {
					'model': model,  # The trained model
					'metrics': metrics,
					'model_name': selected_models,  # Name of the model
					'labels': model.labels_,
					'distinct_labels': distinct_labels,
					'selected_type': selected_type
				}

				# Dump to a joblib file
				model_filename = f"{selected_models}_clust_model_registered.pkl"
				model_filename = model_filename.replace(" ", "")
				# Save the dictionary containing the model and its metrics
				with open(model_filename, "wb") as f:
					joblib.dump(model_info, f)
				st.success("Model Registered Successfully ✅")
			else:
				model_info = {
					'model': model,  # The trained model
					'metrics': metrics,
					'decom_method': model_name,
					'decomn_comp': n_comp,	
					'variance_explained': variance_explained,
					'model_name': selected_models,  # Name of the model
					'labels': model.labels_,
					'distinct_labels': distinct_labels,
					'selected_type': selected_type
				}

				# Dump to a joblib file
				model_filename = f"{selected_models}_decom_clust_model_registered.pkl"
				model_filename = model_filename.replace(" ", "")
				# Save the dictionary containing the model and its metrics
				with open(model_filename, "wb") as f:
					joblib.dump(model_info, f)
				st.success("Model Registered Successfully ✅")


		

except Exception as e:
	# st.error(e)
	print(e)
	
	if "X_train" in str(e):
		st.warning("Check if proper preprocessing and data splitting is done or not!")
	else:
		st.warning("Select custom model type for tuning")
	# pass
	