#For more info: ../notebooks/ClusterAndClassify_Bid-Ask_Behaviors.ipynb

import pandas as pd
import numpy as np

import sklearn.cluster as cluster
import sklearn.preprocessing as process
import sklearn.ensemble as ensemble
import sklearn.multiclass as multiclass

#Fetching data
print "Fetching data..."
train_set = pd.DataFrame.from_csv('../data/subset_train_OHE.csv')

#Constructing Predictors
print "Constructing Predictors..."
all_bid_columns = []
for column in train_set.columns.values:
	if column.startswith('bid'):
		all_bid_columns.append(column)

all_ask_columns = []
for column in train_set.columns.values:
	if column.startswith('ask'):
		all_ask_columns.append(column)

all_bid_prices = np.array(train_set[all_bid_columns])
all_ask_prices = np.array(train_set[all_ask_columns])

all_bid_prices_nm = process.scale(all_bid_prices, axis=1)
all_ask_prices_nm = process.scale(all_ask_prices, axis=1)

#Generating the clusters
print "Modelling Clusters..."
clusters = 4
all_bid_cluster_model = cluster.KMeans(n_clusters=clusters,
	init='k-means++',
	n_init=10,
	max_iter=300,
	tol=0.0001,
	precompute_distances='auto',
	verbose=0,
	random_state=None,
	copy_x=True,
	n_jobs=-1)
all_bid_cluster_model.fit(all_bid_prices_nm)
all_bid_labels = all_bid_cluster_model.predict(all_bid_prices_nm)

all_ask_cluster_model = cluster.KMeans(n_clusters=clusters,
	init='k-means++',
	n_init=10,
	max_iter=300,
	tol=0.0001,
	precompute_distances='auto',
	verbose=0,
	random_state=None,
	copy_x=True,
	n_jobs=-1)
all_ask_cluster_model.fit(all_ask_prices_nm)
all_ask_labels = all_ask_cluster_model.predict(all_ask_prices_nm)

#Classifying on the basis of clusters
print "Classifying..."
bid_cluster_classifier_ada = multiclass.OneVsOneClassifier(estimator=ensemble.AdaBoostClassifier(base_estimator=None,
	n_estimators=50,
	learning_rate=1.0,
	algorithm='SAMME.R',
	random_state=None),
	n_jobs=-1)
bid_cluster_classifier_ada.fit(trainFeatures, all_bid_labels)
print "Bid accuracy with AdaBoost: ", bid_cluster_classifier.score(trainFeatures, all_bid_labels)

ask_cluster_classifier_ada = multiclass.OneVsOneClassifier(estimator=ensemble.AdaBoostClassifier(base_estimator=None,
	n_estimators=50,
	learning_rate=1.0,
	algorithm='SAMME.R',
	random_state=None),
	n_jobs=-1)
ask_cluster_classifier_ada.fit(trainFeatures, all_ask_labels)
print "Ask accuracy with AdaBoost Classifier: ", ask_cluster_classifier.score(trainFeatures, all_ask_labels)


bid_cluster_classifier_bagging = multiclass.OneVsOneClassifier(estimator=ensemble.BaggingClassifier(base_estimator=None, 
	n_estimators=10,
	max_samples=1.0,
	max_features=1.0,
	bootstrap=True,
	bootstrap_features=False,
	oob_score=False,
	warm_start=False,
	n_jobs=1,
	random_state=None,
	verbose=1),
	n_jobs=-1)
bid_cluster_classifier_bagging.fit(trainFeatures, all_bid_labels)
print "Bid accuracy with Bagging: ", bid_cluster_classifier.score(trainFeatures, all_bid_labels)

ask_cluster_classifier_bagging = multiclass.OneVsOneClassifier(estimator=ensemble.BaggingClassifier(base_estimator=None, 
	n_estimators=10,
	max_samples=1.0,
	max_features=1.0,
	bootstrap=True,
	bootstrap_features=False,
	oob_score=False,
	warm_start=False,
	n_jobs=1,
	random_state=None,
	verbose=1),
	n_jobs=-1)
ask_cluster_classifier_bagging.fit(trainFeatures, all_ask_labels)
print "Ask accuracy with Bagging: ", ask_cluster_classifier.score(trainFeatures, all_ask_labels)


bid_cluster_classifier_rfc = multiclass.OneVsOneClassifier(estimator=ensemble.RandomForestClassifier(n_estimators=10,
	criterion='gini',
	max_depth=None,
	min_samples_split=2,
	min_samples_leaf=1,
	min_weight_fraction_leaf=0.0,
	max_features='auto',
	max_leaf_nodes=None,
	min_impurity_split=1e-07,
	bootstrap=True,
	oob_score=False,
	n_jobs=1,
	random_state=None,
	verbose=0,
	warm_start=False,
	class_weight=None),
	n_jobs=-1)
bid_cluster_classifier_rfc.fit(trainFeatures, all_bid_labels)
print "Bid accuracy with Random Forest: ", bid_cluster_classifier.score(trainFeatures, all_bid_labels)

ask_cluster_classifier_rfc = multiclass.OneVsOneClassifier(estimator=ensemble.RandomForestClassifier(n_estimators=10,
	criterion='gini',
	max_depth=None,
	min_samples_split=2,
	min_samples_leaf=1,
	min_weight_fraction_leaf=0.0,
	max_features='auto',
	max_leaf_nodes=None,
	min_impurity_split=1e-07,
	bootstrap=True,
	oob_score=False,
	n_jobs=1,
	random_state=None,
	verbose=0,
	warm_start=False,
	class_weight=None),
	n_jobs=-1)
ask_cluster_classifier_rfc.fit(trainFeatures, all_ask_labels)
print "Ask accuracy with Random Forest: ", ask_cluster_classifier.score(trainFeatures, all_ask_labels)