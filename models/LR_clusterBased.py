#Notebook: Cluster_Based_Modelling.ipynb

import pandas as pd
import numpy as np
import sklearn.cluster as cluster
from scipy.interpolate import interp1d #for smooth line plots
import sklearn.preprocessing as process
import sklearn.ensemble as ensemble
import sklearn.multiclass as multiclass

print "Loading data..."
train_table = pd.DataFrame.from_csv('../data/subset_train_OHE.csv')
test_table = pd.DataFrame.from_csv('../data/subset_test_OHE.csv')

print "Constructing Features..."
bidPredictionColumns = []
for i in range(52, 101):
    for column in train_table.columns.values:
        if column.endswith(str(i)) and column.startswith('bid'):
            bidPredictionColumns.append(column)

askPredictionColumns = []
for i in range(52, 101):
    for column in train_table.columns.values:
        if column.endswith(str(i)) and column.startswith('ask'):
            askPredictionColumns.append(column)

predictionColumns = bidPredictionColumns + askPredictionColumns

featureColumns = []
columnsToIgnore = ['row_id']
for column in train_table.columns.values:
    if ((column not in predictionColumns) and (column not in columnsToIgnore) and (not column.startswith('time'))):
        featureColumns.append(column)

all_bid_columns = []
for column in train_table.columns.values:
    if column.startswith('bid'):
        all_bid_columns.append(column)

all_ask_columns = []
for column in train_table.columns.values:
    if column.startswith('ask'):
        all_ask_columns.append(column)


print "Clustering and Labelling..."
all_bid_prices = np.array(train_table[all_bid_columns])
all_ask_prices = np.array(train_table[all_ask_columns])


all_bid_prices_nm = process.scale(all_bid_prices, axis=1)
all_ask_prices_nm = process.scale(all_ask_prices, axis=1)

all_clusters = 4
all_bid_cluster_model = cluster.KMeans(n_clusters=all_clusters,
                                   init='k-means++',
                                   n_init=10,
                                   max_iter=300,
                                   tol=0.0001,
                                   precompute_distances='auto',
                                   verbose=0,
                                   random_state=None,
                                   copy_x=True,
                                   n_jobs=1)
all_bid_cluster_model.fit(all_bid_prices_nm)
all_bid_labels = all_bid_cluster_model.predict(all_bid_prices_nm)

all_ask_cluster_model = cluster.KMeans(n_clusters=all_clusters,
                                   init='k-means++',
                                   n_init=10,
                                   max_iter=300,
                                   tol=0.0001,
                                   precompute_distances='auto',
                                   verbose=0,
                                   random_state=None,
                                   copy_x=True,
                                   n_jobs=1)
all_ask_cluster_model.fit(all_ask_prices_nm)
all_ask_labels = all_ask_cluster_model.predict(all_ask_prices_nm)

print "Data Construction..."
trainX = np.zeros((train_table.shape[0], len(featureColumns)))

trainY_ask = np.zeros((train_table.shape[0]))
trainY_bid = np.zeros((train_table.shape[0]))

testX = np.zeros((test_table.shape[0], len(featureColumns)))

index = 0
for ix, row in train_table.iterrows():
    X = (np.array(row[featureColumns])).flatten('F')
    Y_bid = row[bidPredictionColumns[0]]
    Y_ask = row[askPredictionColumns[0]]
    
    trainX[index, :] = X
    trainY_ask[index] = Y_ask
    trainY_bid[index] = Y_bid
    
    index = index+1

index = 0
for ix, row in test_table.iterrows():
    X = (np.array(row[featureColumns])).flatten('F')
    testX[index, :] = X

    index = index+1

print "Classifier for Clusters..."
bid_cluster_classifier = multiclass.OneVsOneClassifier(estimator=ensemble.RandomForestClassifier(n_estimators=30,
	criterion='gini',
	max_depth=None,
	min_samples_split=2,
	min_samples_leaf=1,
	min_weight_fraction_leaf=0.0,
	max_features='auto',
	max_leaf_nodes=None,
	bootstrap=True,
	oob_score=False,
	n_jobs=1,
	random_state=None,
	verbose=0,
	warm_start=False,
	class_weight=None),
	n_jobs=-1)
bid_cluster_classifier.fit(trainFeatures, all_bid_labels)
print "Bid accuracy with Random Forest: ", bid_cluster_classifier_rfc.score(trainFeatures, all_bid_labels)

ask_cluster_classifier = multiclass.OneVsOneClassifier(estimator=ensemble.RandomForestClassifier(n_estimators=30,
	criterion='gini',
	max_depth=None,
	min_samples_split=2,
	min_samples_leaf=1,
	min_weight_fraction_leaf=0.0,
	max_features='auto',
	max_leaf_nodes=None,
	bootstrap=True,
	oob_score=False,
	n_jobs=1,
	random_state=None,
	verbose=0,
	warm_start=False,
	class_weight=None),
	n_jobs=-1)
ask_cluster_classifier.fit(trainFeatures, all_ask_labels)


print "Training..."
models_bid = []
models_ask = []
for i in range(all_clusters):
    model_ask = lm.LinearRegression(fit_intercept=True, normalize=False, n_jobs=1)
    model_ask.fit(trainX[all_ask_labels == i, :], trainY_ask[all_ask_labels == i])
    models_ask.append(model_ask)
    
    model_bid = lm.LinearRegression(fit_intercept=True, normalize=False, n_jobs=1)
    model_bid.fit(trainX[all_bid_labels == i, :], trainY_bid[all_bid_labels == i])
    models_ask.append(model_bid)


print "Predicting Labels..."
testX_ask_labels = ask_cluster_classifier.predict(testX)
testX_bid_labels = bid_cluster_classifier.predict(testX)

testY_ask = np.zeros((textX.shape[0]))
testY_ask = np.zeros((textX.shape[0]))

print "Predicting Bid-Ask..."
for i in range(all_clusters):
    testY_ask[testX_ask_labels == i] = models_ask[i].predict(testX[testX_ask_labels == i, :])
    testY_bid[testX_bid_labels == i] = models_bid[i].predict(testX[testX_bid_labels == i, :])

print "Constructing Prediction..."
prediction = pd.DataFrame.from_csv('../predictions/template_prediction.csv')

i = 0
for ix, row in test_table.iterrows():
    row_id = row['row_id']

    index_in_pred = prediction[prediction['row_id'] == row_id].index.tolist()[0]

    bid = testY_bid[i]
    ask = testY_ask[i]
    i = i+1
    
    for column in predictionColumns:
        if column.startswith('bid'):
            prediction.set_value(index_in_pred, column, bid)
        else:
            prediction.set_value(index_in_pred, column, ask)

prediction.to_csv('../predictions/LR_clusterBased.csv')
print "Done!"