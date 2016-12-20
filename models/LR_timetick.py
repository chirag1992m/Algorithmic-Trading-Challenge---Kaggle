#Check notebook: Adding_Time_Parameter.ipynb

import pandas as pd
import numpy as np
from sklearn import linear_model as lm

import pickle

print "Loading Data..."
train_table = pd.DataFrame.from_csv('../data/subset_train_OHE.csv')
test_table = pd.DataFrame.from_csv('../data/subset_test_OHE.csv')

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

print "Creating Predictors..."
modelTill = 10

trainX = np.zeros((train_table.shape[0] * modelTill, len(featureColumns) + 1))

trainY_bid = np.zeros((train_table.shape[0] * modelTill))
trainY_ask = np.zeros((train_table.shape[0] * modelTill))

index = 0
for ix, row in train_table.iterrows():
	X_features = np.array(row[featureColumns])
	for i in range(modelTill):
		X = np.concatenate((X_features, np.array([i])))
		trainX[index, :] = X
		trainY_ask[index] = row[askPredictionColumns[i]]
		trainY_bid[index] = row[bidPredictionColumns[i]]
		index = index+1

testX = np.zeros((test_table.shape[0] * modelTill, len(featureColumns) + 1))
index = 0
for ix, row in test_table.iterrows():
	X_features = np.array(row[featureColumns])
	for i in range(modelTill):
		X = np.concatenate((X_features, np.array([i])))
		testX[index, :] = X
		index = index+1


print "Training Models..."
LR_model_ask = lm.LinearRegression(fit_intercept=True, normalize=False, n_jobs=-1)
LR_model_ask.fit(trainX, trainY_ask)

LR_model_bid = lm.LinearRegression(fit_intercept=True, normalize=False, n_jobs=-1)
LR_model_bid.fit(trainX, trainY_bid)

models = {'bid': LR_model_bid, 'ask': LR_model_ask}

with open('../run_models/LR_timetick.model', 'wb') as output:
	pickle.dump(models, output, -1)


print "Predicting..."
testY_ask = LR_model_ask.predict(testX)
testY_bid = LR_model_bid.predict(testX)

prediction = pd.DataFrame.from_csv('../predictions/template_prediction.csv')

index = 0
for ix, row in test_table.iterrows():
	row_id = row['row_id']

	index_in_pred = prediction[prediction['row_id'] == row_id].index.tolist()[0]

	for i in range(modelTill):
		ask = testY_ask[index]
		bid = testY_bid[index]
		index = index+1

		prediction.set_value(index_in_pred, bidPredictionColumns[i], bid)
		prediction.set_value(index_in_pred, askPredictionColumns[i], ask)

	for i in range(modelTill, len(askPredictionColumns)):
		prediction.set_value(index_in_pred, bidPredictionColumns[i], bid)
		prediction.set_value(index_in_pred, askPredictionColumns[i], ask)

prediction.to_csv('../predictions/LR_timetick.csv')

print "Done!"