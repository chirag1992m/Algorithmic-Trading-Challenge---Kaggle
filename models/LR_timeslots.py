#Notebook: Modelling_Time_Slots.ipynb

import pandas as pd
import numpy as np
from sklearn import linear_model as lm

import pickle

print "Loading Data..."
train_table = pd.DataFrame.from_csv('../data/subset_train_OHE.csv')
test_table = pd.DataFrame.from_csv('../data/subset_test_OHE.csv')

granularity = 10
start = 52
end = 101

ranges = []
current_start = start
while(True):
	current_end = current_start + granularity
	if current_end > end:
		ranges.append(range(current_start, end))
		break
	ranges.append(range(current_start, current_end))
	current_start = current_end

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

featureColumnsTimeWise = [featureColumns]

print "Creating Predictors..."
currentFeatureColumns = featureColumns[:]
index = 0
for i in range(len(ranges) - 1):
	for k in range(len(ranges[i])):
		currentFeatureColumns.append(bidPredictionColumns[index])
		currentFeatureColumns.append(askPredictionColumns[index])
		index = index+1
	featureColumnsTimeWise.append(currentFeatureColumns[:])

trainX = []
trainY_ask = []
trainY_bid = []

for i in range(len(ranges)):
	trainX_intermediate = np.zeros((train_table.shape[0], len(featureColumnsTimeWise[i])))
	trainY_ask_intermediate = np.zeros((train_table.shape[0]))
	trainY_bid_intermediate = np.zeros((train_table.shape[0]))

	index = 0
	for ix, row in train_table.iterrows():
		trainX_intermediate[index, :] = row[featureColumnsTimeWise[i]]
		trainY_ask_intermediate[index] = row['ask' + str(ranges[i][0])]
		trainY_bid_intermediate[index] = row['bid' + str(ranges[i][0])]
		index = index+1

	trainX.append(trainX_intermediate)
	trainY_ask.append(trainY_ask_intermediate)
	trainY_bid.append(trainY_bid_intermediate)


testX = []
for i in range(len(ranges)):
	testX_intermediate = np.zeros((train_table.shape[0], len(featureColumnsTimeWise[i])))

	index = 0
	for ix, row in test_table.iterrows():
		testX_intermediate[index, :] = row[featureColumnsTimeWise[i]]
		index = index+1

	testX.append(testX_intermediate)


print "Training..."
models_ask = []
models_bid = []

for i in range(len(ranges)):
	model_ask = lm.LinearRegression(fit_intercept=True, normalize=False, n_jobs=1)
	model_ask.fit(trainX[i], trainY_ask[i])
	models_ask.append(model_ask)

	model_bid = lm.LinearRegression(fit_intercept=True, normalize=False, n_jobs=1)
	model_bid.fit(trainX[i], trainY_bid[i])
	models_bid.append(model_bid)

models = {'bid': models_bid, 'ask': models_ask}

with open('../run_models/LR_timeslots.model', 'wb') as output:
	pickle.dump(models, output, -1)


print "Predicting..."
testY_ask = []
testY_bid = []
for i in range(len(ranges)):
	testY_ask_temp = ask_models[i].predict(testX[i])
	testY_ask.append(testY_ask_temp)

	testY_bid_temp = bid_models[i].predict(testX[i])
	testY_bid.append(testY_bid_temp)

	print "predicted ", i
	#Use the current prediction to fill
	#test set for upcoming predictions
	lastColumnNumber = testX[i].shape[1]
	if i != len(ranges)-1:
		for k in range(len(ranges[i])):
			for j in range(i+1, len(ranges)):
				testX[j][:, lastColumnNumber] = testY_bid_temp
				testX[j][:, lastColumnNumber+1] = testY_ask_temp
			lastColumnNumber = lastColumnNumber + 2

prediction = pd.DataFrame.from_csv('../predictions/template_prediction.csv')
index = 0
for ix, row in test_table.iterrows():
	row_id = row['row_id']

	index_in_pred = prediction[prediction['row_id'] == row_id].index.tolist()[0]

	for i in range(len(ranges)):
		bid = testY_bid[i][index]
		ask = testY_ask[i][index]

		for k in ranges[i]:
			prediction.set_value(index_in_pred, 'bid' + str(k), bid)
			prediction.set_value(index_in_pred, 'ask' + str(k), ask)
	index = index+1

prediction.to_csv('../predictions/LR_timeslots.csv')

print "Done..."