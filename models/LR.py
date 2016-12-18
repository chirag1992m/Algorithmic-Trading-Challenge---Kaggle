import pandas as pd
import numpy as np
from sklearn import linear_model as lm

train_table = pd.DataFrame.from_csv('../data/subset_train_OHE.csv')
test_table = pd.DataFrame.from_csv('../data/subset_test_OHE.csv')

predictionColumns = []
for i in range(52, 101):
	for column in train_table.columns.values:
		if column.endswith(str(i)) and (column.startswith('ask') or column.startswith('bid')):
			predictionColumns.append(column)

featureColumns = []
for column in train_table.columns.values:
	if ((column not in predictionColumns) and (column != 'row_id') and (not column.startswith('time'))):
		featureColumns.append(column)

trainX = np.zeros((train_table.shape[0] * 2, len(featureColumns) + 1))
trainY = np.zeros((train_table.shape[0] * 2))

testX = np.zeros((test_table.shape[0] * 2, len(featureColumns) + 1))

index = 0
for ix, row in train_table.iterrows():
	X_features = (np.array(row[featureColumns])).flatten('F')
	X = np.concatenate((X_features, np.array([0])))
	Y = row[predictionColumns[0]]
	trainX[index, :] = X
	trainY[index] = Y

	index = index+1

	X = np.concatenate((X_features, np.array([1])))
	Y = row[predictionColumns[1]]
	trainX[index, :] = X
	trainY[index] = Y
	index = index+1


index = 0
for ix, row in test_table.iterrows():
	X_features = (np.array(row[featureColumns])).flatten('F')
	X = np.concatenate((X_features, np.array([0])))
	testX[index, :] = X

	index = index+1

	X = np.concatenate((X_features, np.array([1])))
	testX[index, :] = X
	index = index+1


LR_model = lm.LinearRegression(fit_intercept=True, normalize=False, n_jobs=-1)
LR_model.fit(trainX, trainY)


testY = LR_model.predict(testX)
prediction = pd.DataFrame.from_csv('../predictions/template_prediction.csv')

i = 0
for ix, row in test_table.iterrows():
	row_id = row['row_id']

	index_in_pred = prediction[prediction['row_id'] == row_id].index.tolist()[0]

	bid = testY[i]
	i = i+1
	ask = testY[i]
	i = i+1

	for column in predictionColumns:
		if column.startswith('bid'):
			prediction.set_value(index_in_pred, column, bid)
		else:
			prediction.set_value(index_in_pred, column, ask)

prediction.to_csv('../predictions/LR.csv')