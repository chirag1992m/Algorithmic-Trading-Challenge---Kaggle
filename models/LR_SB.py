#Check notebook: Buyer_Seller_Separated_Modelling.ipynb

#A simple linear regression model which analzes seller 
#and buyer initiated transactions separately.
#Generates the first ask and bid which is copied
#across the all the upcoming bid and ask
#as prediction
import pandas as pd
import numpy as np
from sklearn import linear_model as lm

import pickle

#Read the data
print "Loading data..."
train_table = pd.DataFrame.from_csv('../data/subset_train_OHE.csv')
test_table = pd.DataFrame.from_csv('../data/subset_test_OHE.csv')

predictionColumns = []
for i in range(52, 101):
	for column in train_table.columns.values:
		if column.endswith(str(i)) and (column.startswith('ask') or column.startswith('bid')):
			predictionColumns.append(column)

featureColumns = []
columnsToIgnore = ['row_id', 'is_seller', 'is_buyer']
for column in train_table.columns.values:
	if ((column not in predictionColumns) and (column not in columnsToIgnore) and (not column.startswith('time'))):
		featureColumns.append(column)

train_table_seller = train_table[train_table['is_seller'] == 1]
train_table_buyer = train_table[train_table['is_buyer'] == 1]

test_table_seller = test_table[test_table['is_seller'] == 1]
test_table_buyer = test_table[test_table['is_buyer'] == 1]

trainX_seller = np.zeros((train_table_seller.shape[0] * 2, len(featureColumns) + 1))
trainY_seller = np.zeros((train_table_seller.shape[0] * 2))
trainX_buyer = np.zeros((train_table_buyer.shape[0] * 2, len(featureColumns) + 1))
trainY_buyer = np.zeros((train_table_buyer.shape[0] * 2))

testX_seller = np.zeros((test_table_seller.shape[0] * 2, len(featureColumns) + 1))
testX_buyer = np.zeros((test_table_buyer.shape[0] * 2, len(featureColumns) + 1))

print "Creating Predictors..."
index = 0
for ix, row in train_table_seller.iterrows():
	X_features = (np.array(row[featureColumns])).flatten('F')
	X = np.concatenate((X_features, np.array([0])))
	Y = row[predictionColumns[0]]
	trainX_seller[index, :] = X
	trainY_seller[index] = Y

	index = index+1

	X = np.concatenate((X_features, np.array([1])))
	Y = row[predictionColumns[1]]
	trainX_seller[index, :] = X
	trainY_seller[index] = Y
	index = index+1

index = 0
for ix, row in train_table_buyer.iterrows():
	X_features = (np.array(row[featureColumns])).flatten('F')
	X = np.concatenate((X_features, np.array([0])))
	Y = row[predictionColumns[0]]
	trainX_buyer[index, :] = X
	trainY_buyer[index] = Y

	index = index+1

	X = np.concatenate((X_features, np.array([1])))
	Y = row[predictionColumns[1]]
	trainX_buyer[index, :] = X
	trainY_buyer[index] = Y
	index = index+1

index = 0
for ix, row in test_table_seller.iterrows():
	X_features = (np.array(row[featureColumns])).flatten('F')
	X = np.concatenate((X_features, np.array([0])))
	testX_seller[index, :] = X

	index = index+1

	X = np.concatenate((X_features, np.array([1])))
	testX_seller[index, :] = X
	index = index+1

index = 0
for ix, row in test_table_buyer.iterrows():
	X_features = (np.array(row[featureColumns])).flatten('F')
	X = np.concatenate((X_features, np.array([0])))
	testX_buyer[index, :] = X

	index = index+1

	X = np.concatenate((X_features, np.array([1])))
	testX_buyer[index, :] = X
	index = index+1

print "Training..."
LR_model_seller = lm.LinearRegression(fit_intercept=True, normalize=False, n_jobs=1)
LR_model_seller.fit(trainX_seller, trainY_seller)

LR_model_buyer = lm.LinearRegression(fit_intercept=True, normalize=False, n_jobs=1)
LR_model_buyer.fit(trainX_buyer, trainY_buyer)

models = {'seller': LR_model_seller, 'buyer': LR_model_buyer}

with open('../run_models/LR_SB.model', 'wb') as output:
	pickle.dump(models, output, -1)

print "Predicting..."
testY_seller = LR_model_seller.predict(testX_seller)
testY_buyer = LR_model_buyer.predict(testX_buyer)

prediction = pd.DataFrame.from_csv('../predictions/template_prediction.csv')

i = 0
for ix, row in test_table_seller.iterrows():
	row_id = row['row_id']

	index_in_pred = prediction[prediction['row_id'] == row_id].index.tolist()[0]

	bid = testY_seller[i]
	i = i+1
	ask = testY_seller[i]
	i = i+1

	for column in predictionColumns:
		if column.startswith('bid'):
			prediction.set_value(index_in_pred, column, bid)
		else:
			prediction.set_value(index_in_pred, column, ask)

i = 0
for ix, row in test_table_buyer.iterrows():
	row_id = row['row_id']

	index_in_pred = prediction[prediction['row_id'] == row_id].index.tolist()[0]

	bid = testY_buyer[i]
	i = i+1
	ask = testY_buyer[i]
	i = i+1

	for column in predictionColumns:
		if column.startswith('bid'):
			prediction.set_value(index_in_pred, column, bid)
		else:
			prediction.set_value(index_in_pred, column, ask)

prediction.to_csv('../predictions/LR_SB.csv')

print "Done!"