#A simple Random Forest Regressor model.
#Generates the first ask and bid which is copied
#across the all the upcoming bid and ask
#as prediction
import pandas as pd
import numpy as np
from sklearn import ensemble

import pickle

#Read the data
print "Loading data..."
train_table = pd.DataFrame.from_csv('../data/subset_train_OHE.csv')
test_table = pd.DataFrame.from_csv('../data/subset_test_OHE.csv')

#generate the preditors and prediction columns
predictionColumns = []
for i in range(52, 101):
	for column in train_table.columns.values:
		if column.endswith(str(i)) and (column.startswith('ask') or column.startswith('bid')):
			predictionColumns.append(column)

featureColumns = []
for column in train_table.columns.values:
	if ((column not in predictionColumns) and (column != 'row_id') and (not column.startswith('time'))):
		featureColumns.append(column)

#Generate the predictors from the data
trainX = np.zeros((train_table.shape[0] * 2, len(featureColumns) + 1))
trainY = np.zeros((train_table.shape[0] * 2))

testX = np.zeros((test_table.shape[0] * 2, len(featureColumns) + 1))

print "Creating Predictors..."
index = 0
for ix, row in train_table.iterrows():
	X_features = (np.array(row[featureColumns])).flatten('F')
	X = np.concatenate((X_features, np.array([0]))) #Adding the 0 for bid
	Y = row[predictionColumns[0]]
	trainX[index, :] = X
	trainY[index] = Y

	index = index+1

	X = np.concatenate((X_features, np.array([1]))) #Adding the 1 for ask
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


#Make the model and fit
print "Training..."
RF_model = ensemble.RandomForestRegressor(n_estimators=200,
	criterion='mse',
	max_depth=200,
	min_samples_split=2, 
	min_samples_leaf=1, 
	min_weight_fraction_leaf=0.0, 
	max_features='auto', 
	max_leaf_nodes=None, 
	min_impurity_split=1e-07, 
	bootstrap=True, 
	oob_score=False, 
	n_jobs=-1, 
	random_state=None, 
	verbose=1, 
	warm_start=False)
RF_model.fit(trainX, trainY)

with open('../run_models/RF.model', 'wb') as output:
	pickle.dump(LR_model, output, -1)

#Create the prediction file
print "Predicting..."
testY = RF_model.predict(testX)
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

prediction.to_csv('../predictions/RF.csv')

print "Done!"