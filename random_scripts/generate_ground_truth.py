#Script to generate the ground truth file
#with which the predictions will be compared

#To decode check notebook: Evaluating_Prediction.ipynb

import pandas as pd

test_data = pd.DataFrame.from_csv('../data/subset_test.csv')

columnsToRetain = ['row_id']

for i in range(52, 101):
	for column in test_data.columns.values:
		if column.endswith(str(i)):
			columnsToRetain.append(column)

ground_truth = test_data[columnsToRetain]

ground_truth.to_csv('../predictions/ground_truth.csv')