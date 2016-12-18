#Removes the ground truth columns from the test set
#For any problem consult notebook: Remove_Ground_Truth_From_Test.ipynb

import pandas as pd

test_1 = pd.DataFrame.from_csv('../data/subset_test.csv')
test_2 = pd.DataFrame.from_csv('../data/subset_test_OHE.csv')

columnsToDelete = []

for i in range(52, 101):
	for column in test_1.columns.values:
		if column.endswith(str(i)):
			columnsToDelete.append(column)

for column in columnsToDelete:
	test_1 = test_1.drop(column, 1)
	test_2 = test_2.drop(column, 1)

test_1.to_csv('../data/subset_test.csv')
test_2.to_csv('../data/subset_test_OHE.csv')