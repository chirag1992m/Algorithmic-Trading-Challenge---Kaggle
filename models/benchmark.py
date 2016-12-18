#A benchmark model

#Simply copies takes the last quote/trade bid-ask value
#And predicts them for the next 50 time periods

import numpy as np
import pandas as pd

test_set = pd.DataFrame.from_csv('../data/subset_test.csv')

prediction = pd.DataFrame.from_csv('../predictions/template_prediction.csv')

for ix, row in prediction.iterrows():
	row_id = row['row_id']

	test_row = test_set[test_set['row_id'] == row_id]
	#Get the ask50 and bid50 in the test set
	ask = test_row['ask50'].values[0]
	bid = test_row['bid50'].values[0]

	#And copy them in the full row
	for column in prediction.columns.values:
		if column != 'row_id':
			if column.startswith('ask'):
				prediction.set_value(ix, column, ask)
			else:
				prediction.set_value(ix, column, bid)

prediction.to_csv('../predictions/benchmark.csv')
