#Creates an empty prediction_file so that
#Prediction tables can be easily be generated
#using this empty template

#For doubts check notebook: Evaluating_Prediction.ipynb

import pandas as pd

template_prediction = pd.DataFrame.from_csv('../predictions/ground_truth.csv')

for column in template_prediction.columns.values:
	if column != 'row_id':
		template_prediction[column] = template_prediction[column].apply(lambda x: 0.0)

template_prediction.to_csv('../predictions/template_prediction.csv')