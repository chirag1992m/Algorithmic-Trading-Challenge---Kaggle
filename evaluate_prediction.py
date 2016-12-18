#Script to compare and calculate the RMSE
#of a prediction file
#To decode check notebook: Evaluating_Prediction.ipynb

import sys
import pandas as pd
import math

if len(sys.argv) != 2:
	raise Exception("Run the file as: evaluate_prediction.py <prediction file name in predictions directory>")

fileToCompare = sys.argv[1] + '.csv'

ground_truth = pd.DataFrame.from_csv('./predictions/ground_truth.csv')
prediction = pd.DataFrame.from_csv('./predictions/' + fileToCompare)

for column in ground_truth.columns.values:
    if not column in prediction.columns.values:
        raise Exception(column + " missing in prediction csv file " + fileToCompare)

if ground_truth.shape[0] != prediction.shape[0]:
    raise Exception("Wrong number of rows in prediction csv file " + fileToCompare)

if ground_truth.shape[1] != prediction.shape[1]:
    raise Exception("Wrong number of columns in prediction csv file " + fileToCompare)

MSE = 0.0
total_rows = ground_truth.shape[0]
current = 1
count = 0
for index, row in ground_truth.iterrows():
    row_id = row['row_id']
    prediction_row = prediction[prediction['row_id'] == row_id]
    
    if prediction_row.shape[0] != 1:
        raise Exception("Incorrect prediction for row " + str(row_id) + ", in file " + fileToCompare)
    
    for column in ground_truth.columns.values:
        if column != 'row_id':
            pred = prediction_row[column].values[0]
            truth = row[column]

            MSE += ((pred - truth)**2)
            count += 1
            
    print "%d/%d --> %f" %(current, total_rows, (MSE/count))
    current += 1

RMSE = math.sqrt(MSE/count)

print "Root-Mean-Square-Error: ", RMSE