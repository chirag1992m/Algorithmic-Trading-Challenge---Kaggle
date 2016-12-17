import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

#Read the whole training set.
full_data = pd.read_csv('../data/training.csv')

#Get the securities with the highest row counts
securitiesToGet = 3
topSecurities = (np.array(full_data['security_id']
	.value_counts()
	.order(ascending=False)
	.head(securitiesToGet).axes)).reshape(securitiesToGet)
print "Top 3 Securities with highest row counts: ", topSecurities

#Extract the data for these securities
subset_data = full_data[full_data['security_id'].isin(topSecurities)]

#Save the extracted data
subset_data.to_csv('../data/subset_data.csv')

#Split the new subset data into training and testing data using sklearn
labelForStratifiedSampling = subset_data['security_id'].values
train_set, test_set = train_test_split(subset_data, 
	test_size=0.25, 
	stratify=labelForStratifiedSampling)

train_set.to_csv('../data/subset_train.csv')
test_set.to_csv('../data/subset_test.csv')