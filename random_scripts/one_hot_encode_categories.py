#This file one-hot encodes all the category data
#in the datasets which we will be using and saves them into a 
#new csv file as <old_file_name>_<OHE>.csv
#All the training will be done using the one-hot encoded datasets

#how it works can be seen in the notebook: One-Hot_encode_categorical_data.ipynb

import pandas as pd

def one_hot_encode_data(dataframe):
	#Convert security_id into one-hot encoding
	index = dataframe.columns.get_loc('security_id')
	dataframe.insert(index+1, 'is_security_18', 0)
	dataframe['is_security_18'] = dataframe['security_id'].apply(lambda x: 1 if x == 18 else 0)
	
	index = dataframe.columns.get_loc('security_id')
	dataframe.insert(index+1, 'is_security_102', 0)
	dataframe['is_security_102'] = dataframe['security_id'].apply(lambda x: 1 if x == 102 else 0)

	index = dataframe.columns.get_loc('security_id')
	dataframe.insert(index+1, 'is_security_73', 0)
	dataframe['is_security_73'] = dataframe['security_id'].apply(lambda x: 1 if x == 73 else 0)

	#And ultimately drop the initial column itself
	dataframe = dataframe.drop('security_id', 1)

	#One-Hot encode initiator
	index = dataframe.columns.get_loc('initiator')
	dataframe.insert(index+1, 'is_seller', 0)
	dataframe.insert(index+1, 'is_buyer', 0)
	dataframe['is_seller'] = dataframe['initiator'].apply(lambda x: 1 if x == 'S' else 0)
	dataframe['is_buyer'] = dataframe['initiator'].apply(lambda x: 1 if x == 'B' else 0)

	#And ultimately drop the initial column itself
	dataframe = dataframe.drop('initiator', 1)

	for i in range(1, 51):
		column_name = 'transtype' + str(i)
		new_column_name_Q = 'transtype' + str(i) + '_is_Q'
		new_column_name_T = 'transtype' + str(i) + '_is_T'

		index = dataframe.columns.get_loc(column_name)
		dataframe.insert(index+1, new_column_name_Q, 0)
		dataframe.insert(index+1, new_column_name_T, 0)

		dataframe[new_column_name_Q] = dataframe[column_name].apply(lambda x: +1 if x == 'Q' else 0)
		dataframe[new_column_name_T] = dataframe[column_name].apply(lambda x: +1 if x == 'T' else 0)

		#drop the initial column itself
		dataframe = dataframe.drop(column_name, 1)

	return dataframe

subset_data = pd.DataFrame.from_csv('../data/subset_data.csv')
subset_data = one_hot_encode_data(subset_data)
subset_data.to_csv('../data/subset_data_OHE.csv')

train_set = pd.DataFrame.from_csv('../data/subset_train.csv')
train_set = one_hot_encode_data(train_set)
train_set.to_csv('../data/subset_train_OHE.csv')

test_set = pd.DataFrame.from_csv('../data/subset_test.csv')
test_set = one_hot_encode_data(test_set)
test_set.to_csv('../data/subset_test_OHE.csv')