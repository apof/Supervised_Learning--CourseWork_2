import numpy as np


## given a dataset on n points and a vaalidation round
## the functions return a set of indexes related to the validation and one set related to the validation points
def cross_validation(validation_round,number_of_data,Folds):

	## define the size of each set given the total data number and the Folds we split in
	set_size = number_of_data/Folds

	validation_indexes = []
	training_indexes = []

	## define the data index where validation set starts
	validation_start = validation_round*set_size

	## define the data indexx where validation set ends
	validation_end = validation_start + set_size

	for i in range(0,number_of_data):
		if(i>=validation_start and i<validation_end):
			validation_indexes.append(i)
		else:
			training_indexes.append(i)

	return validation_indexes,training_indexes

def get_train_test_set(validation_indexes,training_indexes,data,labels):

	validation_data = data[validation_indexes[0]:validation_indexes[-1] + 1]
	validation_labels = labels[validation_indexes[0]:validation_indexes[-1] + 1]

	training_data = []
	training_labels = []

	for i in range(0,len(data)):
		if(i in training_indexes):
			training_data.append(data[i])
			training_labels.append(labels[i])

	return np.array(training_data),np.array(training_labels),np.array(validation_data),np.array(validation_labels)




