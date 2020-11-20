import numpy as np
import matplotlib.pyplot as plt

def calculate_accuracy(predictions,labels):
	return (1-((predictions != labels).sum())/len(labels))*100

### implementation of random train-test split
def data_split(Data,Labels,test_size):

	### create a random permuation of the data indexes
	indexes = np.random.permutation(len(Data))
	## define the test instances number
	test_instances = int(test_size*len(Data))


	train_data = []
	test_data = []

	train_labels = []
	test_labels = []
	## parse each datum and place it on the train or test set
	## taking into account its randomly computed permutation index
	for i in range(0,len(Data)):
		if(i > test_instances):
			train_data.append(np.array(Data[indexes[i]]))
			train_labels.append(np.array(Labels[indexes[i]]))
		else:
			test_data.append(np.array(Data[indexes[i]]))
			test_labels.append(np.array(Labels[indexes[i]]))

	return np.array(train_data), np.array(test_data),np.array(train_labels), np.array(test_labels)

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


## read a given data file and convert it into np array
def read_data(path_file,image_width,image_height):

	inputs = []
	labels = []

	try:
		fp = open(path_file)
		line = fp.readline()
		while line:
			line = fp.readline().strip().split(" ")
	
			if(len(line) < 3):
				break
			## the first element refers to the label
			labels.append(int(float(line[0])))

			res = np.zeros(image_width*image_height)

			## collect each pixel of each image
			for index in range(1,len(line)):
				res[index-1] = line[index]
			inputs.append(res)
	finally:
		fp.close()

	return np.array(inputs), np.array(labels)


def visualise_image(array,width,height):
	plt.imshow(np.reshape(array,(height,width)), cmap='gray')
	plt.show()


def collect_each_class_images(images,labels,class_number):

	dataset_per_num = {}
	for number in range(class_number):
		dataset_per_num[number] = []


	for i in range(len(images)):
		dataset_per_num.get(labels[i]).append(images[i])

	for key in dataset_per_num:
		print(str(len(dataset_per_num.get(key))) + " images of class " + str(key) + " found") 

	return dataset_per_num

## suffle data and labels retaining their order
def unison_shuffled_copies(a, b):
	assert len(a) == len(b)
	p = np.random.permutation(len(a))
	return a[p], b[p]

def create_1_VS_others_dataset(class_one,data_dictionary):

	class_one_data = data_dictionary.get(class_one)
	class_one_labels = [1]*len(class_one_data)

	others_dataset = []
	others_labels = []
	for key in data_dictionary:
		if(key!=class_one):
			for image in data_dictionary.get(key):
				others_dataset.append(image)
				others_labels.append(1)

	## all the labels we collected are different from class_one
	others_labels = [i * (-1) for i in others_labels]

	concatenated_data = class_one_data + others_dataset
	concatenated_labels = class_one_labels + others_labels

	return unison_shuffled_copies(np.array(concatenated_data),np.array(concatenated_labels))


def create_1_VS_1_dataset(data_dictionary):

	pair_datasets = {}

	for class_1 in data_dictionary:
		for class_2 in data_dictionary:
			if(pair_datasets.get((class_2,class_1)) == None and class_1!=class_2): 
				concat_labels = [1]*len(data_dictionary.get(class_1)) + [-1]*len(data_dictionary.get(class_2))
				concat_data = data_dictionary.get(class_1) + data_dictionary.get(class_2)
				pair_datasets[(class_1,class_2)] = (unison_shuffled_copies(np.array(concat_data),np.array(concat_labels)))

	return pair_datasets