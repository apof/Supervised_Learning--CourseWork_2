import numpy as np
import matplotlib.pyplot as plt
import AdaBoost
import KernelPerceptron

def classifier(type,hyperparams):

	if(type == 'AdaBoost'):
		return AdaBoost.AdaBoost(hyperparams)
	elif(type == 'KernelPerceptron'):
		return KernelPerceptron.KernelPerceptron(hyperparams)


def confusion_matrix(predictions,labels):

	confusion_dict = {}

	for i in range(10):
		for j in range(10):
			confusion_dict[(i,j)] = 0

	for i in range(0,len(predictions)):
		if(predictions[i]!=labels[i]):
			confusion_dict[(predictions[i],labels[i])] += 1

	for key in confusion_dict:
		print(str(key) + " " + str(confusion_dict.get(key)))



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


def one_VS_all_training(data_per_class_dictionary,class_number,algorithm,hyperparams):
    
	models_dict = {}
	for n in range(0,class_number):
		#print("Training a classifier for: " + str(n))
		dat,lbl = create_1_VS_others_dataset(n,data_per_class_dictionary)
		model = None

		model = classifier(algorithm,hyperparams)
		model.fit(dat,lbl)
            
		models_dict[n] = model
		##model.plot_learning_process()
        
	return models_dict


def one_VS_all_testing(models_dict,data,labels):
    
	models_confidence = []
	for key in models_dict:
		model = models_dict.get(key)
		confidence = None
		_,confidence = model.predict(data)        
		models_confidence.append(confidence)
        
	predictions = []
	confidence_predictions = []
	average_confidence = []
        
	for i in range(len(data)):
		results = []
		avg_conf = 0
		for j in range(len(models_confidence)):
			avg_conf += models_confidence[j][i]
			results.append((j,models_confidence[j][i]))

		avg_conf = avg_conf/len(models_confidence)

		average_confidence.append((i,avg_conf))
            
		## return as the prediction of the i-th datum the prediction of the classifier with the highest confidence
		predictions.append(sorted(results, key=lambda tup: tup[1],reverse = True)[0][0])
		confidence_predictions.append((i,(sorted(results, key=lambda tup: tup[1],reverse = True)[0][1])))
        
	return predictions,confidence_predictions,average_confidence


def one_vs_one_training(pair_datasets,algorithm,hyperparams):
    
	pairwise_models_dict = {}
    
	for  key in pair_datasets:
		(class_1,class_2) = key
        
		#print("Training a classifier for pair: " + str(class_1) + " " + str(class_2))
		model = classifier(algorithm,hyperparams)
		model.fit(pair_datasets.get(key)[0],pair_datasets.get(key)[1])
        
		pairwise_models_dict[key] = model
        
	return pairwise_models_dict



def one_vs_one_testing(pair_models,data,num_classes):
    
	predictions = []
    
	confidence_of_each_classifier = np.zeros((len(data),num_classes))
	for  key in pair_models:
		(class_1,class_2) = key
		model = pair_models.get(key)
		_,confidence = model.predict(data)
		for i in range(len(confidence)):
			if(confidence[i] > 0):
				confidence_of_each_classifier[i][class_1] += abs(confidence[i])
			else:
				confidence_of_each_classifier[i][class_2] += abs(confidence[i])
                
	for result in confidence_of_each_classifier:
		predictions.append(np.where(result == np.amax(result))[0][0])
        
	return predictions

## find the misclassified items with the highest confidence
def get_hardest_to_predict_items(confidence,average_confidence,predictions,labels):

	count = 0
	items_1 = []
	index = 0

	## sort items with the biggest confidence at the start
	## the hard items are misclassified items with big confidence
	## that means that we predict them in a very wrong way
	sorted_confidence = sorted(confidence, key=lambda tup: tup[1],reverse = True)

	while(count!=5):
		ind = sorted_confidence[index][0]
		if(predictions[ind]!=labels[ind]):
			items_1.append(ind)
			count += 1
		index += 1

	count = 0
	items_2 = []
	index = 0

	## as an alternative find the misclasified items with the lower avearge confidence among all classifiers
	sorted_average_confidence = sorted(average_confidence, key=lambda tup: tup[1],reverse = False)

	while(count!=5):
		ind = sorted_average_confidence[index][0]
		if(predictions[ind]!=labels[ind]):
			items_2.append(ind)
			count += 1
		index += 1

	return items_1,items_2

