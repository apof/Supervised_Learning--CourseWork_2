import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import multi_dot
import utilities


class KernelPerceptron:

	def __init__(self,params):

		self.alpha_weights = None
		self.weights = None
		self.kernel_type = None
		self.history_points = None
		self.kernel_parameter = params[1]
		self.kernel_type = params[0]

	## use a specific type of kernel function that we want
	def kernel_function(self,x,y):
		if(self.kernel_type == 'gaussian'):
			return self.gaussian_kernel(x,y,self.kernel_parameter)
		elif(self.kernel_type == 'polynomial'):
			return self.polynomial_kernel(x,y,self.kernel_parameter)

	def Kernel_Matrix(self,x,y):
		if(self.kernel_type == 'gaussian'):
			return self.gaussian_kernel_vectorised(x,y,self.kernel_parameter)
		elif(self.kernel_type == 'polynomial'):
			return self.polynomial_kernel_vectorised(x,y,self.kernel_parameter)

	def polynomial_kernel_vectorised(self,x,y,p):
		K_matrix = (np.dot(x,y.T))**p
		return K_matrix

	def gaussian_kernel_vectorised(self,x,y,gamma,var=1):
		x = np.array(x)
		y = np.array(y)

		X_norm = np.sum(x**2,axis = -1)
		Y_norm = np.sum(y**2,axis = -1)
		K_matrix = var*np.exp(-gamma*(X_norm[:,None] + Y_norm[None,:] - 2*np.dot(x,y.T)))
		return K_matrix

	def polynomial_kernel(self,x,y,p):
		return (np.dot(x,y))**p

	def gaussian_kernel(self,x,y,sigma):
		return np.exp(-np.linalg.norm(x-y)**2/(2*(sigma**2)))

	def naive_fit(self,training_data,training_labels):

		self.alpha_weights = [0]
		self.history_points = [training_data[0]]

		# begin receiving points
		for i in range(1,training_data.shape[0]):

			new_point = training_data[i]
			self.history_points.append(new_point)

			result = []
			temp_list = []
			for j in range(0,i):
				temp_list.append(j)
				result.append(self.alpha_weights[j] * self.kernel_function(training_data[j],new_point))
			prediction = np.sign(sum(result))

			if(prediction != training_labels[i]):
				self.alpha_weights.append(training_labels[i])
			else:
				self.alpha_weights.append(0)

		return

	def fit_2(self,training_data,training_labels):


		self.alpha_weights = [0]
		self.history_points = [training_data[0]]

		Kernel_Matrix = self.Kernel_Matrix(training_data,training_data)

		result = 0

		for i in range(1,training_data.shape[0]):

			new_point = training_data[i]
			self.history_points.append(new_point)

			result += np.dot(np.array(self.alpha_weights),Kernel_Matrix[i][0:i])
			prediction = np.sign(result)

			if(prediction != training_labels[i]):
				self.alpha_weights.append(training_labels[i])
			else:
				self.alpha_weights.append(0)

	def fit(self,training_data,training_labels):

		## split training data into training and validation set
		## validation set helps us determine when to stop our training

		validation_data = training_data[int(0.8*training_data.shape[0]):]
		validation_labels = training_labels[int(0.8*training_data.shape[0]):]

		training_labels = training_labels[0:int(0.8*training_data.shape[0])]
		training_data = training_data[0:int(0.8*training_data.shape[0])]
		
		self.alpha_weights = np.zeros((training_data.shape[0]))
		self.history_points = training_data
		self.history_labels = training_labels

		Kernel_Matrix = self.Kernel_Matrix(training_data,training_data)

		valid_list = []

		condition = True
		epoch = 0
		while (condition == True):

			#print("Training for epoch " + str(epoch))
			for j in range(training_data.shape[0]):
				if(np.sign((self.alpha_weights*training_labels).dot(Kernel_Matrix[:,j])) != training_labels[j]):
					self.alpha_weights[j] += 1

			## evaluate for the training and the validation set at the end of each epoch

			train_preds,_ = self.predict(training_data)
			valid_preds,_ = self.predict(validation_data)

			misclas_train_points = np.sum(np.array((train_preds != training_labels)))
			misclas_valid_points = np.sum(np.array((valid_preds != validation_labels)))

			print("Training and Validation Loss: " + str(misclas_train_points) + " " +  str(misclas_valid_points))

			### stop training if validation accuracy has not been improved for three epochs
			valid_list.append(misclas_valid_points)

			if(len(valid_list) >= 3):
				print()
				if((valid_list[epoch] <= valid_list[epoch - 1]) and (valid_list[epoch - 1] <= valid_list[epoch - 2]) and (valid_list[epoch - 2] <= valid_list[epoch - 3])):
					condition = False
					print("Terminating training at epoch " + str(epoch+1) + " because no validation loss improvement considered for the last three epochs")

			epoch += 1


	def compute_kernel_values(self,epoch_index,sample_index,Kernel_Matrix):

		Kernel_List = []
		for i in range(epoch_index):
			Kernel_List.append(Kernel_Matrix[sample_index][:])

		Kernel_List.append(Kernel_Matrix[sample_index][0:sample_index])

		return np.concatenate(Kernel_List)

	def fit_3(self,training_data,training_labels):

		self.alpha_weights = []

		Kernel_Matrix = self.Kernel_Matrix(training_data,training_data)

		self.history_points = training_data

		epochs = 1

		result = 0

		for epoch_index in range(epochs):
			self.epochs = epoch_index + 1
			#print("Training for epoch: " + str(epoch_index + 1))
			for i in range(0,training_data.shape[0]):

				if (i==0):
					self.alpha_weights.append(0)
				else:
					result += np.dot(np.array(self.alpha_weights),self.compute_kernel_values(epoch_index,i,Kernel_Matrix))
					prediction = np.sign(result)

					if(prediction != training_labels[i]):
						self.alpha_weights.append(training_labels[i])
					else:
						self.alpha_weights.append(0)

			#epoch_predictions,_ = self.predict(training_data)
			#error = (epoch_predictions != training_labels).mean()
			#print("Error of epoch " + str(epoch_index + 1) + " is " + str(error))

	def predict(self,test_data):

		Kernel_Matrix = self.Kernel_Matrix(self.history_points,test_data)

		predictions = []

		for i in range(test_data.shape[0]):
			predictions.append((self.alpha_weights*self.history_labels).dot(Kernel_Matrix[:,i].T))

		return np.sign(predictions),predictions


	def naive_predict(self,test_data):

		predictions = []

		## for all the points we need to test
		for i in range(0,test_data.shape[0]):
			test_point = test_data[i]
			## for all the points that we have already seen
			sum_result = 0
			for j in range(0,len(self.alpha_weights)):
				sum_result += self.alpha_weights[j] * self.kernel_function(self.history_points[j],test_point)
			predictions.append(sum_result)

		return np.sign(predictions),predictions


	def predict_2(self,test_data):
		Kernel_Matrix = self.Kernel_Matrix(self.history_points,test_data)
		predictions = np.dot(self.alpha_weights,Kernel_Matrix)
		return np.sign(predictions),predictions

	def predict_3(self,test_data):

		Kernel_Matrix = self.Kernel_Matrix(self.history_points,test_data)

		predictions = np.zeros(test_data.shape[0])

		start = 0
		end = self.history_points.shape[0]
		for epoch_index in range(self.epochs):
			preds_of_epoch = np.dot(self.alpha_weights[start:end],Kernel_Matrix)

			predictions = np.add(predictions,preds_of_epoch)
			start += self.history_points.shape[0]
			end += self.history_points.shape[0]

		return np.sign(predictions),predictions