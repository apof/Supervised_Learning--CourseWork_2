import matplotlib.pyplot as plt
import numpy as np


class KernelPerceptron:

	def __init__(self):

		self.alpha_weights = None
		self.weights = None
		self.kernel_type = None
		self.history_points = None

	def polynomial_kernel(self,x,y,p = 3):
		return (np.dot(x,y))**p

	def gaussian_kernel(self,x,y,sigma = 0.5):
		return np.exp(-np.linalg.norm(x-y)**2/(2*(sigma**2)))

	def fit(self,training_data,training_labels,kernel_type):

		self.alpha_weights = [0]
		self.history_points = [training_data[0]]
		#self.kernel_type = kernel_type

		# begin receiving points
		for i in range(1,training_data.shape[0]):

			new_point = training_data[i]
			self.history_points.append(new_point)

			#print("Received new point: " + str(i))

			## for all the points before the received point
			result = []
			temp_list = []
			for j in range(0,i):
				temp_list.append(j)
				result.append(self.alpha_weights[j] * self.gaussian_kernel(training_data[j],new_point))
				#print(str(self.alpha_weights[j]) + "   " + str(self.gaussian_kernel(training_data[j],new_point)))
			#print("check this point with: " + str(temp_list))
			#print("Results: " + str(result))
			prediction = np.sign(sum(result))

			if(prediction != training_labels[i]):
				self.alpha_weights.append(training_labels[i])
			else:
				self.alpha_weights.append(0)

		#print(self.alpha_weights)

		return


	def predict(self,test_data):

		predictions = []

		for i in range(0,test_data.shape[0]):
			test_point = test_data[i]
			## for all the points that we have already seen
			sum_result = 0
			for j in range(0,len(self.alpha_weights)):
				sum_result += self.alpha_weights[j] * self.gaussian_kernel(self.history_points[j],test_point)
			predictions.append(sum_result)

		return np.sign(predictions),predictions


