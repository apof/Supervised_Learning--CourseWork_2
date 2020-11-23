import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import math

class AdaBoost:

	def __init__(self,params):

		## weak learners and their weights
		self.stumps = []
		self.stump_weights = []

		## sample weights
		self.weights = None
		
		self.Boosting_Rounds = params[0]

		## the response of the strong learner performance of every round
		self.strong_learner_response = None

		## saved for the final visualisation of the learning process
		self.error_number = []
		self.loss = []


	def plot_learning_process(self):

		loss_plot=plt.plot(range(self.Boosting_Rounds),self.loss ,label="Loss per Boosting Round")
		error_plot=plt.plot(range(self.Boosting_Rounds),self.error_number,label="Error Number per Boosting Round")
		plt.legend()
		plt.title('Loss and Classification Points Per Boosting round')

		plt.show()


	def compute_overall_response(self,boosting_round,training_labels,strong_learner):
			
		self.strong_learner_response = np.add(self.strong_learner_response,strong_learner)
            
		error = 0
		for column in range(training_labels.shape[0]):
			if(np.dot(self.strong_learner_response[column],training_labels[column])<0):
				error += 1
				
		self.error_number.append(error)


	## compute the loss for the current round
	def compute_current_loss(self,boosting_round,training_labels):
		self.loss.append(np.sum(np.exp((-training_labels*self.strong_learner_response[:]))))

	def fit(self,training_data,training_labels):

		self.sample_weights = np.zeros((self.Boosting_Rounds, training_data.shape[0]))
		self.strong_learner_response = np.zeros(training_data.shape[0])


		self.sample_weights[0] = np.ones(shape=training_data.shape[0]) / training_data.shape[0]

		for t in range(self.Boosting_Rounds):

			curr_sample_weights = self.sample_weights[t]
			stump = DecisionTreeClassifier(max_depth=1, max_leaf_nodes=2)
			stump = stump.fit(training_data, training_labels, sample_weight=curr_sample_weights)
			

			weak_learner_of_round = stump.predict(training_data)
			error = curr_sample_weights[(weak_learner_of_round != training_labels)].sum()

			if(error < 1/2):

				## the weight of the weak learner of the round
				a = (1/2)*np.log((1 - error)/error)

				## calculate and update the new weights
				new_sample_weights = (curr_sample_weights * np.exp(-a * training_labels * weak_learner_of_round))
				new_sample_weights /= (2*np.sqrt(error*(1-error)))
				#normalisation_constant_z = np.sum(new_sample_weights)
				#new_sample_weights /= normalisation_constant_z

				# If not final iteration, update sample weights for t+1
				if t+1 < self.Boosting_Rounds:
					self.sample_weights[t+1] = new_sample_weights

				# save results of iteration
				self.stumps.append(stump)
				self.stump_weights.append(a)

				self.compute_overall_response(t,training_labels,a*weak_learner_of_round)
				self.compute_current_loss(t,training_labels)

				#print("Boosting round: " + str(t) + " with error: " + str(error) + " and loss: " + str(self.loss[t]))

			else:
				print("Boosting Terminated Because error was > 1/2")
				return

	def predict(self,testing_data):

		prediction_stumps = []
		for prediction_stump in self.stumps:
			prediction_stumps.append(prediction_stump.predict(testing_data))

		confidence = np.dot(self.stump_weights,prediction_stumps)

		return np.sign(confidence),confidence