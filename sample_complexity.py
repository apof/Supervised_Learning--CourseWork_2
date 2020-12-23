from matplotlib import pyplot as plt
import numpy as np

def least_squares(X,Y,X_test):

	W = np.dot(np.linalg.pinv(X),Y)
	prediction = np.dot(X_test,W) 
	return 2*(prediction > 0) - 1

def one_nn(X,Y,X_test):

	predictions = np.zeros(X_test.shape[0])

	for i in range(X_test.shape[0]):
		distance = float('inf')
		for j in range(X.shape[0]):
			dist = np.count_nonzero(X_test[i] != X[j])
			if(dist < distance):
				distance = dist
				predictions[i] = Y[j]

	return predictions



def perceptron(X,Y,X_test,threshold = 0.001):

	## initialise weight vector
	W = np.zeros(X.shape[1])
	## initialise error with infinity
	error = float('inf')

	## until convergence
	while(error > threshold):
		error = 0
		for i in range(X.shape[0]):
			prediction = 2*(np.dot(W,X[i]) > 0) - 1

			if(prediction != Y[i]):

				error += 1
				W += Y[i]*X[i]

		error /= X.shape[0]

	## return the prediction based on the obtained weights
	return 2*(np.dot(X_test,W) > 0) - 1


def winnow(X,Y,X_test,threshold = 0.001):

	X = (X+1)/2
	Y = (Y+1)/2
	X_test = (X_test+1)/2

	W = np.ones(X.shape[1])
	error = float('inf')

	while (error > threshold):
		error = 0
		for i in range(X.shape[0]):
			prediction = (np.dot(W,X[i]) >= X.shape[1])

			if(prediction!=Y[i]):
				error += 1
				W *= 2 ** ((Y[i] - prediction) * X[i])
		error /= X.shape[0]

	return 2 * (np.dot(X_test,W) >= X.shape[1]) - 1


def sample_complexity(algorithm,maximum_n):


	## array to store th sample complexity
	minimum_m = np.zeros(maximum_n)

	for n in range(1,maximum_n + 1):
		print(n)
		m = 1
		## run until finding the minimum m (number of examples) to incur
		## no more than 10% generalisatuon error (in test set on average)
		while True:
			samples_number = m
			error = 0

			for i in range(samples_number):

				X = 2 * np.random.multinomial(1,pvals=[0.5,0.5],size = m*n).argmax(axis=1) - 1
				X = np.array(X).reshape((m,n))
				Y = X[:,0]

				X_test = 2 * np.random.multinomial(1,pvals=[0.5,0.5],size = n*n).argmax(axis=1) - 1
				X_test = np.array(X_test).reshape((n,n))
				Y_test = X_test[:,0]
         
				prediction = algorithm(X,Y,X_test)
				error += np.count_nonzero(prediction != Y_test)/n

			error /= samples_number

			if(error <= 0.1):
				minimum_m[n-1] = m
				break
			m += 1

	plt.figure()
	plt.plot(range(1,maximum_n + 1),minimum_m)
	plt.xlabel('n (dimension)')
	plt.ylabel('m (number of patterns)')
	plt.title('sample complexity')
	plt.show()


print("Computing sample complexity!")
#sample_complexity(perceptron,100)
sample_complexity(least_squares,100)
#sample_complexity(winnow,300)
#sample_complexity(one_nn,13)