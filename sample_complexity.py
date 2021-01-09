from matplotlib import pyplot as plt
import numpy as np

def least_squares(X,Y,X_test):

	W = np.dot(np.linalg.pinv(X),Y)
	prediction = np.dot(X_test,W)
	return np.sign(prediction) 

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

	return np.sign(np.dot(X_test,W))


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


def create_dataset(a,b):

	## darw m*n samples from the multinomial distribution 
	## which is a multivariate generalization of the binomial distribution
	## pvals are the probabilities of the two different outcomes
	## the first argument refers to the number of experiments
	## the distr has the form [[0,1],[0,1],[1,0],[0,1].......]
	multinomial_distr = np.random.multinomial(1,pvals=[0.5,0.5],size = a*b)

	## convert the distribution to 0,1 ---> [1,1,0,1....]
	multinomial_distr = multinomial_distr.argmax(axis=1)
	
	## convert into -1,1, 0 is converted to -1 and 1 remains 1
	X = 2*multinomial_distr  - 1

	## reshape data in order to create the input X vectors
	X = np.array(X).reshape((a,b))
	## set the labels as the first column of the X array
	Y = X[:,0]

	return X,Y


def sample_complexity(algorithm,maximum_n):


	minimum_m = []


	for n in range(1,maximum_n + 1):

		print(n, end='\r')
		samples_number = 1

		## run until finding the minimum m (number of examples) to incur
		## no more than 10% generalisatuon error (in test set on average)
		while True:
			
			error = 0

			for i in range(samples_number):

				## create training inputs and labels
				X,Y = create_dataset(samples_number,n)
				## create testing inputs and labels
				X_test,Y_test = create_dataset(n,n)

				## train the algorithm and predict with it
				prediction = algorithm(X,Y,X_test)
				error += np.sum(prediction != Y_test)/n

			error /= samples_number

			if(error <= 0.1):
				minimum_m.append(samples_number)
				break
			samples_number += 1

	plt.figure()
	plt.plot(range(1,maximum_n + 1),minimum_m)
	plt.xlabel('n (dimension)')
	plt.ylabel('m (number of patterns)')
	plt.title('sample complexity')
	plt.show()

	return minimum_m


def create_func_values(func_type,k,n):
	y = []
	for i in range(n):
		if(func_type == 'linear'):
			y.append(k*i)
		elif(func_type == 'log'):
			y.append(k*np.log(i))
		elif(func_type == 'exp'):
			y.append(k*(2**i))
	return y

def plot_complexity(k1,k2,n,func_type,algorithm,m_values):

	t = None
	if(func_type=='linear'):
		t = 'n'
	elif(func_type == 'exp'):
		t = '2^n'
	elif(func_type == 'log'):
		t = 'log(n)'

	upper_plot=plt.plot(range(n),create_func_values(func_type,k1,n),label= str(k1) + "*" + str(t))
	lower_plot=plt.plot(range(n),create_func_values(func_type,k2,n),label=str(k2) + "*" + str(t))
	complexity_plot = plt.plot(range(n),m_values ,label=algorithm + " sample_complexity")
	plt.legend()
	plt.xlabel('n (dimension)')
	plt.ylabel('m (number of patterns)')
	plt.title('Sample Complexity of ' + algorithm)

	plt.show()
