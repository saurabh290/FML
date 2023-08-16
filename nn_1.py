import sys
import os
import numpy as np
import pandas as pd
from scipy import stats

import matplotlib.pyplot as plt

# The seed will be fixed to 42 for this assigmnet.
np.random.seed(42)

NUM_FEATS = 90
from sklearn.decomposition import PCA

class Net(object):
	'''
	'''

	def __init__(self, num_layers, num_units):
		'''
		Initialize the neural network.
		Create weights and biases.

		Here, we have provided an example structure for the weights and biases.
		It is a list of weight and bias matrices, in which, the
		dimensions of weights and biases are (assuming 1 input layer, 2 hidden layers, and 1 output layer):
		weights: [(NUM_FEATS, num_units), (num_units, num_units), (num_units, num_units), (num_units, 1)]
		biases: [(num_units, 1), (num_units, 1), (num_units, 1), (num_units, 1)]

		Please note that this is just an example.
		You are free to modify or entirely ignore this initialization as per your need.
		Also you can add more state-tracking variables that might be useful to compute
		the gradients efficiently.


		Parameters
		----------
			num_layers : Number of HIDDEN layers.
			num_units : Number of units in each Hidden layer.
		'''
		self.num_layers = num_layers
		self.num_units = num_units

		self.biases = []
		self.weights = []

		for i in range(num_layers):

			if i==0:
				# Input layer
				self.weights.append(np.random.uniform(-1, 1, size=(NUM_FEATS, self.num_units)))
			else:
				# Hidden layer
				self.weights.append(np.random.uniform(-1, 1, size=(self.num_units, self.num_units)))

			self.biases.append(np.random.uniform(-1, 1, size=(self.num_units, 1)))

		# Output layer
		self.biases.append(np.random.uniform(-1, 1, size=(1, 1)))
		self.weights.append(np.random.uniform(-1, 1, size=(self.num_units, 1)))

	def __call__(self, X):
		'''
		Forward propagate the input X through the network,
		and return the output.

		Note that for a classification task, the output layer should
		be a softmax layer. So perform the computations accordingly

		Parameters
		----------
			X : Input to the network, numpy array of shape m x d
		Returns
		----------
			y : Output of the network, numpy array of shape m x 1
		'''
		a = X
		self.h_states = []
		self.a_states = []
		for i, (w, b) in enumerate(zip(self.weights, self.biases)):

			if i == 0:
				self.h_states.append(a)  # For input layer, both h and a are same
			else:
				self.h_states.append(h)
			self.a_states.append(a)

			h = np.dot(a, w) + b.T


			if i < len(self.weights) - 1:
				a = relu(h)
			else:  # No activation for the output layer
				a = h
			# print("a: ", a)
		self.pred = a
		# print("End")
		return self.pred

	def backward(self, X, y, lamda):
		'''
		Compute and return gradients loss with respect to weights and biases.
		(dL/dW and dL/db)

		Parameters
		----------
			X : Input to the network, numpy array of shape m x d
			y : Output of the network, numpy array of shape m x 1
			lamda : Regularization parameter.

		Returns
		----------
			del_W : derivative of loss w.r.t. all weight values (a list of matrices).
			del_b : derivative of loss w.r.t. all bias values (a list of vectors).

		Hint: You need to do a forward pass before performing backward pass.
		'''
		d_weights = []
		d_biases = []

		d_h_states = []
		d_a_states = []

		dLossdPred = (self.pred - y)
		batch_size = y.shape[0]
		# print(len(self.a_states))
		index = self.num_layers
		#         print(self.pred)
		for i in range(self.num_layers + 1):
			if i == 0:
				d_h_states.insert(0, dLossdPred)
				d_a_states.insert(0, dLossdPred)
			else:
				d_a_states.insert(0, np.dot(d_h_states[0], self.weights[index + 1].T))
				d_h_states.insert(0, d_a_states[0] * d_relu(self.a_states[index + 1]))
			d_weights.insert(0,  ((1/batch_size) * np.dot(self.a_states[index].T, d_h_states[0])) +   lamda * (self.weights[index] )  )  #+  lamda * self.weights[index]
			d_biases.insert(0, ((1/batch_size) * np.array(np.sum(d_h_states[0], axis=0).T)) )
			d_biases[0] = d_biases[0].reshape(d_biases[0].shape[0], 1)
			d_biases[0] = d_biases[0] + lamda * (self.biases[index] )
			index = index - 1
		return d_weights, d_biases

def relu(X):
	a = np.maximum(X,0)
	return a

def d_relu(X):
	X[X <= 0] = 0
	X[X > 0] = 1
	return X


def sigm(X):
	a = 1 / (1 + np.exp(-X))
	return a


def d_sigm(X):
	return X * (1 - X)

class Optimizer(object):
	'''
	'''

	def __init__(self, learning_rate, optimizer_type):
		'''
		Create a Gradient Descent based optimizer with given
		learning rate.

		Other parameters can also be passed to create different types of
		optimizers.

		Hint: You can use the class members to track various states of the
		optimizer.
		'''
		self.learning_rate = learning_rate
		self.optimizer_type = optimizer_type

	def SGD_MR(self, beta, num_layers, num_units):
		self.beta = beta
		self.Vt = []
		self.Bt = []
		for i in range(num_layers):

			if i==0:
				# Input layer
				self.Vt.append(np.zeros(((NUM_FEATS, num_units))))
			else:
				# Hidden layer
				self.Vt.append(np.zeros((num_units, num_units)))

			self.Bt.append(np.zeros((num_units, 1)))

		# Output layer
		self.Bt.append(np.zeros((1, 1)))
		self.Vt.append(np.zeros((num_units, 1)))

	def SGD_ADAM(self, beta1,beta2, num_layers, num_units):
		self.beta1 = beta1
		self.beta2 = beta2
		self.Vt = []
		self.Vt2=[]
		self.Bt = []
		self.Bt2 = []
		for i in range(num_layers):

			if i==0:
				# Input layer
				self.Vt.append(np.zeros(((NUM_FEATS, num_units))))
				self.Vt2.append(np.zeros(((NUM_FEATS, num_units))))

			else:
				# Hidden layer
				self.Vt.append(np.zeros((num_units, num_units)))
				self.Vt2.append(np.zeros((num_units, num_units)))

			self.Bt.append(np.zeros((num_units, 1)))
			self.Bt2.append(np.zeros((num_units, 1)))

		# Output layer
		self.Bt.append(np.zeros((1, 1)))
		self.Bt2.append(np.zeros((1, 1)))
		self.Vt.append(np.zeros((num_units, 1)))
		self.Vt2.append(np.zeros((num_units, 1)))



	def step(self, weights, biases, delta_weights, delta_biases):
		'''
		Parameters
		----------
			weights: Current weights of the network.
			biases: Current biases of the network.
			delta_weights: Gradients of weights with respect to loss.
			delta_biases: Gradients of biases with respect to loss.
		'''
		if self.optimizer_type == "SGD":
			for index in range(len(weights)):
				weights[index] = weights[index] - self.learning_rate * delta_weights[index]
				biases[index] = biases[index] - self.learning_rate * delta_biases[index]

		elif self.optimizer_type == "SGDMomentum":
			for index in range(len(weights)):
				self.Vt[index] = self.beta * self.Vt[index] + (1 - self.beta) * delta_weights[index]
				self.Bt[index] = self.beta * self.Bt[index] + (1 - self.beta) * delta_biases[index]
				weights[index] = weights[index] - self.learning_rate * self.Vt[index]
				biases[index] = biases[index] - self.learning_rate * self.Bt[index]

		elif self.optimizer_type == "RMSProp":
			for index in range(len(weights)):
				self.Vt[index] = self.beta * self.Vt[index] + (1 - self.beta) * np.square(delta_weights[index])
				self.Bt[index] = self.beta * self.Bt[index] + (1 - self.beta) * np.square(delta_biases[index])
				weights[index] = weights[index] - (self.learning_rate / (np.sqrt(self.Vt[index] + 0.00000001))) * (delta_weights[index] )
				biases[index] = biases[index] - (self.learning_rate / (np.sqrt(self.Bt[index] + 0.00000001))) * (delta_biases[index])
			# print(self.Vt)
			# print(self.Bt)

		elif self.optimizer_type == "ADAM":
			for index in range(len(weights)):
				self.Vt[index] = self.beta1 * self.Vt[index] + (1 - self.beta1) * delta_weights[index]
				self.Bt[index] = self.beta1 * self.Bt[index] + (1 - self.beta1) * delta_biases[index]

				self.Vt2[index] = self.beta2 * self.Vt2[index] + (1 - self.beta2) * np.square(delta_weights[index])
				self.Bt2[index] = self.beta2 * self.Bt2[index] + (1 - self.beta2) * np.square(delta_biases[index])
				weights[index] = weights[index] - self.learning_rate * (self.Vt[index] / (np.sqrt(self.Vt2[index] + 0.000001)))
				biases[index] = biases[index] - self.learning_rate * (self.Bt[index]/(np.sqrt(self.Bt2[index] + 0.000001)))


		return weights, biases

def min_max_scaling(X):
	X = (X - 1922)/(2011 - 1922)
	return X

def loss_mse(y, y_hat):
	'''
	Compute Mean Squared Error (MSE) loss betwee ground-truth and predicted values.

	Parameters
	----------
		y : targets, numpy array of shape m x 1
		y_hat : predictions, numpy array of shape m x 1

	Returns
	----------
		MSE loss between y and y_hat.
	'''
	loss = (1/y.shape[0])*np.sum((y-y_hat)**2)
	return loss

def loss_regularization(weights, biases):
	'''
	Compute l2 regularization loss.

	Parameters
	----------
		weights and biases of the network.

	Returns
	----------
		l2 regularization loss 
	'''
	loss = 0
	for i in range(len(weights)):
		loss = loss + np.sum(np.square(weights[i])) + np.sum(np.square(biases[i]))
	return loss

def loss_fn(y, y_hat, weights, biases, lamda):
	'''
	Compute loss =  loss_mse(..) + lamda * loss_regularization(..)

	Parameters
	----------
		y : targets, numpy array of shape m x 1
		y_hat : predictions, numpy array of shape m x 1
		weights and biases of the network
		lamda: Regularization parameter

	Returns
	----------
		l2 regularization loss 
	'''
	loss = loss_mse(y, y_hat) + lamda * loss_regularization(weights, biases)
	return loss

def rmse(y, y_hat):
	'''
	Compute Root Mean Squared Error (RMSE) loss betwee ground-truth and predicted values.

	Parameters
	----------
		y : targets, numpy array of shape m x 1
		y_hat : predictions, numpy array of shape m x 1

	Returns
	----------
		RMSE between y and y_hat.
	'''
	loss = np.sqrt((1 / y.shape[0]) * np.sum((y - y_hat) ** 2))
	return loss


def cross_entropy_loss(y, y_hat):
	'''
	Compute cross entropy loss

	Parameters
	----------
		y : targets, numpy array of shape m x 1
		y_hat : predictions, numpy array of shape m x 1

	Returns
	----------
		cross entropy loss
	'''
	raise NotImplementedError
def train(
	net, optimizer, lamda, batch_size, max_epochs,
	train_input, train_target,
	dev_input, dev_target
):
	'''
	In this function, you will perform following steps:
		1. Run gradient descent algorithm for `max_epochs` epochs.
		2. For each bach of the training data
			1.1 Compute gradients
			1.2 Update weights and biases using step() of optimizer.
		3. Compute RMSE on dev data after running `max_epochs` epochs.

	Here we have added the code to loop over batches and perform backward pass
	for each batch in the loop.
	For this code also, you are free to heavily modify it.
	'''

	# train_loss = []
	# dev_loss = []
	m = train_input.shape[0]

	for e in range(max_epochs):
		epoch_loss = 0
		train = np.hstack((train_input, train_target))
		np.random.shuffle(train)
		train_input = train[:, : -1]
		train_target = train[:, -1]
		train_target = train_target.reshape(train_target.shape[0], 1)
		for i in range(0, m, batch_size):
			batch_input = train_input[i:i+batch_size]
			batch_target = train_target[i:i+batch_size]
			pred = net(batch_input)
			# print(pred)
			# print(loss_mse(batch_target, pred))


			# Compute gradients of loss w.r.t. weights and biases
			dW, db = net.backward(batch_input, batch_target, lamda)



			# Get updated weights based on current weights and gradients
			weights_updated, biases_updated = optimizer.step(net.weights, net.biases, dW, db)

			# Update model's weights and biases
			net.weights = weights_updated
			net.biases = biases_updated

			# Compute loss for the batch
			batch_loss = loss_fn(batch_target, pred, net.weights, net.biases, lamda)
			epoch_loss += batch_loss
		print(e, epoch_loss)

	'''
		dev_pred = net(dev_input)
		dev_rmse = rmse(dev_target, dev_pred)
		print('RMSE on dev data: {:.5f}'.format(dev_rmse))
		# print(epoch_loss / (m ))
		train_loss.append(epoch_loss)
		dev_loss.append(dev_rmse)
	'''
		# Write any early stopping conditions required (only for Part 2)
		# Hint: You can also compute dev_rmse here and use it in the early
		# 		stopping condition.

	# After running `max_epochs` (for Part 1) epochs OR early stopping (for Part 2), compute the RMSE on dev data.

	# '''
	dev_pred = net(dev_input)
	# pd.DataFrame(np.concatenate((dev_pred, dev_target), axis=1)).to_csv("dev_test.csv")

	dev_rmse = rmse(dev_target, dev_pred)
	print('RMSE on dev data: {:.5f}'.format(dev_rmse))

	'''
	plt.plot(np.arange(0, 100), train_loss, label="Train regularized MSE Loss")
	plt.savefig("train_" + str(batch_size) + ".png")
	plt.clf()
	plt.plot(np.arange(0, 100), dev_loss, label="Dev RMSE Loss")
	plt.savefig("dev_" + str(batch_size) + ".png")
	'''

def reverse_min_max(X):
	X = X * (2011 - 1922) + 1922
	return X

def get_test_data_predictions(net, inputs):
	'''
	Perform forward pass on test data and get the final predictions that can
	be submitted on Kaggle.
	Write the final predictions to the part2.csv file.

	Parameters
	----------
		net : trained neural network
		inputs : test input, numpy array of shape m x d

	Returns
	----------
		predictions (optional): Predictions obtained from forward pass
								on test data, numpy array of shape m x 1
	'''
	pred = net(inputs)
	# pred = reverse_min_max(pred)
	# pd.DataFrame(pred).to_csv("test_test.csv")
	return pred


X_mean = None
X_std = None
def normalize(X, train):
	global X_mean, X_std

	if train == True:
		X_mean = X.mean()
		X_std = X.std()
		# X = X[(np.abs(stats.zscore(X)) < 2).all(axis=1)]
		print(X.shape)
		return (X-X.mean())/(X.std())# + 0.00000001)
	else:
		return (X-X_mean)/(X_std)

def read_data():
	'''
	Read the train, dev, and test datasets
	'''
	df_train = pd.read_csv('22m0754/regression/data/train.csv')
	df_dev = pd.read_csv('22m0754/regression/data/dev.csv')
	test_input = pd.read_csv('22m0754/regression/data/test.csv')  # .to_numpy()

	train_input = normalize(df_train.iloc[:, 1:], True).to_numpy()

	test_input = normalize(test_input,False).to_numpy()
	train_target = (df_train.iloc[:, 0]).to_numpy()
	train_target = train_target.reshape(train_target.shape[0], 1)
	dev_input = normalize(df_dev.iloc[:, 1:], False).to_numpy()
	dev_target = (df_dev.iloc[:, 0]).to_numpy()
	dev_target = dev_target.reshape(dev_target.shape[0], 1)
	return train_input, train_target, dev_input, dev_target, test_input



def read_data_non_normalize():
	'''
	Read the train, dev, and test datasets
	'''
	df_train = pd.read_csv('22m0754/regression/data/train.csv')
	df_dev = pd.read_csv('22m0754/regression/data/dev.csv')
	test_input = pd.read_csv('22m0754/regression/data/test.csv')  # .to_numpy()
	
	test_input = (test_input).to_numpy()

	train_input = (df_train.iloc[:, 1:]).to_numpy()
	train_target = df_train.iloc[:, 0].to_numpy()
	train_target = train_target.reshape(train_target.shape[0], 1)
	dev_input = (df_dev.iloc[:, 1:]).to_numpy()
	dev_target = df_dev.iloc[:, 0].to_numpy()
	dev_target = dev_target.reshape(dev_target.shape[0], 1)
	return train_input, train_target, dev_input, dev_target, test_input

def main():

	# Hyper-parameters 
	# max_epochs = 50
	max_epochs = 100
	batch_size = 64
	# batch_size = 64
	learning_rate = 0.002
	num_layers = 7
	num_units = 5
	lamda = 0.32 # Regularization Parameter

	train_input, train_target, dev_input, dev_target, test_input = read_data()
	net = Net(num_layers, num_units)
	optimizer = Optimizer(learning_rate,"ADAM")
	optimizer.SGD_ADAM(0.9,0.999, num_layers, num_units)
	train(
		net, optimizer, lamda, batch_size, max_epochs,
		train_input, train_target,
		dev_input, dev_target
	)
	get_test_data_predictions(net, test_input)




	'''
		-----------SGD Mometum Settings(Before Bug Fix)-----------
			max_epochs = 32
			batch_size = 256
			learning_rate = 0.005
			num_layers = 1
			num_units = 64
			lamda = 0.0001
			Train Error 9.58754
			614813379 491.54745300170805
			RMSE on dev data: 10.70411
			B value : 0.9
			
		-----------RMSPROP------------
			max_epochs = 150
			batch_size = 256
			learning_rate = 0.005
			num_layers = 1
			num_units = 64
			lamda = 0.000 # Regularization Parameter
			optimizer = Optimizer(learning_rate,"RMSProp")
			optimizer.SGD_MR(0.999, num_layers, num_units)
			
		----------ADAM-----------------
			max_epochs = 200
			batch_size = 64
			learning_rate = 0.001
			num_layers = 1
			num_units = 64
			lamda = 0.1 # Regularization Parameter
			optimizer = Optimizer(learning_rate,"ADAM")
			optimizer.SGD_ADAM(0.9,0.999, num_layers, num_units)
			199 325451.9884324858
			RMSE on dev data: 10.94110
	'''


if __name__ == '__main__':
	main()
	# train_input = np.array([[1,2,3],[1,6,7]])
	# train_target = np.array([[12],[43]])
	# train = np.hstack((train_input, train_target))
	# print(train)
	# np.random.shuffle(train)
	# print(train)
	# train_input = train[:, : -1]
	# print(train_input)
	# train_target = train[:, -1]
	#
	# print(train_target)
	# print(np.square(np.array([[1,2,-1],[2,3,-1]])))