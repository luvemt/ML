import numpy as np

class Perceptron(object):

	def __init__(self, no_of_inputs, threshold=10, learning_rate=0.01,random_state =1):
		self.threshold = threshold
		self.learning_rate = learning_rate
		self.random_state = random_state
		rgen = np.random.RandomState(self.random_state)
		self.weights = rgen.normal(loc=0.0, scale=0.01, size = no_of_inputs+1)
		#self.weights = np.zeros(no_of_inputs + 1)
           
	def predict(self, inputs):
		summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
		#print("summation:",summation)
		return np.where(summation >= 0.0, 1, -1)
			

	def train(self, training_inputs, labels):
		self.errors = []
		for _ in range(self.threshold):
			it_errors =0
			for i in range (len(labels)):
				prediction = self.predict(training_inputs.iloc[i,:])
				error = labels[i] - prediction
				if error != 0:
					it_errors += 1 
				self.weights[1:] += self.learning_rate * (error) * training_inputs.iloc[i,:]
				self.weights[0] += self.learning_rate * (error)
			self.errors.append(it_errors)
		return self
