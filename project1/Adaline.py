import numpy as np
class Adaline(object):
	def __init__(self, eta = 0.01, n_iter = 10, random_state =1):
       		self.eta = eta
        	self.n_iter = n_iter
        	self.random_state = random_state
		
	def fit(self, X,y):
		rgen = np.random.RandomState(self.random_state)
		self.w_ = rgen.normal(loc =0.0, scale =0.01, size = 1+X.shape[1])
		self.cost_ =[]
		for i in range(self.n_iter):
			output = np.dot(X, self.w_[1:]) + self.w_[0]
			errors = (y - output)
			self.w_[1:] += self.eta * X.T.dot(errors)
			self.w_[0] += self.eta * errors.sum()
			cost = (errors ** 2).sum() /2.0
			self.cost_.append(cost)
		return self


	def predict(self, X):
		z = np.dot(X, self.w_[1:]) + self.w_[0]
		return np.where(z >= 0.0, 1, -1)
