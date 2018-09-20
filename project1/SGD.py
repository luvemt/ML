import numpy as np
class SGD (object):
    def __init__(self, no_of_inputs,eta =0.01, n_iter = 100, random_state =1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size = no_of_inputs + 1)

    def fit(self, X, y):
         

         self.cost_ = []
         for i in range(self.n_iter):
       
             cost = []
             for i in range(len(y)):
                 cost.append(self.update_weights(X.iloc[i,:], y[i]))
             avg_cost = sum(cost) / len(y)
             self.cost_.append(avg_cost)
         return self


    def update_weights(self, xi, target):
        output = np.dot(xi, self.w_[1:]) + self.w_[0]
        error = (target - output)
        self.w_[1:] += self.eta * np.dot(xi, error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error ** 2
        return cost

    def predict(self, X):
        z = np.dot(X, self.w_[1:]) + self.w_[0]
        return np.where(z >= 0.0, 1, -1)

                             
