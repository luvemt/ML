class SGD (object):
    def __init__(self, eta =0.1, n_iter = 10, random_state =1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        


    def shuffle(self, X,y):
        r = self.rgen.permutation(len(y))
        return X[r], y[r]

    def fit(self, X, y):
         rgen = np.random.RandomState(self.random_state)
         self.w_ = rgen.normal(loc=0.0, scale=0.01, size = 1+X.shape[1])

         self.cost_ = []
         for i in range(self.n_iter):
             X,y = self.shuffle(X,y)
             cost = []
             for xi, target in zip(X,y):
                 cost.append(self.update_weights(xi, target))
             avg_cost = sum(cost) / len(y)
             self.cost_.appen(avg_cost)
         return self


    def update_weights(self, xi, target):
        output = np.dot(xi, self.w_[1:]) + self.w_[0]
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error ** 2
        return cost

    def predict(self, X):
        z = np.dot(X, self.w_[1:]) + self.w_[0]
        return np.where(x >= 0.0, 1, -1)

                             
