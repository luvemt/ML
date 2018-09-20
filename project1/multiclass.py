import numpy as np
from SGD import SGD
import sys
import pandas as pnd
from sklearn.cross_validation import train_test_split

class multicl(object):
	def __init__(self):
		self.cls = []

	def fit (self,X,y):
		self.labels = np.unique(y)
		#print("Labels:", self.labels)
		for i in range(len(self.labels)):
			ytr = np.where(y==self.labels[i], 1,-1)
			cl = SGD(no_of_inputs=len(X.columns))
			cl.fit(X,ytr)
			self.cls.append(cl)

	def predict_cl(self, inputs, cl):
		summation = np.dot(inputs, cl.w_[1:]) + cl.w_[0]
		#print("summation:",summation)
		return summation
			


	def predict (self, X):
		output = []
		for i in range(len(self.labels)):
			output.append(self.predict_cl(X,self.cls[i]))
		indx = output.index(max(output))
		return self.labels[indx]
			
		
def prepare_data(path):
    df = pnd.read_csv(path, header = None)
    df=df.sample(frac=1) 	
    X = df.iloc[:, 0:len(df.columns)-1]
    y = df.iloc[:, -1]
    return X, y 


def prediction_accuracy(obj, X,y):
  
    accuracy = 0
    
    for i in range(len(y)):
    	pred= obj.predict(X.iloc[i,:])
    	#print("Comp:",str(y.iloc[i]))
    	if pred == y.iloc[i]:
            accuracy += 1
    print("Percentage accuracy: ", (accuracy / len(y)) * 100)
    

def main():
	X,y = prepare_data(sys.argv[1])
	Xtr, Xtest, ytr, ytest = train_test_split(X, y, test_size = 0.2)
	cl = multicl()
	cl.fit(Xtr,ytr)
	prediction_accuracy(cl,Xtest,ytest)

main()
	
 		
	
			

		
	
