import sys
import pandas as pnd
import numpy
from sklearn.cross_validation import train_test_split
from perceptron1 import Perceptron
from Adaline import Adaline
from SGD import SGD
import matplotlib as mpl
import matplotlib.pyplot as plt


def prediction_accuracy(obj, X,y):
  
    accuracy = 0
    
    for i in range(len(y)):
    	pred= obj.predict(X.iloc[i,:])
    	#pred = pred[1]
    	if pred == y[i]:
            accuracy += 1
    print("Percentage accuracy: ", (accuracy / len(y)) * 100)
    

def prepare_data(path):
    df = pnd.read_csv(path, header = None)
    df=df.sample(frac=1) 	
    X = df.iloc[:, 0:len(df.columns)-1]
    y = df.iloc[:, -1]
    first = y[3]	
    y = numpy.where(y==first, 1, -1)
    #print(X)	
    return X, y                    
                          
    


def run_classifier(classifier, X, y):
  
    Xtr, Xtest, ytr, ytest = train_test_split(X, y, test_size = 0.2)
    #print("Training",Xtr)
    if classifier == "perceptron":
    	cl = Perceptron(no_of_inputs = len(X.columns))
    	cl.train(Xtr, ytr)
    	plt.plot(cl.errors)
    	plt.xlabel('no_iter')
    	plt.ylabel('errors')
    	plt.savefig('Perceptron.png')
    	prediction_accuracy(cl,Xtest, ytest)                  
                          
                        
    elif classifier == "adaline":
    	cl = Adaline()
    	cl.fit(Xtr, ytr)
    	plt.plot(cl.cost_)
    	plt.xlabel('no_iter')
    	plt.ylabel('cost')
    	plt.savefig('Adaline.png')
    	prediction_accuracy(cl,Xtest, ytest)                  


    elif classifier == "sgd":
    	cl = SGD(no_of_inputs=len(X.columns))
    	cl.fit(Xtr, ytr)
    	plt.plot(cl.cost_)
    	plt.xlabel('no_iter')
    	plt.ylabel('cost')
    	plt.savefig('SGD.png')
    	prediction_accuracy(cl, Xtest, ytest)
                          
    else:
        print("Classifier not recognized")
        sys.exit()
    

def main ():
    if len(sys.argv)!= 3:
        print ("Number of arguments should be 2")
        sys.exit()
    
    X,y = prepare_data(sys.argv[2]);
    run_classifier (sys.argv[1], X, y)

main()

    
