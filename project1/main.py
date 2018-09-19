import sys
import pandas as pnd
import numpy
from sklearn.model_selection import train_test_split


def prediction_accuracy(obj, X, y):
    prediction = []
    accuracy = 0
    
    for i in range(len(y)):
        prediction.append(obj.predict(X[i],y[i]))
        if prediction[i] == y[i]:
            accuracy += 1
        print("Percentage accuracy: ", (accuracy / len(y)) * 100)
    

def prepare_data(path):
    df = pnd.read_csv(path)
    X = df.iloc[:, 0:len(df.columns)-1]
    y = df.iloc[:, -1]
    return X, y                    
                          
    


def run_classifier(classfier, X, y):
  
    Xtr, Xtest, ytr, ytest = train_test_split(df, y, test_size = 0.3)
    if classifier == "perceptron":
        cl = Perceptron()
        cl.fit(Xtr, ytr)
        prediction_accuracy(cl,Xtest, ytest)                  
                          
                        
    elif classifier == "adaline":
        cl = Adaline()
        cl.fit(Xtr, ytr)
        prediction_accuracy(cl,Xtest, ytest)                  


    elif classifier == "sgd":
        cl = SGD()
        cl.fit(Xtr, ytr)
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

    
