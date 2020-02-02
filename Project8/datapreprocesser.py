import pandas as pnd
import datetime as dt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import datasets


def prepare_data(path):  
    df = pnd.read_csv(path, header=None)
    #df=df.sample(frac=1) 
    #print("Cols",len(df.columns)) 
    df = df.replace("?", np.nan)
    #print(df.isnull().sum()) 
    df = df.fillna(0) 
    #print(df.isnull().sum()) 
    X = df.iloc[: , :len(df.columns)-1]

    y = df.iloc[:, -1]
    #first = y[3]   
    #y = numpy.where(y==first, 1, -1)
            
    return X, y   


def prepare_digits():
    #X_train, y_train = loadlocal_mnist(images_path ='MNIST/train-images-idx3-ubyte',
                        #labels_path='MNIST/train-labels-idx1-ubyte')
    #X_test, y_test = loadlocal_mnist(images_path='MNIST/t10k-images-idx3-ubyte',
                       #labels_path='MNIST/t10k-labels-idx1-ubyte')
    #return X_train, X_test, y_train, y_test
    X,y = datasets.load_digits(return_X_y=True)
    return X,y


     


def standardize (Xtr, Xtest):
    X_tr_std = preprocessing.scale(Xtr)
    X_test_std = preprocessing.scale(Xtest)
    return X_tr_std, X_test_std



def split_data(X,y):
    X_tr, X_test, y_tr, y_test = train_test_split(X, y, test_size = 0.2)
    return X_tr, X_test, y_tr, y_test