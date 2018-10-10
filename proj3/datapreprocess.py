import pandas as pnd
import numpy
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import datasets

def prepare_data(path):
    if path=="digits":
        X,y = datasets.load_digits(return_X_y=True)
        
    else:   
            df = pnd.read_csv(path, sep=',' ,header= 17)
            df=df.sample(frac=1)    
            X = df.iloc[: , :len(df.columns)-1]
            y = df.iloc[:, -1]
            #first = y[3]   
            #y = numpy.where(y==first, 1, -1)
            
    return X, y                    


def standardize (Xtr, Xtest):
    X_tr_std = preprocessing.scale(Xtr)
    X_test_std = preprocessing.scale(Xtest)
    return X_tr_std, X_test_std



def split_data(X,y):
    X_tr, X_test, y_tr, y_test = train_test_split(X, y, test_size = 0.2)
    return X_tr, X_test, y_tr, y_test
