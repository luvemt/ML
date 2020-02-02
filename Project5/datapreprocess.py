import pandas as pnd
import numpy
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import datasets

def prepare_data(path):
    if path=="iris":
        X,y = datasets.load_iris(return_X_y=True)
        
    else:   
            df = pnd.read_csv(path, sep=',')
            df=df.sample(frac=1)    
            X = df.iloc[: , :len(df.columns)-7]
            y = df.iloc[:, 28:]
            #first = y[3]   
            #y = numpy.where(y==first, 1, -1)
            
    return X, y 