import pandas as pnd
import datetime as dt
import numpy
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import datasets

def prepare_data(path):  
    df = pnd.read_table(path, delim_whitespace=True)
    #df=df.sample(frac=1) 
    #print("Cols",len(df.columns))   
    X = df.iloc[: , :len(df.columns)-1]
    y = df.iloc[:, -1]
    #first = y[3]   
    #y = numpy.where(y==first, 1, -1)
            
    return X, y   


def prepare_cal_data(path):
    df = pnd.read_csv(path, sep=',')
    df['TIMESTAMP'] = pnd.to_datetime(df['TIMESTAMP'])
    df['TIMESTAMP']=df['TIMESTAMP'].map(dt.datetime.toordinal)
    df=df.fillna(450)
    X = df.iloc[: , :len(df.columns)-1]
    y = df.iloc[:, -1]
    #first = y[3]   
    #y = numpy.where(y==first, 1, -1)
    print(X)        
    return X, y   

     


def standardize (Xtr, Xtest):
    X_tr_std = preprocessing.scale(Xtr)
    X_test_std = preprocessing.scale(Xtest)
    return X_tr_std, X_test_std



def split_data(X,y):
    X_tr, X_test, y_tr, y_test = train_test_split(X, y, test_size = 0.2)
    return X_tr, X_test, y_tr, y_test