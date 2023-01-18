import numpy as np
from numpy import random as rnd

def random(a,b,n):
    return (a+rnd.rand(n)*(b-a))

def inlocuireNAN(X):
    #we replace the missing values with the average because we have quantitative variables
    avgs=np.nanmean(a=X,axis=0 ); #computes the mean without the missing values, axis=0 - average on columns

    pos = np.where(np.isnan(X));  #the position where i do not have a value
    #it will return 2 dimensions: lines, and columns

    X[pos]=avgs[pos[1]]; #we take the values form the columns

    return X
