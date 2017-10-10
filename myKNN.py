# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 15:40:28 2017

@author: thanos_kats
"""
import sklearn
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from optparse import OptionParser

optParser = OptionParser()
optParser.add_option("-n", "--n_neighbors", dest="neighbors", type="int",
                  help="choose k neighbors for KNN model")
optParser.add_option("-w", "--weights", dest="weights", default='uniform',
		  help="choose weight metric for KNN model. distance/uniform")
optParser.add_option("-m", "--metric", dest="metric", default='minkowski',
                  help="choose metric for KNN model. minkowski/ euclidean/ manhattan")
optParser.add_option("-p", "--p", dest="_p", default=2, type="int",
                  help="choose p for KNN model, p = 1 metric = manhattan\
                        p = 2 : metric = euclidean        \
                        p = arbitrary : metric = minkowski")
optParser.add_option("-s", "--split", dest="split", default='45', type="int",
                  help="choose train test split value for dataset. 0<x<100")

(options, args) = optParser.parse_args()

np.random.seed(0)
iris = datasets.load_iris()

class MyKnn(KNeighborsClassifier):
    
    def __init__(self, n_neighbors=None, metric=None, weights=None, metric_params=None, p=None, algorithm='auto', n_jobs=1, leaf_size=30):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.weights = weights
        self.p = p
	self._init_params(n_neighbors=n_neighbors,
                          algorithm=algorithm,
                          leaf_size=leaf_size,
			  metric=metric,p=p,
                          metric_params=metric_params, n_jobs=n_jobs)
        self.X = iris.data
        self.Y = iris.target
        indices = np.random.permutation(len(self.X))
        self.X_Train = self.X[indices[:-10]]
        self.X_Test = self.X[indices[-10:]]
        self.Y_Train = self.Y[indices[:-10]]
        self.Y_Test = self.Y[indices[-10:]]
    
    def __fit(self, _X, _Y):
        
        return self._fit(_X)
    
    def _predict(self, _X, _Y):
        
        return self.predict(_X)
    
    def show_accuracy(self,_X,_Y):
        
        return accuracy_score(_Y, self._predict(_X, _Y))

if __name__ == '__main__':
    clf = MyKnn(n_neighbors=options.neighbors , metric=options.metric, weights=options.weights, p=options._p)
    clf.fit(clf.X_Train,clf.Y_Train)    
    print("Model : ",clf.fit(clf.X_Train,clf.Y_Train)) 
    print("Prediction : ",clf._predict(clf.X_Test, clf.Y_Test))
    print("MyKNN model accuracy : {}".format(str(float(clf.show_accuracy(clf.X_Test,clf.Y_Test)) *100)+" %."))

