# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 15:40:28 2017

@author: thanos_kats
"""
import sklearn
from sklearn import datasets
import numpy as np
from optparse import OptionParser
import random
import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
optParser = OptionParser()
optParser.add_option("-n", "--n_neighbors", dest="neighbors", type="int",
                  help="choose k neighbors for KNN model")
optParser.add_option("-s", "--split", dest="split", default='45', type="int",
                  help="choose train test split value for dataset. 0<x<100")
(options, args) = optParser.parse_args()

iris = datasets.load_iris()

class MyKnn(object):
    
    def __init__(self, n_neighbors=None,split=None):
        self.n_neighbors = n_neighbors
        self.split = split
	self.X, self.Y = (None, )*2
        self.X_Train,self.X_Test,self.Y_Train,self.Y_Test = [],[],[],[]
    
    def distance(self, x1, x2, length):
    	distance = math.sqrt(sum([pow((x1[i] - x2[i]), 2)\
  		            for i in range(length)]))
    	return distance
   	 
    def neighbors(self, train, inst, k):
   	length = len(inst)-1
   	distances = sorted([(np.append(train[x],self.Y_Train[x]),\
   			  self.distance(inst, train[x], len(inst)))\
                	  for x in range(len(train))], key=lambda x: x[1])
   	neigh = [distances[x][0] for x in range(k)]
	
    	return neigh 
 	
    def _loadData(self):
	self.iris = datasets.load_iris()
        self.X = self.iris.data.astype(list)
        self.Y = self.iris.target.astype(list)
	training_set, test_set = [],[]
	for i in range(len(self.X)-1):
               if random.random() < (self.split*0.01):
                    training_set.append(np.append(self.X[i],self.Y[i]))
	       else:
	            test_set.append(np.append(self.X[i],self.Y[i]))
	
	return training_set, test_set

    def _split(self, _X, _Y):
	self.X = _X
	self.Y = _Y
	training_set, test_set = [],[]
        for i in range(len(self.X)-1):
               if random.random() < (self.split*0.01):
                    training_set.append(np.append(self.X[i],self.Y[i]))
               else:
                    test_set.append(np.append(self.X[i],self.Y[i]))
	self.X_Train = [training_set[i][:-1] for i in range(len(training_set))]
        self.Y_Train = [training_set[i][-1] for i in range(len(training_set))]
        self.X_Test = [test_set[i][:-1] for i in range(len(test_set))]
        self.Y_Test = [test_set[i][-1] for i in range(len(test_set))]

	return self.X_Train, self.Y_Train, self.X_Test, self.Y_Test
 
    def _fit(self, _X, _Y):
	training, test = self._loadData()
	self.X_Train = [training[i][:-1] for i in range(len(training))]
	self.Y_Train = [training[i][-1] for i in range(len(training))]
	self.X_Test = [test[i][:-1] for i in range(len(test))]
	self.Y_Test = [test[i][-1] for i in range(len(test))]
	
        return self.X_Train, self.Y_Train, self.X_Test, self.Y_Test

    def getVotes(self,neigh):
	votes = {neigh[x][-1] : 0 for x in range(len(neigh))}
   	for x in range(len(neigh)):
		votes[neigh[x][-1]] = (1,votes[neigh[x][-1]]+1)\
      					[neigh[x][-1] in votes]
   	votes = sorted(votes.iteritems(), key=lambda x: x[1], reverse=True)
   
   	return votes[0][0]	
    
    def _predict(self, _X, _Y):
	y_pred=[self.getVotes(self.neighbors(self.X_Train, _X[x], self.n_neighbors)) for x in range(len(_Y))]
	accuracy = self._accuracy(_Y, y_pred)
	
	return accuracy 
    
    def _accuracy(self,y_true,y_pred):
        cnt = 0
   	for x in range(len(y_true)):
        	cnt = (cnt, cnt+1)[y_true[x] == y_pred[x]]
   	return (cnt/float(len(y_true))) * 100.0

if __name__ == '__main__':
    clf = MyKnn(n_neighbors=options.neighbors , split=options.split)
    print("MyKNN model :",repr(clf))
    clf.X_Train, clf.Y_Train, clf.X_Test, clf.Y_Test, = clf._fit(clf.X_Train, clf.Y_Train)
    print("MyKNN model accuracy : {}".format(str(float(clf._predict(clf.X_Test,clf.Y_Test)))+" %."))
    clf_sk = KNeighborsClassifier(n_neighbors=options.neighbors)
    print("Sklearn KNN Model :",clf_sk.fit(clf.X_Train, clf.Y_Train))
    print("SKLearn KNN model accuracy : {}".format(str(float(accuracy_score(clf.Y_Test, clf_sk.predict(clf.X_Test))) *100)+" %."))
