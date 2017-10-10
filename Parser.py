# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 14:22:25 2017

@author: thanos_kats
"""
import glob
import re
import os
from myKNN import MyKnn
import numpy as np
import pandas as pd
import tensorflow as tf
from optparse import OptionParser
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.contrib.keras.python import keras
from tensorflow.contrib.keras.python.keras.preprocessing import text, sequence
from tensorflow.contrib.keras.python.keras import utils
from sklearn.neighbors import KNeighborsClassifier, DistanceMetric

optParser = OptionParser()
optParser.add_option("-n", "--n_neighbors", dest="neighbors", type="int",
                  help="choose k neighbors for KNN model")
optParser.add_option("-w", "--weights", dest="weights", default='uniform',
                  help="choose weights metric for KNN model. Uniform/ Distance")
optParser.add_option("-m", "--metric", dest="metric", default='minkowski',
                  help="choose weights metric for KNN model. minkowski/ euclidean/ manhattan")
optParser.add_option("-p", "--p", dest="_p", default=2, type="int",
                  help="choose p for KNN model, p = 1 metric = manhattan\
			p = 2 : metric = euclidean        \
		  	p = arbitrary : metric = minkowski")
optParser.add_option("-s", "--split", dest="split", default='45', type="int",
                  help="choose train test split value for dataset. 0<x<100")

(options, args) = optParser.parse_args()

path = '/home/thanos_kats/Downloads/misc-master/spam_data1/'

path_flag = (False, True)[os.path.exists(path)]
csv_flag = (False, True)[os.path.exists(os.path.join(os.getcwd(),"spam.csv"))]

def cleanString(raw):
	nohtml = re.sub('<.*?>', '',raw)
	cleanxml = re.sub('<[^<]+>', '',nohtml)
	clean = re.sub('\r\n','\n',cleanxml)
	clean = clean.replace('“','"').replace('”','"')

	return clean

def stop_words(string):
	word_list =[word for word in string.rstrip().split(' ') if word not in set(stopwords.words('english'))]

	return ' '.join(word_list)	

def create_df():
	if path_flag:
		df = pd.DataFrame(columns=['Text','Labels', 'Names'])
		email_list=[]
		for _file in glob.glob(os.path.join(path, '*.txt')):
			with open(_file ,'r') as input_file:
				if 'labels' in _file:
					lines = input_file.read().split('\n')
					labels =[el.split(' ')[0] for el in lines]
					names = [el.split(' ')[1] for el in lines]
					df['Labels'] = labels
					df['Names'] = names
					df.set_index(df.Names, inplace=True, drop=True)
				else:
					reader = input_file.read()
					clean = cleanString(reader)
					email_list.append(clean)
		df['Text'] = email_list
	elif csv_flag:
		df = pd.read_csv(os.path.join(os.getcwd(), 'spam.csv'))
		df['Text'] = df['Text'].apply(lambda x : cleanString(x))
		df['Text'] = df['Text'].apply(lambda x : stop_words(x))
	else:
		raise ValueError("Must have either csv_file or data set.")
	
	return df

class Parser(object):
	def __init__(self):
		self.data = create_df()
    		self.X = None
		self.Y = None

	def set_X_Y(self, X, Y):
		self.X = np.array(self.data['Text'])
		self.Y = np.array(self.data['Labels'].astype(int))

		return self.X, self.Y

    	def train_test(self):
		X, Y = self.set_X_Y(self.X,self.Y)
		x_train,x_test,Y_train,Y_test = train_test_split(X, Y, test_size=options.split*0.01, random_state=0)
		max_words = 1000
		tokenize = text.Tokenizer(num_words=max_words, char_level=False)
		tokenize.fit_on_texts(x_train)
		X_train = tokenize.texts_to_matrix(x_train)
		X_test = tokenize.texts_to_matrix(x_test)
		
		return X_train, X_test, Y_train, Y_test

if __name__ == '__main__':
    parser = Parser()
    X_Train, X_Test, Y_Train, Y_Test = parser.train_test()
    clf = MyKnn(n_neighbors=options.neighbors , metric=options.metric, weights=options.weights, p=options._p)
    clf.fit(X_Train,Y_Train)    
    print("KNN Model for text classification: ",clf.fit(X_Train,Y_Train)) 
    print("KNN Prediction for text classification: ",clf._predict(X_Test, Y_Test))
    print("MyKNN model accuracy for text classification: {}".format(str(clf.show_accuracy(X_Test,Y_Test) *100)+" %."))
    print("\n")
    print("Checking nothing got irreversibly messed up... \n")
    knn = KNeighborsClassifier(n_neighbors=options.neighbors, metric=options.metric, weights=options.weights, p=options._p)
    knn.fit(X_Train,Y_Train)
    print("Built-In knn model accuracy for text classification : \n")
    print(str(accuracy_score(Y_Test, knn.predict(X_Test))*100) +" %.")

