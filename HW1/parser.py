# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 14:22:25 2017

@author: thanos_kats
"""
import glob
import re
import os
from myknn import MyKnn
import numpy as np
import pandas as pd
import random
from optparse import OptionParser
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from tensorflow.contrib.keras.python import keras
from tensorflow.contrib.keras.python.keras.preprocessing import text, sequence
from tensorflow.contrib.keras.python.keras import utils

optParser = OptionParser()
optParser.add_option("-n", "--n_neighbors", dest="neighbors", type="int",
                  help="choose k neighbors for KNN model")
optParser.add_option("-s", "--split", dest="split", default='45', type="int",
                  help="choose train test split value for dataset. 0<x<100")

(options, args) = optParser.parse_args()

path = '/home/thanos_kats/Downloads/misc-master/spam_data1/'

path_flag = (False, True)[os.path.exists(path)]
csv_flag = (False, True)[os.path.exists(os.path.join(os.getcwd(),"spam.csv"))]

def cleanString(raw):
	txt = raw.rstrip()
	nohtml = re.sub('<.*?>', '',txt)
	cleanxml = re.sub('<[^<]+>', '',nohtml)
	clean = re.sub('\r\n','\n',cleanxml)
	clean = clean.replace('“','"').replace('”','"')
	return clean

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
					df.set_index(df.Names, inplace=True)
					del df['Names']
				else:
					reader = input_file.read()
					clean = cleanString(reader)
					email_list.append(clean)
		df['Text'] = email_list
		df.to_csv('spam.csv',sep=',',index=False)
	elif csv_flag:
		df = pd.read_csv(os.path.join(os.getcwd(), 'spam.csv'))
		df['Test'] = df['Text'].apply(lambda x : cleanString(x))
	else:
		raise ValueError("No csv_file or valid path provided.")
	
	return df

class Parser(MyKnn):
	def __init__(self, split = None):
		self.data = create_df()
    		self.X = None
		self.Y = None
		self.split = options.split
		self.n_neighbors = options.neighbors
	
	def set_X_Y(self, X, Y):
		self.X = self.data['Text']
		self.Y = self.data['Labels'].astype(int)

		return self.X, self.Y
	
	def _fit(self,_X,_Y):

		return MyKnn._split(self,_X, _Y)

    	def _fit_transform(self):
		X, Y = self.set_X_Y(self.X,self.Y)
		X_Train, Y_Train, X_Test, Y_Test = self._fit(X, Y)
		max_words = 1000
		tokenize = text.Tokenizer(num_words=max_words, char_level=False)
		tokenize.fit_on_texts(pd.Series([y for x in self.X_Train for y in x]))
		self.X_Train = tokenize.texts_to_matrix(pd.Series([y for x in X_Train for y in x]))
		self.X_Test = tokenize.texts_to_matrix(pd.Series([y for x in X_Test for y in x]))
		
		return self.X_Train, self.X_Test, self.Y_Train, self.Y_Test
	
	def predict(self, _X, _Y):
	 	self.X_Train = self._fit_transform()[0]
		
		return MyKnn._predict(self, _X, _Y)

if __name__ == '__main__':
    parser = Parser()
    print("My KNN mdel :",repr(parser))
    parser.X_Train, parser.X_Test, parser.Y_Train, parser.Y_Test = parser._fit_transform()
    print("MyKNN model accuracy for text classification: {}".format(str(float(parser.predict(parser.X_Test,parser.Y_Test)))+" %."))
    clf_sk = KNeighborsClassifier(n_neighbors=options.neighbors)
    print("Sklearn KNN Model :",clf_sk.fit(parser.X_Train, parser.Y_Train))
    print("SKLearn KNN model accuracy : {}".format(str(float(accuracy_score(parser.Y_Test, clf_sk.predict(parser.X_Test))) *100)+" %."))

