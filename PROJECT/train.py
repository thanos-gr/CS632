#!/home/thanos_kats/anaconda2/bin/python
import os
import pandas as pd
import numpy as np
import keras
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Merge, Bidirectional,Input, Embedding, LSTM
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.layers.convolutional import Conv1D, MaxPooling1D
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from nltk import tokenize
import gensim

os.environ['KERAS_BACKEND'] = 'tensorflow'

MAX_SEQ_LEN = 1000
MAX_NUM_WORDS = 20000
EMBED_DIM = 200
VAL_SPLIT = 10000
TRAIN_TEST = 0.1754878562403482
raw_data = pd.read_csv(os.path.join(os.getcwd(),'Health.csv'), sep=',')

def train_test():
	texts = []
	labels = []
	for i in range(raw_data.Text.shape[0]):
		texts.append(raw_data.Text[i].strip().lower())
		labels.append(raw_data.Labels[i])

	tokenizer = Tokenizer(num_words = MAX_NUM_WORDS)
	tokenizer.fit_on_texts(texts)
	sequences = tokenizer.texts_to_sequences(texts)

	word_index = tokenizer.word_index
	print('%s unique tokens.' % len(word_index))
	data = pad_sequences(sequences, maxlen=MAX_SEQ_LEN)
	le = LabelEncoder()
	le.fit(labels)
	Y = le.transform(labels)

	labels = to_categorical(np.asarray(Y),23)

	print('Data Tensor Shape: ',data.shape)
	print('Label Tensor Shape: ',labels.shape)

	cv = StratifiedShuffleSplit(n_splits=5, test_size=TRAIN_TEST, random_state=0)

        for train_index, test_index in cv.split(data, labels):
                train_data, test_data = data[train_index], data[test_index]
                train_labels, test_labels = labels[train_index], labels[test_index]

        print('Train Index', train_index, 'Test Index', test_index)

        X_train = train_data[:-VAL_SPLIT]
        Y_train = train_labels[:-VAL_SPLIT]
        X_val = train_data[-VAL_SPLIT:]
        Y_val = train_labels[-VAL_SPLIT:]

        X_test = test_data
        Y_test = test_labels

        print('TRAIN SHAPE : ',X_train.shape, 'TEST SHAPE :', X_test.shape, 'VAL SHAPE :',  X_val.shape)
        print('Y TRAIN SHAPE : ',Y_train.shape, 'TEST SHAPE :', Y_test.shape, 'VAL SHAPE :',  Y_val.shape)
        print('Number of classes: ', labels.shape[1])

	return X_train, Y_train, X_val, Y_val, X_test, Y_test, word_index

def build_model(word_index, w2v):
	embedding_matrix = np.random.random((len(word_index) +1, EMBED_DIM))
	for word, i in word_index.items():
		if word in w2v.wv.vocab:
			embedding_vector = w2v[word]
			if embedding_vector is not None:
				embedding_matrix[i] = embedding_vector

	embedding_layer = Embedding(len(word_index) + 1, EMBED_DIM,
				    weights = [embedding_matrix],
				    input_length = MAX_SEQ_LEN,
				    trainable = True)
	convs = []
	filter_sizes = [3,4,5]

	seq_input = Input(shape=(MAX_SEQ_LEN, ),dtype='float32')
	emb_seq = embedding_layer(seq_input)

	for flt in filter_sizes:
		conv = Conv1D(filters=128, kernel_size=flt)(emb_seq)
		relu = Activation('relu')(conv)
		l_pool = MaxPooling1D(5)(relu)
		convs.append(l_pool)

	merge = Merge(mode='concat', concat_axis=1)(convs)
	Conv1= Conv1D(128, 5)(merge)
	Relu1 = Activation('relu')(Conv1)
	Pool1 = MaxPooling1D(5)(Relu1)
	Conv2 = Conv1D(128, 5)(Pool1)
	Relu2 = Activation('relu')(Conv2)
	GPool = MaxPooling1D(30)(Relu2)
	Lstm1 = Bidirectional(LSTM(256, return_sequences=True))(GPool)
	Lstm2 = Bidirectional(LSTM(512, return_sequences=True))(Lstm1)
	Flat = Flatten()(Lstm2)
	Dense1 = Dense(512)(Flat)
	Batch4 = BatchNormalization()(Dense1)
	Relu4 = Activation('relu')(Batch4)
	Drop = Dropout(0.2)(Relu4)
	pred = Dense(23, activation='softmax')(Drop)

	model = Model(seq_input, pred)

	model.summary()
	model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

	return model

def main():
	w2v = gensim.models.Word2Vec.load('w2v_health.bin')
	X_train, Y_train, X_val, Y_val, X_test, Y_test, word_index = train_test()
	model = build_model(word_index, w2v)
	filepath="CNN-LSTM-weights-epoch:{epoch:02d}-val acc:{val_acc:.2f}.hdf5"
	checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	callbacks_list = [checkpoint]

	model.fit(X_train, Y_train, batch_size=32, epochs=200,\
		 validation_data=(X_val, Y_val), callbacks=[checkpoint], verbose=1)

	print("Training Complete.")
	model.save(model_name)
	test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=1)
	print('Test accuracy: ',test_acc,' %')

if __name__=='__main__':
	main()
