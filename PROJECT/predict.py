import os
import pandas as pd
import numpy as np
import keras
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Merge, Bidirectional,Input, Embedding, LSTM, BatchNormalization
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.layers.convolutional import Conv1D, MaxPooling1D
from sklearn.preprocessing import LabelEncoder
from nltk import tokenize
import gensim

os.environ['KERAS_BACKEND'] = 'tensorflow'

MAX_SEQ_LEN = 1000
MAX_NUM_WORDS = 20000
TRAIN_TEST = 40000
EMBED_DIM = 200
MODEL_WEIGHTS ='CNN-LSTM-weights.hdf5'
raw_data = pd.read_csv(os.path.join(os.getcwd(),'Health.csv'), sep=',')

def test_split():
	texts = []
	diagnosis = []
	labels = []
	for i in range(raw_data.Text.shape[0]):
		texts.append(raw_data.Text[i])
		diagnosis.append(tokenize.sent_tokenize(raw_data.Text[i]))

		labels.append(raw_data.Labels[i])

	tokenizer = Tokenizer(num_words = MAX_NUM_WORDS)
	tokenizer.fit_on_texts(texts)
	sequences = tokenizer.texts_to_sequences(texts)

	word_index = tokenizer.word_index
	data = pad_sequences(sequences, maxlen=MAX_SEQ_LEN)
	le = LabelEncoder()
	le.fit(labels)
	Y = le.transform(labels)

	labels = to_categorical(np.asarray(Y),23)

	test = data[TRAIN_TEST:]
	X_test = test
	Y_test = labels[TRAIN_TEST:]

	return X_test, Y_test, word_index

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

	seq_input = Input(shape=(MAX_SEQ_LEN, ),dtype='int32')
	emb_seq = embedding_layer(seq_input)

	for flt in filter_sizes:
		conv = Conv1D(filters=128, kernel_size=flt)(emb_seq)
		batch = BatchNormalization()(conv)
		relu = Activation('relu')(batch)
		l_pool = MaxPooling1D(5)(relu)
		convs.append(l_pool)

	merge = Merge(mode='concat', concat_axis=1)(convs)
	Conv1= Conv1D(64, 5, padding='valid')(merge)
	Batch1 = BatchNormalization()(Conv1)
	Relu1 = Activation('relu')(Batch1)
	Pool1 = MaxPooling1D(5)(Relu1)
	Conv2 = Conv1D(128, 5, padding='same')(Pool1)
	Batch2 = BatchNormalization()(Conv2)
	Relu2 = Activation('relu')(Batch2)
	GPool = MaxPooling1D(35)(Relu2)
	Lstm1 = Bidirectional(LSTM(256, return_sequences=True))(GPool)
	Batch3 = BatchNormalization()(Lstm1)
	Lstm2 = Bidirectional(LSTM(512, return_sequences=True))(Batch3)
	Batch4 = BatchNormalization()(Lstm2)
	Flat = Flatten()(Batch4)
	Dense1 = Dense(512)(Flat)
	Batch4 = BatchNormalization()(Dense1)
	Relu4 = Activation('relu')(Batch4)
	Drop = Dropout(0.5)(Relu4)
	pred = Dense(23, activation='softmax')(Drop)

	model = Model(seq_input, pred)

	model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
	
	model.summary()

	return model

def main():
	w2v = gensim.models.Word2Vec.load('w2v_health.bin')
	X_Test, Y_Test, WordIndex = test_split()
	print('Building Model.')
	model = build_model(WordIndex, w2v)
	print('Loading Weights.')
	model.load_weights(MODEL_WEIGHTS)	
	print('Evaluating test sets.')
	test_loss, test_acc = model.evaluate(X_Test, Y_Test, verbose=1)
	print("Pretrained Model %s: %.2f%%" % (model.metrics_names[1], test_acc*100))

if __name__=='__main__':
        main()	
