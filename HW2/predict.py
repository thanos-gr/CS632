import os
os.environ['KERAS_BACKEND']='tensorflow'
import keras
from keras import layers
from keras import models
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd

VAL_PATH = "validation.npy"

MODEL_WEIGHTS = 'CNN-epoch:328-val_acc:0.87.hdf5' 
test_size = 1984
num_classes =2
predict_file = 'predictions.csv'

def load(npy_file):
    data = np.load(npy_file).item()
    return data['images'], data['labels']

def build_model():
	model = models.Sequential()
	model.add(layers.Conv2D(32, (3, 3), input_shape=(32, 32, 3)))
	model.add(layers.BatchNormalization())
	model.add(layers.Activation('relu'))
	model.add(layers.Conv2D(64, (3, 3)))
	model.add(layers.BatchNormalization())
	model.add(layers.Activation('relu'))
	model.add(layers.MaxPooling2D((2, 2)))

	model.add(layers.Conv2D(128, (3, 3)))
	model.add(layers.BatchNormalization())
	model.add(layers.Activation('relu'))
	model.add(layers.Conv2D(64, (3, 3)))
	model.add(layers.BatchNormalization())
	model.add(layers.Activation('relu'))
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.Flatten())

	model.add(layers.Dense(512))
	model.add(layers.BatchNormalization())
	model.add(layers.Activation('relu'))
	model.add(layers.Dropout(0.5))
	model.add(layers.Dense(num_classes, activation='softmax'))
	
	model.summary()
	
	model.compile(optimizer='adam',
		      loss='categorical_crossentropy',
		      metrics=['accuracy'])

	return model

def main():
	test_images, test_labels = load(VAL_PATH)
        X_Test = test_images[:test_size].astype('float32') / 255
        Y_Test = to_categorical(test_labels[:test_size],)
	print('Model : ')
	model = build_model()
	print("loading weights...")
	model.load_weights(MODEL_WEIGHTS)
	scores = model.evaluate(X_Test, Y_Test, verbose=1)
	print("Created model and loaded weights from file.")
	print("Pretrained Model %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
	print("Writting predictions to file %s"%(predict_file))
	preds={}
	with open(predict_file,'w') as outfile:
		outfile.write('Test_Index,Prediction\n')
		for i, inst in enumerate(X_Test):
			inst = inst.reshape(1,32,32,3)
			prediction = [el for el in model.predict(inst)]
			outfile.write(str(i)+','+str(np.argmax(prediction))+'\n')
	print('Predictions saved in %s'%(predict_file))		

if __name__=='__main__':
	main()
