import os
os.environ['KERAS_BACKEND']='tensorflow'
import keras
from keras import layers
from keras import models
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

TRAIN_PATH = "train.npy"
VAL_PATH = "validation.npy"

CAT_OUTPUT_LABEL = 1
DOG_OUTPUT_LABEL = 0
num_classes = 2
num_epochs = 100
train_val_split = 7000
test_size = 1984
model_name = 'CNN.hdf5'

def load(npy_file):
    data = np.load(npy_file).item()
    return data['images'], data['labels']

def train_val_gen(train_images, train_labels, val_images, val_labels):

	train_gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,\
                         height_shift_range=0.08, zoom_range=0.08, horizontal_flip=True)
	train_gen.fit(train_images)
	val_gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,\
                         height_shift_range=0.08, zoom_range=0.08, horizontal_flip=True)
	val_gen.fit(val_images)
	train_generator = train_gen.flow(train_images, train_labels, batch_size=64)
	val_generator = val_gen.flow(val_images, val_labels, batch_size=64)

	return train_generator, val_generator

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
	train_images, train_labels = load(TRAIN_PATH)
	test_images, test_labels = load(VAL_PATH)

	val_images = train_images[train_val_split:].astype('float32') /255
	train_images = train_images[:train_val_split].astype('float32') / 255
	X_test = test_images[:test_size].astype('float32') / 255

	val_labels = to_categorical(train_labels[train_val_split:], num_classes)
	train_labels = to_categorical(train_labels[:train_val_split], num_classes)
	Y_Test = to_categorical(test_labels[:test_size], num_classes)

	train_generator, val_generator= train_val_gen(train_images, train_labels, val_images, val_labels)

	model = build_model()
	
	filepath="CNN-weights-epoch:{epoch:02d}-val acc:{val_acc:.2f}.hdf5"
	checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	callbacks_list = [checkpoint]

	model.fit_generator(train_generator, steps_per_epoch=train_images.shape[0]//64, epochs=num_epochs,\
		    validation_data=val_generator, validation_steps=val_images.shape[0]//64,\
	    	callbacks=[checkpoint], workers=10)
	print("Training Complete.")
	model.save(model_name)
	test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=1)
	print('Test accuracy: ',test_acc,' %')
	print('Saved weights as %s ' % filepath.split(':')[0])
	print('Saved trained model as %s ' % model_name)


if __name__=='__main__':
	main()
