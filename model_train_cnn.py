# -*- coding: utf-8 -*-
"""
Created on Mon May 21 10:12:28 2018

@author: dutta
"""

import numpy as np
import pickle
import cv2, os
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras import backend as K
K.set_image_dim_ordering('tf')

def size_img():
	img = cv2.imread('gestures/train1/100.jpg', 0)
	return img.shape
rows, cols = size_img()

def count_classes():
	return len(os.listdir('gestures/'))

def cnn_model():
	class_count = count_classes()
	model = Sequential()
	model.add(Conv2D(32, (5,5), input_shape=(rows, cols, 1), activation='sigmoid'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
	model.add(Conv2D(64, (5,5), activation='sigmoid'))
	model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='valid'))
	model.add(Flatten())
	model.add(Dense(1024, activation='sigmoid'))
	model.add(Dropout(0.6))
	model.add(Dense(class_count, activation='sigmoid'))
	stochastic_desc = optimizers.SGD(lr=1e-2)
	model.compile(loss='categorical_crossentropy', optimizer=stochastic_desc, metrics=['accuracy'])
	filepath="model"
	step = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	cb = [step]
	return model, cb

def model_train():
	with open("train_images", "rb") as f:
		trainI = np.array(pickle.load(f))
	trainI = np.reshape(trainI, (trainI.shape[0], rows, cols, 1))
	with open("train_labels", "rb") as f:
		trainL = np_utils.to_categorical(np.array(pickle.load(f), dtype=np.int32))
	with open("test_images", "rb") as f:
		valI = np.array(pickle.load(f))
	valI = np.reshape(valI, (valI.shape[0], rows, cols, 1))
	with open("test_labels", "rb") as f:
		valL = np_utils.to_categorical(np.array(pickle.load(f), dtype=np.int32))
	model, cb = cnn_model()
	model.fit(trainI, trainL, validation_data=(valI, valL), epochs=25, batch_size=100, callbacks=cb)
    
model_train()
K.clear_session();
	
        
        
    




