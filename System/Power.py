# -*- coding:utf-8 -*-  
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adadelta, Adagrad, Adam, RMSprop
from keras.utils import np_utils
from keras.regularizers import l1, l2
from keras.layers.convolutional import Convolution2D, MaxPooling2D


def NNet(TrainX, TrainY, TestX):

	## the network remained to be XJBplay

	model = Sequential()

	numNode = 20

	model.add(Dense(numNode, input_dim=len(TrainX[0]), W_regularizer=l1(0.1)))
	model.add(BatchNormalization())
	model.add(Activation('tanh'))
	model.add(Dropout(0.5))


	model.add(Dense(numNode, W_regularizer=l2(1)))
	model.add(BatchNormalization())
	model.add(Activation('tanh'))
	model.add(Dropout(0.2))


	model.add(Dense(numNode, W_regularizer=l1(0.1)))
	model.add(BatchNormalization())
	model.add(PReLU())
	model.add(Dropout(0.5))

	model.add(Dense(numNode, W_regularizer=l2(1)))
	model.add(BatchNormalization())
	model.add(Activation('tanh'))
	model.add(Dropout(0.2))


	model.add(Dense(output_dim=2))
	model.add(Activation("softmax"))
	model.compile(loss='sparse_categorical_crossentropy',
				optimizer="adam",
				metrics=["accuracy"])

	model.fit(np.array(TrainX), TrainY, batch_size=int(len(TrainX)*.9), nb_epoch = 1000, shuffle=True, verbose=0, validation_split=0.2)
	PredY = model.predict_classes(np.array(TestX), batch_size=int(len(TrainX)*.9),verbose=0,)

	return PredY


if __name__ == '__main__':
	pass


