from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from utils import save_image, is_int

import glob
import cv2
import os
import numpy as np

first_char_code = 33 

def load_data():
	image_list = []
	class_list = []
	dir_list = os.listdir('cropped_images')
	listlen = len(dir_list)
	dir_list = [x for x in dir_list if is_int(x)]
	assert len(dir_list) == listlen - 1

	print('len(dir_list): {}'.format(len(dir_list)))
 
	for dirname in dir_list:
		ascii_code = dirname
		for filename in glob.glob('cropped_images/{}/*.jpg'.format(ascii_code)):
			img = cv2.imread(filename, 0)
			image_list.append(img)
			class_list.append(ascii_code)

	print('len(image_list): {}'.format(len(image_list)))

	return image_list, class_list


def prepare_dataset(data_x, data_y):
	data_y = [int(x) - first_char_code for x in data_y]
	new_data_x = data_x
	for index, image in enumerate(data_x):
		treshold_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 20)
		final_image = cv2.resize(treshold_image, (28, 28))
		final_image = 255 - final_image
		new_data_x[index] = final_image
		save_image(index, new_data_x[index], 'out')

	new_data_x = np.array(new_data_x)
	train_x, test_x, train_y, test_y = train_test_split(new_data_x, data_y, test_size=0.25)
	
	train_x = train_x.reshape(train_x.shape[0], 28, 28, 1)
	test_x = test_x.reshape(test_x.shape[0], 28, 28, 1)
	
	train_y = to_categorical(train_y)
	test_y = to_categorical(test_y)

	num_of_categories = len(test_y[0])

	return train_x, train_y, test_x, test_y, num_of_categories
 
def prepare_image_data(train, test):
	train_norm = [x.astype('float32')  for x in train]
	test_norm = [x.astype('float32')  for x in test]
	train_norm = [x / 255.0  for x in train_norm]
	test_norm = [x / 255.0  for x in test_norm]

	train_norm = np.array(train_norm)
	test_norm = np.array(test_norm)

	return train_norm, test_norm
 
def initialize_model(num_of_categories):
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(num_of_categories, activation='softmax'))
	opt = SGD(lr=0.01, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model
 
def train_and_save():
	data_x, data_y = load_data()
	train_x, train_y, test_x, test_y, num_of_categories = prepare_dataset(data_x, data_y)
	train_x, test_x = prepare_image_data(train_x, test_x)
	model = initialize_model(num_of_categories)
	model.fit(train_x, train_y, epochs=10, batch_size=32, verbose=0)
	model.save('final_model.h5')
 
train_and_save()
