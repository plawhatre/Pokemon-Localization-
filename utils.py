import tensorflow as tf
import numpy as np
import copy, os
from colorama import init
from termcolor import *
import cv2
init()
from glob import glob
import matplotlib.pyplot as plt
import imageio 

def get_foreground_labels():
	label = []
	for image in glob('Original_Images\\Foreground\\*.png'):
		label.append((image[len('Original_Images\\Foreground\\'):]).split('.')[0])

	return label

def get_background_labels():
	label = []
	for image in glob('Original_Images\\Background\\*.jpg'):
		label.append((image[len('Original_Images\\Background\\'):]).split('.')[0])

	return label

def resize_foreground(size=(2**6, 2**6)):
	if not os.path.exists('Resized_Images\\Foreground'):
		print('Creating Foreground Directories.....')
		os.makedirs('Resized_Images\\Foreground\\')
	for image in glob('Original_Images\\Foreground\\*.png'):
		I = cv2.imread(image, cv2.IMREAD_UNCHANGED)
		I = cv2.resize(I, size, interpolation = cv2.INTER_AREA)
		cv2.imwrite('Resized_Images\\'+ image[len('Original_Images\\'):], I)

def resize_background(size=(2**8, 2**7)):
	if not os.path.exists('Resized_Images\\Background'):
		print('Creating Background Directories.....')
		os.makedirs('Resized_Images\\Background\\')
	for image in glob('Original_Images\\Background\\*.jpg'):
		I = cv2.imread(image)
		I = cv2.resize(I, size)
		cv2.imwrite('Resized_Images\\'+ image[len('Original_Images\\'):], I)

def loss_func(y_true, y_pred):
	loc_loss = tf.keras.losses.binary_crossentropy(y_true[:, :4], y_pred[:, :4])
	cls_loss = tf.keras.losses.categorical_crossentropy(y_true[:, 4:-1], y_pred[:, 4:-1])
	obj_loss = tf.keras.losses.binary_crossentropy(tf.reshape(y_true[:,-1],[-1,1]), tf.reshape(y_pred[:,-1],[-1,1]))
	loss = loc_loss * y_true[:,-1] + cls_loss * y_true[:,-1] + obj_loss / 2
	return loss

def create_model(n_classes):
	vgg = tf.keras.applications.VGG16(
		input_shape=[2**7, 2**8, 3], 
		include_top=False, 
		weights='imagenet')
	x = tf.keras.layers.Flatten()(vgg.output)
	# Localization
	x1 = tf.keras.layers.Dense(4, activation='sigmoid')(x)
	# Classes
	x2 = tf.keras.layers.Dense(n_classes, activation='softmax')(x)
	# Object or not
	x3 = tf.keras.layers.Dense(1, activation='sigmoid')(x)
	x = tf.keras.layers.Concatenate()([x1, x2, x3])
	model = tf.keras.models.Model(vgg.input, x)
	model.compile(loss=loss_func, optimizer=tf.keras.optimizers.Adam(lr=0.001))
	return model

def image_generator(n_classes=5, batch_size=64):
	f_files = glob('Resized_Images\\Foreground\\*.png')
	b_files = glob('Resized_Images\\Background\\*.jpg')

	f_images = [cv2.imread(i, cv2.IMREAD_UNCHANGED) for i in f_files]
	b_images = [cv2.imread(i) for i in b_files]

	f_row, f_col = 2**6, 2**6
	b_row, b_col = 2**7, 2**8
	input_dim = [b_row, b_col]
	
	while  True:
		for _ in range(50):
			X = np.zeros((batch_size, input_dim[0], input_dim[1], 3))
			y = np.zeros((batch_size, 4+n_classes+1))
			for i in range(batch_size):

				# Select Background
				L = np.random.choice(len(b_images))
				bg = b_images[L].copy()

				appear = (np.random.random()> 0.25)
				if appear: 
					# Select Foreground
					L = np.random.choice(len(f_images))
					fg = f_images[L].copy()

					# Resize the pokemon
					scale = 0.5 + np.random.random()
					y[i,2], y[i,3] = scale*f_row, scale*f_col
					fg = cv2.resize(fg, (int(y[i,2]), int(y[i,3])),interpolation = cv2.INTER_AREA)

					# Flip the image side wise
					if np.random.random() < 0.5:
						fg = cv2.flip(fg, 1)

					# Choose random location to place the pokemon
					row_l = np.random.randint(input_dim[0] - y[i,2]) 
					row_h = row_l + int(y[i,2])
					col_l = np.random.randint(input_dim[1] - y[i,3])
					col_h = col_l + int(y[i,3])

					# Create mask using alpha channel
					mask = np.expand_dims((fg[:,:,3]==0), axis=-1)
					bg_slice = bg[row_l:row_h, col_l:col_h,:]
					bg_slice = (mask * bg_slice)
					bg_slice += fg[:, :, :-1]

					# Final Image
					bg[row_l:row_h, col_l:col_h,:] = bg_slice

					# Target vector
					y[i,0] = row_l / input_dim[0]
					y[i,1] = col_l / input_dim[1]
					y[i,2] /= input_dim[0]
					y[i,3] /= input_dim[1]
					y[i,4+L] = 1  
				
				# Image 
				X[i] = bg

				# Target vector
				y[i,-1] = appear

			yield X/255.0, y

def test_sample(batch_size=64):
	n_classes=5
	f_files = glob('Resized_Images\\Foreground\\*.png')
	b_files = glob('Resized_Images\\Background\\*.jpg')

	f_images = [cv2.imread(i, cv2.IMREAD_UNCHANGED) for i in f_files]
	b_images = [cv2.imread(i) for i in b_files]

	f_row, f_col = 2**6, 2**6
	b_row, b_col = 2**7, 2**8
	input_dim = [b_row, b_col]
	
	X = np.zeros((batch_size, input_dim[0], input_dim[1], 3))
	y = np.zeros((batch_size, 4+n_classes+1))
	for i in range(batch_size):

		# Select Background
		L = np.random.choice(len(b_images))
		bg = b_images[L].copy()

		appear = (np.random.random()> 0.25)
		if appear: 
			# Select Foreground
			L = np.random.choice(len(f_images))
			fg = f_images[L].copy()

			# Resize the pokemon
			scale = 0.5 + np.random.random()
			y[i,2], y[i,3] = scale*f_row, scale*f_col
			fg = cv2.resize(fg, (int(y[i,2]), int(y[i,3])),interpolation = cv2.INTER_AREA)

			# Flip the image side wise
			if np.random.random() < 0.5:
				fg = cv2.flip(fg, 1)

			# Choose random location to place the pokemon
			row_l = np.random.randint(input_dim[0] - y[i,2]) 
			row_h = row_l + int(y[i,2])
			col_l = np.random.randint(input_dim[1] - y[i,3])
			col_h = col_l + int(y[i,3])

			# Create mask using alpha channel
			mask = np.expand_dims((fg[:,:,3]==0), axis=-1)
			bg_slice = bg[row_l:row_h, col_l:col_h,:]
			bg_slice = (mask * bg_slice)
			bg_slice += fg[:, :, :-1]

			# Final Image
			bg[row_l:row_h, col_l:col_h,:] = bg_slice

			# Target vector
			y[i,0] = row_l / input_dim[0]
			y[i,1] = col_l / input_dim[1]
			y[i,2] /= input_dim[0]
			y[i,3] /= input_dim[1]
			y[i,4+L] = 1 
		
		# Image 
		X[i] = bg

		# Target vector 
		y[i,-1] = appear

	return X/255.0, y

def draw_box(X,Y):
	for i in range(np.shape(Y)[0]):
		x, y = X[i], Y[i]
		labels = get_foreground_labels()
		if y[-1]==0:
			y_label = 'None'

		else:
			y_label = labels[np.argmax(y[4:-1])]

		# Box coordinates
		row0 = int(y[0]*(2**7))
		col0 = int(y[1]*(2**8))
		row1 = int(row0 + y[2]*(2**7))
		col1 = int(col0 + y[3]*(2**8))
		print(f"pred: {y_label},{y}")


		fig, ax = plt.subplots(1)
		ax.imshow(x)
		rect = Rectangle(
		  (y[1]*(2**8), y[0]*(2**7)),
		  y[3]*(2**8), y[2]*(2**7),linewidth=1,edgecolor='r',facecolor='none')
		ax.add_patch(rect)
		plt.title(y_label)
		plt.show()
		ans = input('Do you want to exit?')
		if ans.lower() == 'y':
			break