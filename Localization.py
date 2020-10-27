import tensorflow as tf
import numpy as np
import copy
from colorama import init
from termcolor import *
import cv2
init()
from glob import glob
import matplotlib.pyplot as plt
import imageio 
from utils import *
from matplotlib.patches import Rectangle



if __name__ == "__main__":
	# Create Model
	model = create_model(n_classes=10)

	# Train model
	model.fit(
		image_generator(), 
		steps_per_epoch=50, 
		epochs=5)

	# Test model
	X_test, y_test = test_sample(100)
	y_pred = model.predict(X_test)

	# Plot Results
	draw_box(X_test, y_pred)







