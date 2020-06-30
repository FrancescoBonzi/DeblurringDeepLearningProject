import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.layers import Conv2D, Conv2DTranspose

import numpy as np
import os
import dill

from utilities import SSIMLoss, PSNR, build_dataset, print_dataset, inspect_report

tf.keras.backend.set_floatx('float64')

width = 32
height = 32
EPOCHS = 35
model_folder = "ResNet"
metrics = ['loss', 'mae', 'mse', 'PSNR']
test_lower_bound = 10
test_upper_bound = 20

######################################
### LOAD THE MODEL AND THE DATASET ###
######################################

model = tf.keras.models.load_model("./models/" + model_folder + "/" + "epochs" + str(EPOCHS), custom_objects={'SSIMLoss': SSIMLoss, 'PSNR': PSNR})
(_, _), (test_images, _) = datasets.cifar10.load_data()
test_images =  test_images / 255.0 # Normalize pixel values to be between 0 and 1
test_images = test_images[test_lower_bound:test_upper_bound, :, :, :]


########################################
### LOAD THE REPORT AND SHOW RESULTS ###
########################################

filename = "./reports/" + model_folder + "/" + "epochs" + str(EPOCHS) + ".obj"
filehandler = open(filename, 'rb') 
report = dill.load(filehandler)

inspect_report(report, metrics)

###############################################
###BLURRED IMAGES GENERATION AND PREDICTION ###
###############################################

test_blurred_images, test_rands = build_dataset(test_images)

predicted_images = model.predict(test_blurred_images)
print_dataset(test_images, test_blurred_images, test_rands, predicted_images=predicted_images, num=5)
