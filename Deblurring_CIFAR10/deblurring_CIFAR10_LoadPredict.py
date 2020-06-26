import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.layers import Conv2D, Conv2DTranspose

import numpy as np
import os

from utilities import SSIMLoss, PSNR, build_dataset, print_dataset, inspect_report

tf.keras.backend.set_floatx('float64')
width = 32
height = 32
name_model = "SkipConnections"


######################################
### LOAD THE MODEL AND THE DATASET ###
######################################

model = tf.keras.models.load_model("./models/"+name_model, custom_objects={'SSIMLoss': SSIMLoss, 'PSNR': PSNR})
(_, _), (test_images, _) = datasets.cifar10.load_data()
test_images =  test_images / 255.0 # Normalize pixel values to be between 0 and 1
test_images = test_images[10:20, :, :, :]


###############################################
###BLURRED IMAGES GENERATION AND PREDICTION ###
###############################################

test_blurred_images, test_rands = build_dataset(test_images)

predicted_images = model.predict(test_blurred_images)
print_dataset(test_images, test_blurred_images, test_rands, predicted_images=predicted_images, num=5)
