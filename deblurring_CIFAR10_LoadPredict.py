import tensorflow as tf
from tensorflow.keras import datasets
import dill
import numpy as np

from utilities import SSIMLoss, PSNR, build_dataset, print_dataset, inspect_report
from deblurring_CIFAR10_configuration import *

#metrics = ['loss', 'mae', 'PSNR', 'SSIM']

######################################
### LOAD THE MODEL AND THE DATASET ###
######################################

loaded_model = tf.keras.models.load_model("./CIFAR10/models/" + model_name + "/" + "epochs" + str(EPOCHS) + "_" + loss_name, custom_objects={'SSIMLoss': SSIMLoss, 'PSNR': PSNR})
(_, _), (test_images, _) = datasets.cifar10.load_data()
test_images =  test_images / 255.0 # Normalize pixel values to be between 0 and 1
test_images = test_images[test_lower_bound:test_upper_bound, :, :, :]


########################################
### LOAD THE REPORT AND SHOW RESULTS ###
########################################

filename = "./CIFAR10/reports/" + model_name + "/" + "epochs" + str(EPOCHS) + "_" + loss_name + ".obj"
filehandler = open(filename, 'rb') 
report = dill.load(filehandler)

inspect_report(report, metrics)

################################################
### BLURRED IMAGES GENERATION AND PREDICTION ###
################################################

test_blurred_images, test_rands = build_dataset(test_images, always_the_same=True)
predicted_images = loaded_model.predict(test_blurred_images)

mse = 0
mae = 0
for i in range(len(test_blurred_images)):
    mse += np.square(np.subtract(test_images[i], predicted_images[i])).mean()
    mae += np.abs(np.subtract(test_images[i], predicted_images[i])).mean()

print("mse: ", mse)
print("mae: ", mae)

print_dataset(test_images, test_blurred_images, test_rands, predicted_images=predicted_images, num=5)

