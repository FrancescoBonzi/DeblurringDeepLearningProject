import tensorflow as tf
from tensorflow.keras import datasets

import numpy as np
import os
import dill
import time

from utilities import build_dataset, print_dataset, extract_from_report
from deblurring_CIFAR10_configuration import *

loaded_model = tf.keras.models.load_model("./CIFAR10/models/" + model_name + "/" + "epochs" + str(EPOCHS) + "_" + loss_name, custom_objects={'SSIMLoss': SSIMLoss, 'PSNR': PSNR})
loaded_model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(), metrics=other_metrics)

########################################
### LOAD THE REPORT AND SHOW RESULTS ###
########################################

filename = "./CIFAR10/reports/" + model_name + "/" + "epochs" + str(EPOCHS) + "_" + loss_name + ".obj"
filehandler = open(filename, 'rb') 
report = dill.load(filehandler)

#########################################################
### LOADING DATASET AND GENERATION OF BLURRING IMAGES ###
#########################################################

(train_images, _), (_, _) = datasets.cifar10.load_data()

train_images = train_images / 255.0 # Normalize pixel values to be between 0 and 1
if demo: train_images = train_images[1:100,:,:,:]

train_blurred_images, train_rands = build_dataset(train_images)

##################################################
#### BUILDING, COMPILING AND FITTING THE MODEL ###
##################################################

# The patience parameter is the amount of epochs to check for improvement
early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0001, patience=3)

start = time.time()
report = loaded_model.fit(x=train_blurred_images, 
                   y=train_images, 
                   batch_size=32, 
                   epochs=other_EPOCHS, 
                   callbacks=[early_stop], 
                   validation_split=0.25)
end = time.time()
with open("./CIFAR10/times.txt", "a") as myfile:
    myfile.write("models: {:15}, {:>2} epochs, loss: {:8} --> TIME: {:>8.1f}s\n".format(model_name, str(EPOCHS+other_EPOCHS), loss_name, (end - start)))

#################################
#### SAVING REPORTS AND MODEL ###
#################################

filename = "./CIFAR10/reports/" + model_name + "/" + "epochs" + str(EPOCHS) + "_" + loss_name + ".obj"
filehandler = open(filename, 'rb') 
old_report = dill.load(filehandler)

filename = "./CIFAR10/reports/" + model_name + "/" + "epochs" + str(EPOCHS+other_EPOCHS) + "_" + loss_name + ".obj"
filehandler = open(filename, 'wb')
list = extract_from_report(report, metrics)
all_reports = []
for i in range(len(old_report)):
    all_reports.append(old_report[i] + list[i])
print(len(old_report[0]))
print(len(list[0]))
print(len(all_reports[0]))
dill.dump(all_reports, filehandler)

loaded_model.save("./CIFAR10/models/" + model_name + "/" + "epochs" + str(EPOCHS+other_EPOCHS) + "_" + loss_name)