import tensorflow as tf
from tensorflow.keras import datasets

import numpy as np
import os
import dill
import time

from utilities import build_dataset, print_dataset, extract_from_report
from deblurring_CIFAR10_configuration import *

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

model.build(input_shape=(len(train_images), 32, 32, 3))
model.summary()

# The patience parameter is the amount of epochs to check for improvement
early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0001, patience=3)

model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(), metrics=other_metrics)
start = time.time()
report = model.fit(x=train_blurred_images, 
                   y=train_images, 
                   batch_size=32, 
                   epochs=EPOCHS, 
                   callbacks=[early_stop], 
                   validation_split=0.25)
end = time.time()
print(end - start)

#################################
#### SAVING REPORTS AND MODEL ###
#################################


filename = "./CIFAR10/reports/" + model_name + "/" + "epochs" + str(EPOCHS) + "_" + loss_name + ".obj"
filehandler = open(filename, 'wb')
list = extract_from_report(report, metrics)
dill.dump(list, filehandler)

model.save("./CIFAR10/models/" + model_name + "/" + "epochs" + str(EPOCHS) + "_" + loss_name)