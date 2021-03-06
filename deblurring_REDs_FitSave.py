import tensorflow as tf
import numpy as np
import os
import dill
import time

from utilities import load_REDs, print_dataset, extract_from_report, split_REDs
from deblurring_REDs_configuration import *


##################################################
### LOADING DATASET AND SPLIT IMAGES IN PATCHES###
##################################################

train_blurred_REDs = load_REDs(train_blurred_videos_directory, num_videos, frames_per_video, original_height, original_width)
train_sharped_REDs = load_REDs(train_sharped_videos_directory, num_videos, frames_per_video, original_height, original_width)

print(train_blurred_REDs.shape)
print(train_sharped_REDs.shape)

#print_dataset(train_blurred_REDs, train_sharped_REDs)

train_blurred_dataset = split_REDs(train_blurred_REDs, num_videos, frames_per_video, num_patches_width, num_patches_height, height, width, num_conv)
train_sharped_dataset = split_REDs(train_sharped_REDs, num_videos, frames_per_video, num_patches_width, num_patches_height, height, width, num_conv)

#print_dataset(train_sharped_dataset, train_blurred_dataset)


##################################################
#### BUILDING, COMPILING AND FITTING THE MODEL ###
##################################################

model.build(input_shape=(num_videos*frames_per_video *
                         num_patches_width*num_patches_height, height+2*num_conv, width+2*num_conv, 3))
model.summary()


# The patience parameter is the amount of epochs to check for improvement
early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0001, patience=3)

model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(), metrics=other_metrics)
start = time.time()
report = model.fit(x=train_blurred_dataset,
                   y=train_sharped_dataset,
                   batch_size=16,
                   epochs=EPOCHS,
                   callbacks=[early_stop],
                   validation_split=0.25)
end = time.time()
with open("./REDs/times.txt", "a") as myfile:
    myfile.write("models: {:15}, {:>2} epochs, loss: {:8} --> TIME: {:>8.1f}s\n".format(model_name, str(EPOCHS), loss_name, (end - start)))



#################################
#### SAVING REPORTS AND MODEL ###
#################################


filename = "./REDs/reports/" + model_name + "/" + "epochs" + str(EPOCHS) + "_" + loss_name + ".obj"
filehandler = open(filename, 'wb')
list = extract_from_report(report, metrics)
dill.dump(list, filehandler)

model.save("./REDs/models/" + model_name + "/" + "epochs" + str(EPOCHS) + "_" + loss_name)



