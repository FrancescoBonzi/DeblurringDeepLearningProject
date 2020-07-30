import tensorflow as tf
from tensorflow.keras import datasets

import numpy as np
import os
import dill
import time

from utilities import load_REDs, split_REDs, print_dataset, extract_from_report
from deblurring_REDs_configuration import *

loaded_model = tf.keras.models.load_model("./REDs/models/" + model_name + "/" + "epochs" + str(EPOCHS) + "_" + loss_name, custom_objects={'SSIMLoss': SSIMLoss, 'PSNR': PSNR})
loaded_model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(), metrics=other_metrics)

#########################################################
### LOADING DATASET AND GENERATION OF BLURRING IMAGES ###
#########################################################

train_blurred_REDs = load_REDs(train_blurred_videos_directory, num_videos, frames_per_video, original_height, original_width, video_shift=video_shift, frame_shift=frame_shift)
train_sharped_REDs = load_REDs(train_sharped_videos_directory, num_videos, frames_per_video, original_height, original_width, video_shift=video_shift, frame_shift=frame_shift)

print("video_shift = ", video_shift)
print("frame_shift = ", frame_shift)
print(train_blurred_REDs.shape)
print(train_sharped_REDs.shape)

train_blurred_dataset = split_REDs(train_blurred_REDs, num_videos, frames_per_video, num_patches_width, num_patches_height, height, width, num_conv)
train_sharped_dataset = split_REDs(train_sharped_REDs, num_videos, frames_per_video, num_patches_width, num_patches_height, height, width, num_conv)

##################################################
#### BUILDING, COMPILING AND FITTING THE MODEL ###
##################################################

# The patience parameter is the amount of epochs to check for improvement
early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0001, patience=3)

start = time.time()
report = loaded_model.fit(x=train_blurred_dataset, 
                   y=train_sharped_dataset, 
                   batch_size=32, 
                   epochs=other_EPOCHS, 
                   callbacks=[early_stop], 
                   validation_split=0.25)
end = time.time()
with open("./REDs/times.txt", "a") as myfile:
    myfile.write("models: {:15}, {:>2} epochs, loss: {:8} --> TIME: {:>8.1f}s\n".format(model_name, str(EPOCHS+other_EPOCHS), loss_name, (end - start)))

#################################
#### SAVING REPORTS AND MODEL ###
#################################

filename = "./REDs/reports/" + model_name + "/" + "epochs" + str(EPOCHS) + "_" + loss_name + ".obj"
filehandler = open(filename, 'rb') 
old_report = dill.load(filehandler)

filename = "./REDs/reports/" + model_name + "/" + "epochs" + str(EPOCHS+other_EPOCHS) + "_" + loss_name + ".obj"
filehandler = open(filename, 'wb')
list = extract_from_report(report, metrics)
all_reports = []
for i in range(len(old_report)):
    all_reports.append(old_report[i] + list[i])
print(len(old_report[0]))
print(len(list[0]))
print(len(all_reports[0]))
dill.dump(all_reports, filehandler)

loaded_model.save("./REDs/models/" + model_name + "/" + "epochs" + str(EPOCHS+other_EPOCHS) + "_" + loss_name)