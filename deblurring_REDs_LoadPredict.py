import tensorflow as tf
from tensorflow.keras import datasets
import dill
import numpy as np
import matplotlib.pyplot as plt
import cv2

from utilities import SSIMLoss, PSNR, load_REDs, split_REDs, print_dataset, inspect_report, get_overlap, rebuild_images
from deblurring_REDs_configuration import *

######################################
### LOAD MODEL, DATASET AND REPORT ###
######################################

loaded_model = tf.keras.models.load_model("./REDs/models/" + model_name + "/" + "epochs" + str(EPOCHS) + "_" + loss_name, custom_objects={'SSIMLoss': SSIMLoss, 'PSNR': PSNR})

test_blurred_REDs = load_REDs(test_blurred_videos_directory, test_num_videos, test_frames_per_video, original_height, original_width)
test_sharped_REDs = load_REDs(test_sharped_videos_directory, test_num_videos, test_frames_per_video, original_height, original_width)
test_blurred_dataset = split_REDs(test_blurred_REDs, test_num_videos, test_frames_per_video, num_patches_width, num_patches_height, height, width, num_conv)
test_sharped_dataset = split_REDs(test_sharped_REDs, test_num_videos, test_frames_per_video, num_patches_width, num_patches_height, height, width, num_conv)

'''
### Used when .DS_Store directory is present
test_blurred_REDs = test_blurred_REDs[test_frames_per_video:]
test_sharped_REDs = test_sharped_REDs[test_frames_per_video:]
test_blurred_dataset = test_blurred_dataset[test_frames_per_video*num_patches_width*num_patches_height:]
test_sharped_dataset = test_sharped_dataset[test_frames_per_video*num_patches_width*num_patches_height:]
'''

filename = "./REDs/reports/" + model_name + "/" + "epochs" + str(EPOCHS) + "_" + loss_name + ".obj"
filehandler = open(filename, 'rb')
report = dill.load(filehandler)

inspect_report(report, metrics)

################################
### PREDICT AND SHOW RESULTS ###
################################

predictions = loaded_model.predict(test_blurred_dataset)
#print_dataset(test_blurred_dataset, predictions)

##########################
### EVALUATE THE MODEL ###
##########################

print(loaded_model.evaluate(x=predictions, y=test_sharped_dataset, batch_size=32))
print(loaded_model.metrics_names)

################################
### RESTORE AND SHOW RESULTS ###
################################

restored_images = rebuild_images(predictions, num_patches_height, num_patches_width, original_height, original_width, height, width, num_conv)

##############################################
### COMPUTE METRICS ON THE RESTORED IMAGES ###
##############################################

num_imgs = test_num_videos * test_frames_per_video
psnr = 0
ssim = 0
for i in range(num_imgs):
    psnr_tmp = PSNR(tf.convert_to_tensor(test_sharped_REDs[i]), tf.convert_to_tensor(restored_images[i])).numpy()
    ssim_tmp = SSIMLoss(tf.convert_to_tensor(test_sharped_REDs[i]), tf.convert_to_tensor(restored_images[i])).numpy()
    psnr += psnr_tmp
    ssim += ssim_tmp
psnr /= num_imgs
ssim = 1 - ssim / num_imgs

print('PSNR: ', psnr)
print('SSIM: ', ssim)

for i in range(5):
    print_dataset(test_sharped_REDs[i:], test_blurred_REDs[i:], predicted_images=restored_images[i:], num=1)