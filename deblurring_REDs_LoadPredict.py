import tensorflow as tf
from tensorflow.keras import datasets
import dill
import numpy as np

from utilities import SSIMLoss, PSNR, load_REDs, split_REDs, print_dataset, inspect_report, get_overlap, rebuild_images
from deblurring_REDs_configuration import *

test_blurred_REDs = load_REDs(test_blurred_videos_directory, test_num_videos, test_frames_per_video, original_height, original_width)
test_sharped_REDs = load_REDs(test_sharped_videos_directory, test_num_videos, test_frames_per_video, original_height, original_width)

#print("TEST: ",test_blurred_REDs.shape)
#print("TEST: ",test_sharped_REDs.shape)

train_blurred_REDs = load_REDs(train_blurred_videos_directory, test_num_videos, test_frames_per_video, original_height, original_width)
train_blurred_dataset = split_REDs(train_blurred_REDs, test_num_videos, test_frames_per_video, num_patches_width, num_patches_height, height, width, num_conv)

test_blurred_dataset = split_REDs(test_blurred_REDs, test_num_videos, test_frames_per_video, num_patches_width, num_patches_height, height, width, num_conv)
#test_sharped_dataset = split_REDs(test_sharped_REDs, test_num_videos, test_frames_per_video, num_patches_width, num_patches_height, height, width, num_conv)
#print_dataset(test_blurred_dataset, test_blurred_dataset)
prediction = model.predict(train_blurred_dataset)
print(prediction[0].shape)
print(prediction[0])
print_dataset(train_blurred_dataset, prediction)
restored_images = rebuild_images(prediction, num_patches_height, num_patches_width, original_height, original_width, height, width, num_conv)

print_dataset(test_sharped_REDs, test_blurred_REDs, predicted_images=restored_images)

