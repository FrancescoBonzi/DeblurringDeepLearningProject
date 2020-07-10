import tensorflow as tf
from tensorflow.keras import datasets
import dill
import numpy as np

from utilities import SSIMLoss, PSNR, load_REDs, split_REDs, print_dataset, inspect_report, get_overlap
from deblurring_REDs_configuration import *

test_blurred_REDs = load_REDs(test_blurred_videos_directory, test_num_videos, test_frames_per_video, original_height, original_width)
test_sharped_REDs = load_REDs(test_sharped_videos_directory, test_num_videos, test_frames_per_video, original_height, original_width)

print("TEST: ",test_blurred_REDs.shape)
print("TEST: ",test_sharped_REDs.shape)

test_blurred_dataset = split_REDs(test_blurred_REDs, test_num_videos, test_frames_per_video, num_patches_width, num_patches_height, height, width, num_conv)
#test_sharped_dataset = split_REDs(test_sharped_REDs, test_num_videos, test_frames_per_video, num_patches_width, num_patches_height, height, width, num_conv)
patches = num_patches_height * num_patches_width
prediction = model.predict(test_blurred_dataset)
restored_images = np.zeros((int(len(prediction)/patches), original_height, original_width, 3))

for i in range(int(len(prediction)/patches)):
    for w in range(num_patches_width):
        start_width = get_overlap(w, num_patches_width, num_conv)
        for h in range(num_patches_height):
            start_height = get_overlap(h, num_patches_height, num_conv)
            restored_images[i, h*height:(h+1)*height, w*width:(w+1)*width, :] = prediction[i*patches + w*num_patches_height+h, start_height:start_height+height, start_width:start_width+width, :]
                                   

print_dataset(test_sharped_REDs, test_blurred_REDs, predicted_images=prediction)