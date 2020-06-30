
import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.layers import Conv2D, Conv2DTranspose

import matplotlib.pyplot as plt
import numpy as np
import random
import os

import cv2
from scipy.ndimage import gaussian_filter
from skimage.measure import compare_psnr, compare_ssim, compare_mse
from utilities import load_REDs, split_REDs, print_dataset

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.keras.backend.set_floatx('float64')

EPOCHS = 2

num_conv = 3
num_patches_width = 4
num_patches_height = 2
patches = num_patches_width * num_patches_height
original_width = 1280
original_height = 720
width = int(original_width/num_patches_width)
height = int(original_height/num_patches_height)

num_videos = len(os.listdir(directory))
print(num_videos)
frame_per_video = len(os.listdir(blurred_videos_directory + "/" + os.listdir(blurred_videos_directory)[1]))
print(frame_per_video)


blurred_videos_directory = "./train_blur"
original_videos_directory = "./train_sharp"
train_blurred_REDs = load_REDs(blurred_videos_directory)
train_sharped_REDs = load_REDs(original_videos_directory)

print(train_blurred_REDs.shape)
print(train_sharped_REDs.shape)

print_dataset(train_blurred_REDs, train_sharped_REDs)

train_blurred_dataset = split_REDs(train_blurred_REDs)
train_sharped_dataset = split_REDs(train_sharped_REDs)

print_dataset(train_sharped_dataset, train_blurred_dataset)

"""# DEFINITION OF THE CONVOLUTIONAL NEURAL NETWORK"""

model = DeblurringImagesModel()
model.build(input_shape=(num_videos*frame_per_video *
                         patches, height+2*num_conv, width+2*num_conv, 3))
model.summary()


# The patience parameter is the amount of epochs to check for improvement
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='loss', min_delta=0.0001, patience=3)
model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(),
              metrics=[SSIMLoss, 'mae', PSNR])
report = model.fit(x=train_blurred_dataset,
                   y=train_sharped_dataset,
                   batch_size=16,
                   epochs=EPOCHS,
                   callbacks=[early_stop],
                   validation_split=0.25)

#inspect_report(report)

'''PREDICTION'''

test_blurred_videos_directory = "./test_blur"
test_original_videos_directory = "./test_sharp"
test_blurred_REDs = load_REDs(test_blurred_videos_directory)
test_sharped_REDs = load_REDs(test_original_videos_directory)

print("TEST: ",test_blurred_REDs.shape)
print("TEST: ",test_sharped_REDs.shape)

test_blurred_dataset = split_REDs(test_blurred_REDs)
#test_sharped_dataset = split_REDs(test_sharped_REDs)

restored_images = np.zeros(int(len(prediction)/patches),
                           original_height, original_width, 3)
prediction = model.predict(test_blurred_dataset)
for i in range(int(len(prediction)/patches)):
    for w in range(num_patches_width):
        start_width = get_left_overlap(w, num_patches_width)
        for h in range(num_patches_height):
            start_height = get_left_overlap(h, num_patches_height)
            restored_images[i, w*width:(w+1)*width, h*height:(h+1)*height, :] = prediction[i*patches + w*num_patches_height+h, start_height:start_height+height, start_width:start_width+width, :]
                                   

#print_dataset(test_sharped_REDs, test_blurred_REDs, predicted_images=prediction)
