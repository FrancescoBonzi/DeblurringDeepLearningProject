import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.layers import Conv2D, Conv2DTranspose

import matplotlib.pyplot as plt
import numpy as np
import random
import os

from scipy.ndimage import gaussian_filter
from skimage.measure import compare_psnr, compare_ssim, compare_mse

tf.keras.backend.set_floatx('float64')

# Loss functions
def SSIMLoss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))

def PSNR(y_true, y_pred):
    mse = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
    log10 = 2.303  # it is equivalent to ln(10)
    return 10 * tf.keras.backend.log(255 * 255 / mse) / log10


'''LOAD MODEL'''

model = tf.keras.models.load_model(
    "/Users/francescobonzi/Desktop/model", custom_objects={'SSIMLoss': SSIMLoss, 'PSNR': PSNR})


'''PREPARING DATASET'''

(train_images, _), (test_images, _) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0 # Normalize pixel values to be between 0 and 1
train_images = train_images[5000:10000, :, :, :]

def build_dataset(images):
    blurred_images = []
    rands = []
    for i in range(images.shape[0]):
        rand = random.uniform(0, 3)
        rands.append(rand)
        img_gauss = gaussian_filter(images[i], rand)
        blurred_images.append(img_gauss)
    blurred_images = np.array(blurred_images)
    return blurred_images, rands

train_blurred_images, train_rands = build_dataset(train_images)
test_blurred_images, test_rands = build_dataset(test_images)


'''FITTING THE NETWORK'''

EPOCHS = 4

# The patience parameter is the amount of epochs to check for improvement
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='loss', min_delta=0.0001, patience=3)

train_blurred_images = train_blurred_images.reshape(
    len(train_blurred_images), 32, 32, 3)
train_images = train_images.reshape(len(train_images), 32, 32, 3)
report = model.fit(x=train_blurred_images,
                   y=train_images,
                   batch_size=512,
                   epochs=EPOCHS,
                   callbacks=[early_stop],
                   validation_split=0.25)


'''SAVING THE MODEL'''

model.save("/Users/francescobonzi/Desktop/model")

import json
with open('/Users/francescobonzi/Desktop/report.json', 'w') as f:
    json.dump(report.history, f)