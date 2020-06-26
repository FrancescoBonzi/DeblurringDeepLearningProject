import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.layers import Conv2D, Conv2DTranspose

import matplotlib.pyplot as plt
import numpy as np
import random

from scipy.ndimage import gaussian_filter
#from skimage.measure import compare_psnr, compare_ssim, compare_mse


########################################
### UTILITIES FOR DATASET PROCESSING ###
########################################

def print_dataset(images, blurred_images, sigma, predicted_images = [], num = 10):
    num_plots_per_image = 2
    if len(predicted_images) != 0:
        num_plots_per_image = 3
    plt.figure(figsize=(4*num_plots_per_image,4*num))
    for i in range(num):
        plt.subplot(num,num_plots_per_image,i*num_plots_per_image+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i])
        plt.xlabel("Original")
        plt.subplot(num,num_plots_per_image,i*num_plots_per_image+2)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(blurred_images[i])
        plt.xlabel("Blurred with sigma={:.2f}".format(sigma[i]))
        if len(predicted_images) != 0:
            plt.subplot(num,num_plots_per_image,i*num_plots_per_image+3)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(predicted_images[i])
            plt.xlabel("Predicted")
    plt.show()

''' Construction of blurred images with a gaussian filter of random deviation. For CIFAR10 Dataset '''
def build_dataset(images):
    blurred_images = []
    rands = []
    for i in range(images.shape[0]):
        rand = random.uniform(0,3)
        rands.append(rand)
        img_gauss = gaussian_filter(images[i], rand)
        blurred_images.append(img_gauss)
    blurred_images = np.array(blurred_images)
    return blurred_images, rands


########################################
### DEFINITION OF SOME LOSS FUNCTIONS ###
########################################

def SSIMLoss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))

def PSNR(y_true, y_pred):
    mse = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
    log10 = 2.303 # it is equivalent to ln(10)
    return 10 * tf.keras.backend.log(255 * 255 / mse) / log10


####################################
### REPORT OF ACCURACY FUNCTIONS ###
####################################

def inspect_report(report):
    
    """ Retrieve a list of list results on training and test data
        sets for each training epoch """

    mae = report.history['mae']
    val_mae = report.history['val_mae']
    mse = report.history['mse']
    val_mse = report.history['val_mse']
    loss = report.history['loss']
    val_loss = report.history['val_loss']
    psnr = report.history['PSNR']
    val_psnr = report.history['val_PSNR']

    epochs = range(len(mae)) # Get number of epochs
    
    """ Plot training and validation accuracies per epoch """
    
    plt.plot  ( epochs,     mae )
    plt.plot  ( epochs, val_mae )
    plt.title ('Training and validation mae')
    plt.show()

    plt.plot  ( epochs,     mse )
    plt.plot  ( epochs, val_mse )
    plt.title ('Training and validation mse')
    plt.show()

    plt.plot  ( epochs,     loss )
    plt.plot  ( epochs, val_loss )
    plt.title ('Training and validation SSIM')
    plt.show()

    plt.plot  ( epochs,     psnr )
    plt.plot  ( epochs, val_psnr )
    plt.title ('Training and validation PSNR')
    plt.show()