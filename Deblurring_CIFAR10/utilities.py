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

    epochs = range(1, len(mae)+1) # Get number of epochs
    xlabel = 'epochs'
    
    """ Plot training and validation accuracies per epoch """
    plt.plot  ( epochs,     mae, label = 'mae' )
    plt.plot  ( epochs, val_mae, label = 'val_mae' )
    plt.title ('Training and validation mae')
    plt.xlabel(xlabel)
    plt.legend()
    plt.show()

    plt.plot  ( epochs,     mse, label = 'mse' )
    plt.plot  ( epochs, val_mse, label = 'val_mse' )
    plt.title ('Training and validation mse')
    plt.xlabel(xlabel)
    plt.legend()
    plt.show()

    plt.plot  ( epochs,     loss, label = 'loss' )
    plt.plot  ( epochs, val_loss, label = 'val_loss' )
    plt.title ('Training and validation SSIM')
    plt.xlabel(xlabel)
    plt.legend()
    plt.show()

    plt.plot  ( epochs,     psnr, label = 'PSNR' )
    plt.plot  ( epochs, val_psnr, label = 'val_PSNR' )
    plt.title ('Training and validation PSNR')
    plt.xlabel(xlabel)
    plt.legend()
    plt.show()


##############################
### RESIDUAL NETWORK LAYER ###
##############################

''' From KERAS documentation '''
def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x