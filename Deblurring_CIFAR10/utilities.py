import tensorflow as tf
from autoencoder_models import *

import matplotlib.pyplot as plt
import numpy as np
import random

from scipy.ndimage import gaussian_filter

########################################
### DEFINITION OF SOME LOSS FUNCTIONS ###
########################################

def SSIM(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))


def PSNR(y_true, y_pred):
    mse = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
    log10 = 2.303  # it is equivalent to ln(10)
    return 10 * tf.keras.backend.log(255 * 255 / mse) / log10


###################################
### UTILITIES FOR CONFIGURATION ###
###################################

encode_loss = {
        'SSIMLoss': SSIM,
        'mae': 'mae',
        'mse': 'mse',
        'PSNR': PSNR
    }

def get_metrics(loss):
    base_metrics = ['SSIMLoss', 'mae', 'mse', 'PSNR']
    metrics = ['loss']
    for i in range(len(base_metrics)):
        if base_metrics[i] != loss:
            metrics.append(base_metrics[i])
    print('Metrics: ', metrics)
    return metrics

def get_other_metrics(metrics):
    other_metrics = []
    for m in metrics:
        if m != 'loss':
            other_metrics.append(encode_loss[m])
    print('Other Metrics: ', other_metrics)
    return other_metrics

def get_model(model_name):
    encode = {
        'CNNBase_v1': DeblurringCNNBase_v1(),
        'CNNBase_v2': DeblurringCNNBase_v2(),
        'ResNet_v1': DeblurringResnet_v1(),
        'ResNet_v2': DeblurringResnet_v2(),
        'SkipConnections': DeblurringSkipConnections()
    }
    model = encode[model_name]
    print(model)
    return model
    
def get_loss(loss_name):
    return encode_loss[loss_name]

########################################
### UTILITIES FOR DATASET PROCESSING ###
########################################

def print_dataset(images, blurred_images, sigma, predicted_images=[], num=10):
    num_plots_per_image = 2
    if len(predicted_images) != 0:
        num_plots_per_image = 3
    plt.figure(figsize=(4*num_plots_per_image, 4*num))
    for i in range(num):
        plt.subplot(num, num_plots_per_image, i*num_plots_per_image+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i])
        plt.xlabel("Original")
        plt.subplot(num, num_plots_per_image, i*num_plots_per_image+2)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(blurred_images[i])
        plt.xlabel("Blurred with sigma={:.2f}".format(sigma[i]))
        if len(predicted_images) != 0:
            plt.subplot(num, num_plots_per_image, i*num_plots_per_image+3)
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
        rand = random.uniform(0, 3)
        rands.append(rand)
        img_gauss = gaussian_filter(images[i], rand)
        blurred_images.append(img_gauss)
    blurred_images = np.array(blurred_images)
    return blurred_images, rands


####################################
### REPORT OF ACCURACY FUNCTIONS ###
####################################

def extract_from_report(report, metrics):
    list = []
    for i in range(len(metrics)):
        list.append(report.history[metrics[i]])
        list.append(report.history['val_' + metrics[i]])
    return list


def inspect_report(report, metrics):
    """ Retrieve a list of list results on training and test data
        sets for each training epoch """

    epochs = range(1, len(report[0])+1)  # Get number of epochs
    xlabel = 'epochs'

    for i in range(len(metrics)):
        label = metrics[i]
        plt.plot(epochs, report[i], label=label)
        plt.plot(epochs, report[i+1], label='val_' + label)
        plt.title('Training and validation ' + label)
        plt.xlabel(xlabel)
        plt.legend()
        plt.show()



""" 
#################################
############## REDS #############
#################################
"""

def count_frame_per_video(directory):
    videos = sorted(os.listdir(directory))
    path = directory + "/" + enumerate(videos)[1]
    return len(os.listdir(directory))


def get_left_overlap(k, num_patches):
    if k == 0:
        left_overlap_factor = 0
    elif k == num_patches-1:
        left_overlap_factor = 2*num_conv
    else:
        left_overlap_factor = num_conv
    return left_overlap_factor


def load_REDs(directory):
    loaded_dataset = np.zeros(
        (num_videos*frame_per_video, original_height, original_width, 3))
    videos = sorted(os.listdir(directory))
    for i, dir in enumerate(videos):
        path = directory + "/" + dir
        if os.path.isdir(path):
            frames = sorted(os.listdir(path))
            print("loading ", path, "...")
            for j, frame in enumerate(frames):
                path_frame = path + "/" + frame
                if os.path.isfile(path_frame) and path_frame.endswith(".png"):
                    img = cv2.imread(path_frame)
                    loaded_dataset[(i-1)*frame_per_video+j-1, :, :, :] = img/255
    return loaded_dataset

def split_REDs(loaded_dataset):
    splitted_dataset = np.zeros(
        (num_videos*frame_per_video*patches, height+2*num_conv, width+2*num_conv, 3))
    for i in range(int(num_videos*frame_per_video/patches)):
        for w in range(num_patches_width):
            left_overlap_factor_width = get_left_overlap(
                w, num_patches_width)
            start_width = w*width-left_overlap_factor_width
            for h in range(num_patches_height):
                left_overlap_factor_height = get_left_overlap(
                    h, num_patches_height)
                start_heigth = h*height-left_overlap_factor_height
                splitted_dataset[i*patches+w*num_patches_height+h, :, :, :] = loaded_dataset[i, start_heigth:(
                    start_heigth+height+2*num_conv), start_width:(start_width+width+2*num_conv), ]
    return splitted_dataset