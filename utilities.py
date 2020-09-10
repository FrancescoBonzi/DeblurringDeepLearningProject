import tensorflow as tf
from autoencoder_models import *
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
import random
import os
import cv2

from scipy.ndimage import gaussian_filter
from sys import exit

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


#########################################
### DEFINITION OF SOME LOSS FUNCTIONS ###
#########################################

def SSIMLoss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))


def PSNR(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, 1.0)


###################################
### UTILITIES FOR CONFIGURATION ###
###################################

encode_loss = {
        'SSIMLoss': SSIMLoss,
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
    #print('Metrics: ', metrics)
    return metrics

def get_other_metrics(metrics):
    other_metrics = []
    for m in metrics:
        if m != 'loss':
            other_metrics.append(encode_loss[m])
    #print('Other Metrics: ', other_metrics)
    return other_metrics

def get_model(model_name):
    encode = {
        'CNNBase_v1': DeblurringCNNBase_v1(),
        'CNNBase_v2': DeblurringCNNBase_v2(),
        'ResNet_v1': DeblurringResnet_v1(),
        'ResNet_v2': DeblurringResnet_v2(),
        'SkipConnections_v1': DeblurringSkipConnections_v1(),
        'SkipConnections_v2': DeblurringSkipConnections_v2()
    }
    model = encode[model_name]
    #print(model)
    return model
    
def get_loss(loss_name):
    return encode_loss[loss_name]

########################################
### UTILITIES FOR REDs CONFIGURATION ###
########################################

def get_num_videos(blurred_videos_directory, sharped_videos_directory):
    if len(os.listdir(blurred_videos_directory)) == len(os.listdir(sharped_videos_directory)):
        return len(os.listdir(blurred_videos_directory))
    else:
        exit('Sharped and blurred images numbers do not match') 

def get_frames_per_video(blurred_videos_directory):
    return len(os.listdir(blurred_videos_directory + "/" + os.listdir(blurred_videos_directory)[1]))


def get_num_conv(model_name):
    encode = {
        'CNNBase_v1': 3,
        'CNNBase_v2': 3,
        'ResNet_v1': 3,
        'ResNet_v2': 3,
        'SkipConnections_v1': 4,
        'SkipConnections_v2': 15
    }
    return encode[model_name]

########################################
### UTILITIES FOR DATASET PROCESSING ###
########################################

def print_dataset(images, blurred_images, sigma=[], predicted_images=[], num=5):
    num_plots_per_image = 2
    print_sigma = True
    if len(sigma) == 0:
        print_sigma = False
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
        if print_sigma: plt.xlabel("Blurred with sigma={:.2f}".format(sigma[i]))
        if len(predicted_images) != 0:
            plt.subplot(num, num_plots_per_image, i*num_plots_per_image+3)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(predicted_images[i])
            plt.xlabel("Predicted")
    plt.show()

##################################
### PROCESSING CIFAR10 DATASET ###
##################################

def build_dataset(images, always_the_same=False):
    blurred_images = []
    rands = []
    if always_the_same: random.seed(9001)
    for i in range(images.shape[0]):
        rand = random.uniform(0, 3)
        rands.append(rand)
        img_gauss = gaussian_filter(images[i], rand)
        blurred_images.append(img_gauss)
    blurred_images = np.array(blurred_images)
    return blurred_images, rands

###############################
### PROCESSING REDS DATASET ###
###############################

def get_overlap(k, num_patches, num_conv):
    if k == 0:
        left_overlap_factor = 0
    elif k == num_patches-1:
        left_overlap_factor = 2*num_conv
    else:
        left_overlap_factor = num_conv
    return left_overlap_factor


def load_REDs(directory, num_videos, frames_per_video, original_height, original_width, video_shift=0, frame_shift=0):
    loaded_dataset = np.zeros(
        (num_videos*frames_per_video, original_height, original_width, 3))
    videos = sorted(os.listdir(directory))
    for i in range(video_shift, num_videos + video_shift):
        path = directory + "/" + videos[i]
        if os.path.isdir(path):
            frames = sorted(os.listdir(path))
            print("loading ", path, "...")
            for j in range(frame_shift, frames_per_video + frame_shift):
                path_frame = path + "/" + frames[j]
                if os.path.isfile(path_frame) and path_frame.endswith(".png"):
                    img = cv2.imread(path_frame)
                    loaded_dataset[(i-video_shift)*frames_per_video+(j-frame_shift), :, :, :] = img/255
    return loaded_dataset

def split_REDs(loaded_dataset, num_videos, frames_per_video, num_patches_width, num_patches_height, height, width, num_conv):
    patches = num_patches_width*num_patches_height
    splitted_dataset = []
    for i in range(int(num_videos*frames_per_video)):
        for w in range(num_patches_width):
            left_overlap_factor_width = get_overlap(w, num_patches_width, num_conv)
            start_width = w*width-left_overlap_factor_width
            for h in range(num_patches_height):
                upper_overlap_factor_height = get_overlap(h, num_patches_height, num_conv)
                start_heigth = h*height-upper_overlap_factor_height
                splitted_dataset.append(loaded_dataset[i, start_heigth:(start_heigth+height+2*num_conv), start_width:(start_width+width+2*num_conv),:])
            #plt.imshow(splitted_dataset[i*patches+w*num_patches_height+h, :, :, :])
            #plt.show()
    return np.array(splitted_dataset)


def rebuild_images(patches, num_patches_height, num_patches_width, original_height, original_width, height, width, num_conv):
    num_patches = num_patches_height*num_patches_width
    restored_images = np.zeros((int(len(patches)/num_patches), original_height, original_width, 3))
    for i in range(int(len(patches)/num_patches)):
        for w in range(num_patches_width):
            start_width = get_overlap(w, num_patches_width, num_conv)
            for h in range(num_patches_height):
                start_height = get_overlap(h, num_patches_height, num_conv)
                restored_images[i, h*height:(h+1)*height, w*width:(w+1)*width, :] = patches[i*num_patches + w*num_patches_height+h, start_height:start_height+height, start_width:start_width+width, :]
    return restored_images


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
        plt.plot(epochs, report[2*i], label=label)
        plt.plot(epochs, report[2*i+1], label='val_' + label)
        plt.title('Training and validation ' + label)
        plt.xlabel(xlabel)
        plt.legend()
        plt.show()


