
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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.keras.backend.set_floatx('float64')

num_videos = 2
frame_per_video = 10
num_conv = 3
num_patches_width = 4
num_patches_height = 2
patches = num_patches_width * num_patches_height
original_width = 1280
original_height = 720
width = int(original_width/num_patches_width)
height = int(original_height/num_patches_height)


class DeblurringImagesModel(tf.keras.Model):
    def __init__(self):
        super(DeblurringImagesModel, self).__init__()
        input_shape = (None, height+2*num_conv, width+2*num_conv, 3)
        output_shape = (None, height+2*num_conv, width+2*num_conv, 3)
        self.conv1 = Conv2D(32, 3, activation='relu', input_shape=input_shape)
        self.conv2 = Conv2D(64, 3, activation='relu')
        self.conv3 = Conv2D(128, 3, activation='relu')
        self.deconv1 = Conv2DTranspose(128, 3, activation='relu')
        self.deconv2 = Conv2DTranspose(64, 3, activation='relu')
        self.deconv3 = Conv2DTranspose(32, 3, activation='relu')
        self.output_layer = Conv2DTranspose(
            3, 3, activation='relu', padding='same')

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.output_layer(x)
        return x


"""# DATASET PREPROCESSING"""


def print_dataset(images, blurred_images, predicted_images=[], num=10):
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
        plt.xlabel("Blurred")
        if len(predicted_images) != 0:
            plt.subplot(num, num_plots_per_image, i*num_plots_per_image+3)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(predicted_images[i])
            plt.xlabel("Predicted")
    plt.show()


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

# Loss function


def SSIMLoss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))


def PSNR(y_true, y_pred):
    mse = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
    log10 = 2.303  # it is equivalent to ln(10)
    return 10 * tf.keras.backend.log(255 * 255 / mse) / log10


def inspect_report(report):
    # -----------------------------------------------------------
    # Retrieve a list of list results on training and test data
    # sets for each training epoch
    # -----------------------------------------------------------
    mae = report.history['mae']
    val_mae = report.history['val_mae']
    mse = report.history['mse']
    val_mse = report.history['val_mse']
    loss = report.history['loss']
    val_loss = report.history['val_loss']
    psnr = report.history['PSNR']
    val_psnr = report.history['val_PSNR']

    epochs = range(len(mae))  # Get number of epochs

    # ------------------------------------------------
    # Plot training and validation accuracy per epoch
    # ------------------------------------------------
    plt.plot(epochs,     mae)
    plt.plot(epochs, val_mae)
    plt.title('Training and validation mae')
    plt.show()

    # ------------------------------------------------
    # Plot training and validation accuracy per epoch
    # ------------------------------------------------
    plt.plot(epochs,     mse)
    plt.plot(epochs, val_mse)
    plt.title('Training and validation mse')
    plt.show()

    # ------------------------------------------------
    # Plot training and validation loss per epoch
    # ------------------------------------------------
    plt.plot(epochs,     loss)
    plt.plot(epochs, val_loss)
    plt.title('Training and validation loss/SSIM')
    plt.show()

    # ------------------------------------------------
    # Plot training and validation loss per epoch
    # ------------------------------------------------
    plt.plot(epochs,     psnr)
    plt.plot(epochs, val_psnr)
    plt.title('Training and validation PSNR')
    plt.show()


EPOCHS = 1

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
