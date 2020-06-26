# -*- coding: utf-8 -*-
"""DeblurringImagesBase.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1F_kGCyM8Jytvxtehz7jsv0rctXqdw-i7
"""

import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.layers import Conv2D, Conv2DTranspose

import matplotlib.pyplot as plt
import numpy as np
import random
import os

from scipy.ndimage import gaussian_filter
from skimage.measure import compare_psnr, compare_ssim, compare_mse

class DeblurringImagesModel(tf.keras.Model):
  def __init__(self):
    super(DeblurringImagesModel, self).__init__()
    input_shape = (None, 32, 32, 3)
    output_shape = (None, 32, 32, 3)
    self.conv1 = Conv2D(32, 3, activation='relu', input_shape=input_shape)
    self.conv2 = Conv2D(64, 3, activation='relu')
    self.conv3 = Conv2D(128, 3, activation='relu')
    self.deconv1 = Conv2DTranspose(128, 3, activation='relu')
    self.deconv2 = Conv2DTranspose(64, 3, activation='relu')
    self.deconv3 = Conv2DTranspose(32, 3, activation='relu')
    self.output_layer = Conv2DTranspose(3, 3, activation='relu', padding='same')
  
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

(train_images, _), (test_images, _) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

train_images = train_images[1:1000,:,:,:]

def build_dataset(images):
  blurred_images = []
  rands = []
  for i in range(images.shape[0]):
    rand = random.uniform(0,3)
    rands.append(rand)
    img_gauss = gaussian_filter(images[i], rand)
    blurred_images.append(img_gauss)
  blurred_images = np.array(blurred_images)
  #print("blurred_images: ", blurred_images.shape)
  #print("images: ", images.shape)
  return blurred_images, rands

train_blurred_images, train_rands = build_dataset(train_images)
test_blurred_images, test_rands = build_dataset(test_images)

#print_dataset(train_images, train_blurred_images, train_rands, num = 2)
print(train_images.shape)

"""# DEFINITION OF THE CONVOLUTIONAL NEURAL NETWORK"""

model = DeblurringImagesModel()
model.build(input_shape=(len(train_images),32,32,3))
model.summary()

### Loss function
def SSIMLoss(y_true, y_pred):
  return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))

def PSNR(y_true, y_pred):
  mse = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
  log10 = 2.303 # it is equivalent to ln(10)
  return 10 * tf.keras.backend.log(255 * 255 / mse) / log10

def inspect_report(report):
    #-----------------------------------------------------------
    # Retrieve a list of list results on training and test data
    # sets for each training epoch
    #-----------------------------------------------------------
    mae = report.history['mae']
    val_mae = report.history['val_mae']
    mse = report.history['mse']
    val_mse = report.history['val_mse']
    loss = report.history['loss']
    val_loss = report.history['val_loss']
    psnr = report.history['PSNR']
    val_psnr = report.history['val_PSNR']

    epochs = range(len(mae)) # Get number of epochs

    #------------------------------------------------
    # Plot training and validation accuracy per epoch
    #------------------------------------------------
    plt.plot  ( epochs,     mae )
    plt.plot  ( epochs, val_mae )
    plt.title ('Training and validation mae')
    plt.show()

     #------------------------------------------------
    # Plot training and validation accuracy per epoch
    #------------------------------------------------
    plt.plot  ( epochs,     mse )
    plt.plot  ( epochs, val_mse )
    plt.title ('Training and validation mse')
    plt.show()

    #------------------------------------------------
    # Plot training and validation loss per epoch
    #------------------------------------------------
    plt.plot  ( epochs,     loss )
    plt.plot  ( epochs, val_loss )
    plt.title ('Training and validation loss/SSIM')
    plt.show()

    #------------------------------------------------
    # Plot training and validation loss per epoch
    #------------------------------------------------
    plt.plot  ( epochs,     psnr )
    plt.plot  ( epochs, val_psnr )
    plt.title ('Training and validation PSNR')
    plt.show()

EPOCHS = 2

# The patience parameter is the amount of epochs to check for improvement
early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0001, patience=3)

train_blurred_images=train_blurred_images.reshape(len(train_blurred_images),32,32,3)
train_images=train_images.reshape(len(train_images),32,32,3)
model.compile(loss=SSIMLoss, optimizer=tf.keras.optimizers.Adam(), metrics=['mse', 'mae', PSNR])
report = model.fit(x=train_blurred_images, 
                   y=train_images, 
                   batch_size=32, 
                   epochs=EPOCHS, 
                   callbacks=[early_stop], 
                   validation_split=0.25)

#inspect_report(report)
model.save("/Users/francescobonzi/Desktop/model")

test_blurred_images=test_blurred_images.reshape(len(test_blurred_images),32,32,3)
test_images=test_images.reshape(len(test_images),32,32,3)
predicted_images = model.predict(test_blurred_images)


