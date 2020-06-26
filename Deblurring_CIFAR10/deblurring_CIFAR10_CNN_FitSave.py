import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.layers import Conv2D, Conv2DTranspose

import numpy as np
import os

from utilities import SSIMLoss, PSNR, build_dataset, print_dataset, inspect_report

tf.keras.backend.set_floatx('float64')
width = 32
height = 32

########################################
### DEFINITION OF THE NEURAL NETWORK ###
########################################

class DeblurringImagesModel(tf.keras.Model):
    def __init__(self):
        super(DeblurringImagesModel, self).__init__()
        input_shape = (None, height, width, 3)
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

#########################################################
### LOADING DATASET AND GENERATION OF BLURRING IMAGES ###
#########################################################

(train_images, _), (_, _) = datasets.cifar10.load_data()
''' Normalize pixel values to be between 0 and 1 '''
train_images = train_images / 255.0
train_images = train_images[1:1000,:,:,:]

train_blurred_images, train_rands = build_dataset(train_images)


#######################################################
#### DEFINITION OF THE CONVOLUTIONAL NEURAL NETWORK ###
#######################################################

model = DeblurringImagesModel()
model.build(input_shape=(len(train_images),height, width,3))
model.summary()

########################################
#### COMPILING AND FITTING THE MODEL ###
########################################

EPOCHS = 2

# The patience parameter is the amount of epochs to check for improvement
early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0001, patience=3)

model.compile(loss=SSIMLoss, optimizer=tf.keras.optimizers.Adam(), metrics=['mse', 'mae', PSNR])
report = model.fit(x=train_blurred_images, 
                   y=train_images, 
                   batch_size=32, 
                   epochs=EPOCHS, 
                   callbacks=[early_stop], 
                   validation_split=0.25)


inspect_report(report)
model.save("./models/CNN")