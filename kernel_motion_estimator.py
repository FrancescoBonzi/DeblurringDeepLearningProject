import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten

tf.keras.backend.set_floatx('float64')

import matplotlib.pyplot as plt
from skimage.restoration import richardson_lucy
import numpy as np
import os
import random
import cv2
from math import sin, pi
from utilities import print_dataset

class KernelMotionEstimator(tf.keras.Model):
    def __init__(self):
        super(KernelMotionEstimator, self).__init__()
        self.conv1 = Conv2D(96, 7, activation='relu')
        self.maxpool2 = MaxPool2D(2, strides=2)
        self.conv3 = Conv2D(256, 5, activation='relu')
        self.maxpool4 = MaxPool2D(2, strides=2)
        self.flatten5 = Flatten()
        self.dense6 = Dense(1024, activation='relu')
        self.dense7 = Dense(78, activation='softmax')

    def call(self, input_img):
        x = self.conv1(input_img)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool4(x)
        x = self.flatten5(x)
        x = self.dense6(x)
        x = self.dense7(x)
        return x

def extract_patches(conv_img, num_patches, dim_patches):
    patches = []
    for i in range(num_patches):
        u = random.randint(0, conv_img.shape[0] - dim_patches)
        v = random.randint(0, conv_img.shape[1] - dim_patches)
        patches.append(conv_img[u:u+dim_patches, v:v+dim_patches])
    return np.array(patches, dtype=float)

def motion_kernel_generator(k, length): #k indicate which multiple of 30°
    l = length
    p = int(l/2)
   
    motion_kernel = np.zeros((l,l))
    for x in range(int(length/2)+1):
        if k == 0:
            motion_kernel[p, p+x] = 1 / length
            motion_kernel[p, p-x] = 1 / length
        elif k == p:
            motion_kernel[p+x, p] = 1 / length
            motion_kernel[p-x, p] = 1 / length
        elif k>0 and k<3:
            motion_kernel[round(p-x*sin(k*pi/6)), p+x] = 1 / length
            motion_kernel[round(p+x*sin(k*pi/6)), p-x] = 1 / length
        else:
            motion_kernel[round(p+x*sin(k*pi/6)), p+x] = 1 / length
            motion_kernel[round(p-x*sin(k*pi/6)), p-x] = 1 / length
    return motion_kernel

# It builds the REDs dataset of motion blurred patches
# We suppose .DS_Store is not in the directories
def build_dataset_for_motion_blur(directory, num_patches=20, dim_patches=30):
    dataset = []
    labels = []
    num_videos = len(os.listdir(directory))
    for i in range(num_videos):
        num_frames = len(os.listdir(directory + "/" + os.listdir(directory)[i]))
        video = os.listdir(directory)[i]
        for j in range(num_frames):
            path = directory + "/" + os.listdir(directory)[i] + "/" + os.listdir(directory + "/" + os.listdir(directory)[i])[j]
            print(path)
            img = cv2.imread(path)
            for orientation in range(6):
                for length in range(1, 26, 2):
                    motion_kernel = motion_kernel_generator(orientation, length)
                    conv_img = cv2.filter2D(img, -1, motion_kernel)
                    patches = extract_patches(conv_img, num_patches, dim_patches)
                    label = convert_to_label(orientation, length)
                    for patch in patches:
                        dataset.append(patch)
                        labels.append(label)
    return np.array(dataset), np.array(labels, dtype=float)

def convert_to_motion_vector(label):
    count = 0
    for orientation in range(6):
        for length in range(1, 26, 2):
            if count == label:
                return (orientation, length)
            count += 1
    exit("Label Out of Bounds")

def convert_to_label(o, l):
    count = 0
    for orientation in range(6):
        for length in range(1, 26, 2):
            if o == orientation and l == length:
                return count
            count += 1
    exit("Motion Vector not Recognised")

train_frames, train_labels = build_dataset_for_motion_blur("./datasetREDs/train_sharp", num_patches=5)
print(train_frames.shape)
print(train_labels.shape)

def show_image_with_label(img, label)
    plt.figure(12)
    plt.imshow(img)
    plt.xlabel(label)
    plt.show()

def show_images_with_label(images, labels, starting_from=0)

batch_size = 32

model = KernelMotionEstimator()
model.build(input_shape=(batch_size, 30, 30, 3))
model.summary()
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['mse'])

history = model.fit(train_frames, train_labels, batch_size=batch_size, epochs=2, validation_split=0.25)

### TEST ###

test_frames, test_labels = build_dataset_for_motion_blur("./datasetREDs/test_sharp", num_patches=5)
predictions = model.predict(test_frames)
for i in range(10):
    print(i, " --> predicted:", np.argmax(predictions[i]), ", real: ", int(test_labels[i]))