import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten

tf.keras.backend.set_floatx('float64')

import matplotlib.pyplot as plt
#from skimage.restoration import richardson_lucy
from skimage import transform
import numpy as np
import os
import random
import cv2
import dill
from math import cos, sin, pi
from utilities import print_dataset, rebuild_images, load_REDs, get_frames_per_video, get_num_videos, split_REDs, extract_from_report
from REDs_directories import *

num_patches_width = 16 
num_patches_height = 9 
original_width = 320
original_height = 180
width = int(original_width/num_patches_width)       #patches dimensions without considering overlapping
height = int(original_height/num_patches_height)
motion_kernel_size = 20
patches_size = motion_kernel_size

class KernelMotionEstimator(tf.keras.Model):
    def __init__(self):
        super(KernelMotionEstimator, self).__init__()
        self.conv1 = Conv2D(96, 5, activation='relu')
        self.maxpool2 = MaxPool2D(2, strides=2)
        self.conv3 = Conv2D(256, 3, activation='relu')
        self.maxpool4 = MaxPool2D(2, strides=2)
        self.flatten5 = Flatten()
        self.dense6 = Dense(1024, activation='relu')
        self.dense7 = Dense((motion_kernel_size//2+1)*6-5, activation='softmax')

    def call(self, input_img):
        x = self.conv1(input_img)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool4(x)
        x = self.flatten5(x)
        x = self.dense6(x)
        x = self.dense7(x)
        return x

# It extract num_patches patches from each image in the dataset.
# It returns an array of patches to use as training dataset.
def extract_patches(conv_img, num_patches, dim_patches):
    patches = []
    for i in range(num_patches):
        u = random.randint(0, conv_img.shape[0] - dim_patches)
        v = random.randint(0, conv_img.shape[1] - dim_patches)
        patches.append(conv_img[u:u+dim_patches, v:v+dim_patches])
    return np.array(patches, dtype=float)

def motion_kernel_generator(angle, length): 
    motion_kernel = np.zeros((motion_kernel_size, motion_kernel_size), dtype=np.float64)
    cx, cy = motion_kernel_size//2, motion_kernel_size//2
    dx, dy = round(length//2 * cos(angle)), round(length//2 * sin(angle))
    motion_kernel = cv2.line(motion_kernel, (int(cx-dx), int(cy-1+dy)), (int(cx-1+dx), int(cy-dy)), 1.0 )
    assert motion_kernel.sum() != 0
    motion_kernel /= motion_kernel.sum()
    return motion_kernel

# It builds the REDs dataset of motion blurred patches
# We suppose .DS_Store is not in the directories
def build_dataset_for_motion_blur(directory, num_videos, num_frames, num_patches=20, dim_patches=patches_size):
    dataset = []
    labels = []
    for i in range(num_videos):
        video = os.listdir(directory)[i]
        for j in range(num_frames):
            path = directory + "/" + os.listdir(directory)[i] + "/" + os.listdir(directory + "/" + os.listdir(directory)[i])[j]
            #print(path)
            img = cv2.imread(path) / 255
            for length in range(1, motion_kernel_size+1, 2):
                for k in range(6):
                    if k != 0 and length == 1: break # for the identity matrix we don't want repetitions
                    orientation = k*pi/6
                    motion_kernel = motion_kernel_generator(orientation, length)
                    conv_img = cv2.filter2D(img, -1, motion_kernel)
                    patches = extract_patches(conv_img, num_patches, dim_patches)
                    label = convert_to_label(length, k)
                    for p in range(num_patches):
                        dataset.append(patches[p])
                        labels.append(label)
    
    return np.array(dataset), np.array(labels, dtype=float)

def convert_to_motion_vector(label):
    count = 0
    for length in range(1, motion_kernel_size+1, 2):
        for k in range(6):
            if k != 0 and length == 1: break
            if count == label:
                return (length, k*pi/6)
            count += 1
    exit("Label Out of Bounds")

def convert_to_label(l, o):
    count = 0
    for length in range(1, motion_kernel_size+1, 2):
        for k in range(6):
            if o == k and l == length:
                return count
            if not (k != 0 and length == 1): 
                count += 1
    exit(f"Motion Vector not Recognised: l={l}, o={o}")

def rotate_image(patch, angle = 6, num_rotation = 6):
    rotated_patches = np.zeros(
        (num_rotation, height, width, 3))
    for k in range(num_rotation):
        rotated_patch = transform.rotate(patch, -k*angle, resize = False, mode = 'constant')
        rotated_patches[k, :, :, :] = rotated_patch
        #plt.imshow(rotated_patch)
        #plt.show()
    return rotated_patches


def get_motion_vector_prediction(model, patch):
    rotated_patches = rotate_image(patch)
    predictions = model.predict(rotated_patches) 
    #expected for each patch a vector of 61 elements containing the probability associated to each label

    max_probabilities_per_prediction = []
    max_label_per_prediction = []
    for k in range(6):
        max_probabilities_per_prediction.append(np.max(predictions[k])) #vector of the highest probabilities of each rotated_image
        #max_label_per_prediction.append(np.argmax(predictions[k])) #vector of the label associated to the highest probibilities'''
    index = np.argmax(max_probabilities_per_prediction) #it indicates which rotation has the highest probability
    label = np.argmax(predictions[index]) #it indicates the most probable motion vector for the selcted rotated_image

    length, orientation = convert_to_motion_vector(label) #orientation is a number in range(6) which indicate a multiple of pi/6
    selected_motion_orientation = orientation + index*pi/30 # pi/30 rad == 6 degrees

    return length, selected_motion_orientation

def motion_field_predictor(model, images):
    patches = split_REDs(images, test_num_videos, test_frames_per_video, num_patches_width, num_patches_height, height, width, 0)
    #show_image_with_label(patches[1], 1)
    predicted_motion_vectors = []
    predicted_motion_kernels = []
    for patch in patches:
        predicted_length, predicted_orientation = get_motion_vector_prediction(model, patch)
        predicted_motion_vectors.append((predicted_length, predicted_orientation))
        predicted_motion_kernels.append(motion_kernel_generator(predicted_length, predicted_orientation))
    return predicted_motion_kernels, predicted_motion_vectors


def show_image_with_label(img, label):
    plt.figure(12)
    plt.imshow(img)
    plt.xlabel(label)
    plt.show()


############
### MAIN ###
############

train_frames, train_labels = build_dataset_for_motion_blur(train_sharped_videos_directory, train_num_videos, train_frames_per_video, num_patches=5)
print(train_frames.shape)
print(train_labels.shape)
#show_image_with_label(train_frames[1], train_labels[1])

batch_size = 32
EPOCHS = 15

model = KernelMotionEstimator()
model.build(input_shape=(batch_size, patches_size, patches_size, 3))
model.summary()
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy())

report = model.fit(train_frames, train_labels, batch_size=batch_size, epochs=EPOCHS, validation_split=0.2)

### SAVE MODEL AND REPORT ###

filename = "./REDs/reports/KernelMotion/" + "epochs" + str(EPOCHS) + ".obj"
filehandler = open(filename, 'wb')
list = extract_from_report(report, [])
dill.dump(list, filehandler)
model.save("./REDs/models/KernelMotion/" + "epochs" + str(EPOCHS))

### TEST ###

test_frames, test_labels = build_dataset_for_motion_blur(test_sharped_videos_directory, train_num_videos, train_frames_per_video, num_patches=5)
predictions = model.predict(test_frames)
predicted_labels = []
for i in range(len(predictions)):
    predicted_labels.append(np.argmax(predictions[i]))
print(test_labels - predicted_labels)

'''test_blurred_REDs = load_REDs("./datasetREDs/test_blur", test_num_videos, test_frames_per_video, original_height, original_width)
#show_image_with_label(test_blurred_REDs[1], 1)
predicted_motion_kernels, predicted_motion_vectors = motion_field_predictor(model, test_blurred_REDs)
motion_fields = rebuild_images(predicted_motion_kernels, num_patches_height, num_patches_width, original_width, original_height, height, width, 0)
print_dataset(test_blurred_REDs, motion_fields)'''


