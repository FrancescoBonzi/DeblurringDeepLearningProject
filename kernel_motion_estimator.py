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
from math import sin, pi
from utilities import print_dataset, rebuild_images, load_REDs, get_frames_per_video, get_num_videos, split_REDs

num_patches_width = 16 
num_patches_height = 9 
original_width = 320
original_height = 180
width = int(original_width/num_patches_width)       #patches dimensions without considering overlapping
height = int(original_height/num_patches_height)
motion_kernel_size = 20
patches_size = 20

test_directory = "./datasetREDs/test_blur"
test_num_videos = len(os.listdir(test_directory))
test_frames_per_video = len(os.listdir(test_directory + "/" + os.listdir(test_directory)[1]))

class KernelMotionEstimator(tf.keras.Model):
    def __init__(self):
        super(KernelMotionEstimator, self).__init__()
        self.conv1 = Conv2D(96, 5, activation='relu')
        self.maxpool2 = MaxPool2D(2, strides=2)
        self.conv3 = Conv2D(256, 3, activation='relu')
        self.maxpool4 = MaxPool2D(2, strides=2)
        self.flatten5 = Flatten()
        self.dense6 = Dense(1024, activation='relu')
        self.dense7 = Dense(73, activation='softmax')

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

def motion_kernel_generator(angle, length): 
    l = motion_kernel_size  #size of motion kernel should be an even number
    p = int(l/2)-1
   
    motion_kernel = np.zeros((l,l))
    for x in range(int(length/2)+1):
        if angle == 0:
            motion_kernel[p, p+x] = 1 / length
            motion_kernel[p, p-x] = 1 / length
        elif angle == pi/2:
            motion_kernel[p+x, p] = 1 / length
            motion_kernel[p-x, p] = 1 / length
        elif angle < pi/2:
            motion_kernel[round(p-x*sin(angle)), p+x] = 1 / length
            motion_kernel[round(p+x*sin(angle)), p-x] = 1 / length
        else:
            motion_kernel[round(p+x*sin(angle)), p+x] = 1 / length
            motion_kernel[round(p-x*sin(angle)), p-x] = 1 / length
    return motion_kernel

# It builds the REDs dataset of motion blurred patches
# We suppose .DS_Store is not in the directories
def build_dataset_for_motion_blur(directory, num_patches=20, dim_patches= patches_size):
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
            for k in range(6):
                for length in range(1, motion_kernel_size+1, 2):
                    if k != 0 and length == 1: break # for the identity matrix we don't want repetitions
                    orientation = k*pi/6
                    motion_kernel = motion_kernel_generator(orientation, length)
                    conv_img = cv2.filter2D(img, -1, motion_kernel)
                    patches = extract_patches(conv_img, num_patches, dim_patches)
                    label = convert_to_label(length, orientation)
                    for patch in patches:
                        dataset.append(patch)
                        labels.append(label)
    return np.array(dataset), np.array(labels, dtype=float)

def convert_to_motion_vector(label):
    count = 0
    for orientation in range(6):
        for length in range(1, motion_kernel_size+1, 2):
            if orientation != 0 and length == 1: break
            if count == label:
                return (length, orientation)
            count += 1
    exit("Label Out of Bounds")

def convert_to_label(l, o):
    count = 0
    for orientation in range(6):
        for length in range(1, motion_kernel_size+1, 2):
            if orientation != 0 and length == 1: break
            if o == orientation and l == length:
                return count
            count += 1
    exit("Motion Vector not Recognised")

def rotate_image(image, angle = 6, num_rotation = 6):
    rotated_images = []
    for k in range(num_rotation):
        rotated_image = transform.rotate(image, -k*angle, resize = False, mode = 'constant')
        rotated_images.append(rotated_image)
        #plt.imshow(rotated_image)
        #plt.show()
    return np.array(rotated_images)


def get_motion_vector_prediction(model, image):
    rotated_images = rotate_image(image)
    predictions = model.predict(rotated_images) 
    #expected for each images a vector of 73 elements containing the probability associated to each label

    max_probabilities_per_prediction = []
    max_label_per_prediction = []
    for k in range(6):
        max_probabilities_per_prediction.append(np.max(predictions[k])) #vector of the highest probabilities of each rotated_image
        max_label_per_prediction.append(np.argmax(predictions[k])) #vector of the label associated to the highest probibilities'''
    index = np.argmax(max_probabilities_per_prediction) #it indicates which rotation has the highest probability
    label = np.argmax(predictions[index]) #it indicates the most probable motion vector for the selcted rotated_image

    length, orientation = convert_to_motion_vector(label)
    selected_motion_orientation = orientation + index*6

    return length, selected_motion_orientation

def motion_field_predictor(model, images):
    patches = split_REDs(images, test_num_videos, test_frames_per_video, num_patches_width, num_patches_height, height, width, 0)
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

train_frames, train_labels = build_dataset_for_motion_blur("./datasetREDs/train_sharp", num_patches=5)
print(train_frames.shape)
print(train_labels.shape)

batch_size = 32

model = KernelMotionEstimator()
model.build(input_shape=(batch_size, patches_size, patches_size, 3))
model.summary()
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['mse'])

history = model.fit(train_frames, train_labels, batch_size=batch_size, epochs=2, validation_split=0.25)

### TEST ###

test_blurred_REDs = load_REDs("./datasetREDs/test_blur", test_num_videos, test_frames_per_video, original_height, original_width)
predicted_motion_kernels, predicted_motion_vectors = motion_field_predictor(model, test_blurred_REDs)
motion_fields = rebuild_images(predicted_motion_kernels, num_patches_height, num_patches_width, original_width, original_height, height, width, 0)
print_dataset(test_blurred_REDs, motion_fields)


