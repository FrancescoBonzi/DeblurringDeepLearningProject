import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

import tensorflow as tf
from tensorflow.keras.layers import ConvLSTM2D, Conv2DTranspose
from skimage.measure import compare_psnr, compare_ssim, compare_mse

tf.keras.backend.set_floatx('float64')

num_videos = 4
frame_per_video = 10
original_width = 1280
original_height = 720
scale_factor = 0.3


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = ConvLSTM2D(
            16, 3, activation='relu', data_format='channels_last', return_sequences=True)
        self.output_layer = Conv2DTranspose(3, 3, activation='relu')

    # x is (num_samples = num_videos, time_steps = frames_per_video, rows, cols, channels)
    def call(self, x):
        #print('SHAPE X init:', x.shape)
        x = self.conv1(x)
        #print('SHAPE X conv', x.shape)
        # Cast from 5 to 4 dimensions
        x = tf.reshape(x, [num_videos*x.shape[1],
                           x.shape[2], x.shape[3], x.shape[4]])
        #print('SHAPE X reshaped:', x.shape)
        x = self.output_layer(x)
        #print('SHAPE X deconv', x.shape)
        # Cast from 4 to 5 dimensions
        x = tf.reshape(x, [num_videos, frame_per_video,
                           x.shape[1], x.shape[2], x.shape[3]])
        #print('SHAPE X output', x.shape)
        return x


def plot_image(img, figsize=(10, 10)):
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.show()


def resize_image(img):
    width = int(img.shape[1] * scale_factor)
    height = int(img.shape[0] * scale_factor)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized


def load_videos(videos_directory):
    loaded_dataset = np.zeros((num_videos, frame_per_video, int(
        original_height*scale_factor), int(original_width*scale_factor), 3))
    videos = sorted(os.listdir(videos_directory))
    for i, video_directory in enumerate(videos):
        path_video = videos_directory + "/" + video_directory
        if os.path.isdir(path_video):
            frames = sorted(os.listdir(path_video))
            print("loading ", path_video, "...")
            for j, frame in enumerate(frames):
                path_frame = path_video + "/" + frame
                if os.path.isfile(path_frame) and path_frame.endswith(".png"):
                    img = cv2.imread(path_frame)
                    img = resize_image(img)
                    loaded_dataset[i-1, j-1, :, :, :] = img
    return loaded_dataset

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


blurred_videos_directory = "/Users/francescobonzi/Downloads/blurred_videos"
original_videos_directory = "/Users/francescobonzi/Downloads/original_videos"
loaded_blurred_dataset = load_videos(blurred_videos_directory)
loaded_original_dataset = load_videos(original_videos_directory)

print(loaded_blurred_dataset.shape)
print(loaded_original_dataset.shape)
# print(loaded_dataset[0,0,:,:,:])
#plot_image(loaded_blurred_dataset[0, 0, :, :, 0])


model = MyModel()
model.build(input_shape=(num_videos, frame_per_video, int(
    original_height*scale_factor), int(original_width*scale_factor), 3))
model.summary()

EPOCHS = 1

model.compile(loss=SSIMLoss, optimizer=tf.keras.optimizers.Adam(),
              metrics=['mse', 'mae', PSNR])
report = model.fit(x=loaded_blurred_dataset,
                   y=loaded_original_dataset,
                   batch_size=16,
                   epochs=EPOCHS)


blurred_test_videos_directory = "/Users/francescobonzi/Downloads/blurred_test_videos"
original_test_videos_directory = "/Users/francescobonzi/Downloads/original_test_videos"
loaded_blurred_test_dataset = load_videos(original_test_videos_directory)
loaded_original_test_dataset = load_videos(original_test_videos_directory)

prediction = model.predict(loaded_blurred_test_dataset)
for i in range(10):
  plt.figure(i)
  plt.imshow(prediction[0][i])
  plt.show()
