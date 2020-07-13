import tensorflow as tf
from utilities import get_metrics, get_other_metrics, get_model, get_num_conv, get_loss, SSIMLoss, PSNR, get_num_videos, get_frames_per_video
from REDs_directories import *

tf.keras.backend.set_floatx('float64')

demo = True
model_name = 'CNNBase_v1'
loss_name = 'SSIMLoss' # one of: mse, mae, SSIMLoss, PSNR
EPOCHS = 3

metrics = get_metrics(loss_name)
other_metrics = get_other_metrics(metrics)
model = get_model(model_name)
loss = get_loss(loss_name)


#######################################
### INITIALIZATION OF VARIABLES FOR ###
### RESIZING ORIGINAL IMAGES SHAPE  ###
#######################################
    
num_patches_width = 4
num_patches_height = 2
original_width = 320
original_height = 180

if demo:
    train_blurred_videos_directory = "./datasetREDs/train_blur"
    train_sharped_videos_directory = "./datasetREDs/train_sharp"
    test_blurred_videos_directory = "./datasetREDs/test_blur"
    test_sharped_videos_directory = "./datasetREDs/test_sharp"

num_conv = get_num_conv(model_name)      
width = int(original_width/num_patches_width)       #patches dimensions without considering overlapping
height = int(original_height/num_patches_height)
num_videos = get_num_videos(train_blurred_videos_directory, train_sharped_videos_directory)
frames_per_video = get_frames_per_video(train_blurred_videos_directory)
frames_per_video = 50
test_num_videos = get_num_videos(test_blurred_videos_directory, test_sharped_videos_directory)
test_frames_per_video = get_frames_per_video(test_blurred_videos_directory)



