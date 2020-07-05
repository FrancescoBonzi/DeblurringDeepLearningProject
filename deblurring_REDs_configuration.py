import tensorflow as tf
from utilities import get_metrics, get_other_metrics, get_model, get_loss, SSIMLoss, PSNR, get_num_videos, get_frames_per_video

tf.keras.backend.set_floatx('float64')

demo = True
model_name = 'CNNBase_v1'
loss_name = 'SSIMLoss' # one of: mse, mae, SSIMLoss, PSNR
EPOCHS = 1

metrics = get_metrics(loss_name)
other_metrics = get_other_metrics(metrics)
model = get_model(model_name)
loss = get_loss(loss_name)


#######################################
### INITIALIZATION OF VARIABLES FOR ###
### RESIZING ORIGINAL IMAGES SHAPE  ###
#######################################

num_conv = 3              
num_patches_width = 4
num_patches_height = 2
original_width = 1280
original_height = 720
blurred_videos_directory = "./train_blur"
sharped_videos_directory = "./train_sharp"

width = int(original_width/num_patches_width)       #patches dimensions without considering overlapping
height = int(original_height/num_patches_height)
num_videos = get_num_videos(blurred_videos_directory, sharped_videos_directory)
frames_per_video = get_frames_per_video(blurred_videos_directory)



