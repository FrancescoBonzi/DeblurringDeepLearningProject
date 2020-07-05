import tensorflow as tf
from tensorflow.keras import datasets
import dill

from utilities import SSIMLoss, PSNR, load_REDs, split_REDs, print_dataset, inspect_report
from deblurring_REDs_configuration import *

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